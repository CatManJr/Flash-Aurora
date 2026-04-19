#!/usr/bin/env python3
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Profile a single :class:`Swin3DTransformerBlock` to localize bottlenecks (SDPA vs GEMM vs
AdaLN vs layout) without encoder/decoder/time-MLP noise from the full backbone.

On CUDA, KPI buckets sum only ``aten::`` and memcpy rows so GEMM/FMHA **kernel** lines are not
double-counted with their parent ATen ops. SDPA focus uses ``_efficient_attention_forward`` etc.,
not only ``scaled_dot_product`` (PyTorch naming varies by backend).

Run from the repository root::

    uv run python aurora/profiling_swin3d_block.py
    uv run python aurora/profiling_swin3d_block.py --preset aurora --patch-h 180 --patch-w 360
    uv run python aurora/profiling_swin3d_block.py --shifted --report-out profiling/swin3d_block.md
    uv run python aurora/profiling_swin3d_block.py --use-triton-layout --use-triton-adaln
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse table helpers from the backbone profiler (same torch.profiler aggregation model).
from profiling_swin3d import (
    _extract_addmm_stats,
    _shorten,
    _top_ops_ms,
)


def _bucket_swin3d_block(name: str) -> str:
    """Coarse buckets tuned for one Swin3D block (W-MSA / SW-MSA + AdaLN + MLP)."""
    n = name.lower()
    if "memcpy" in n or "dtoh" in n or "htod" in n or "memset" in n:
        return "memcpy"
    if (
        "scaled_dot_product" in n
        or "sdpa" in n
        or "efficient_attention" in n
        or "flash" in n
        or "fmha" in n
    ):
        return "attention (SDPA / FMHA)"
    if (
        "addmm" in n
        or "aten::mm" in n
        or "magma" in n
        or "cutlass" in n
        or n.endswith("::mm")
        or "cublas" in n
        or "gemm" in n
    ):
        return "GEMM (Linear / matmul)"
    if "native_layer_norm" in n or "layer_norm" in n:
        return "LayerNorm"
    if "gelu" in n:
        return "GELU"
    if "silu" in n:
        return "SiLU (AdaLN modulation MLP)"
    if "aten::copy_" in n or "copy_kernel" in n or "direct_copy" in n:
        return "copy / scatter"
    if "roll" in n:
        return "roll / pad / window layout"
    if "triton" in n or "compiledfxgraph" in n or ("compiled" in n and "fx" in n):
        return "triton / torch.compile"
    if (
        "elementwise" in n
        or "::mul" in n
        or "::add" in n
        or "::div" in n
        or "where" in n
        or "masked_fill" in n
    ):
        return "elementwise (other)"
    return "other"


def _include_row_in_cuda_bucket_sum(key: str) -> bool:
    """Avoid double-counting: PyTorch lists both ``aten::addmm`` and ``magma_*`` / ``fmha_*`` with
    nearly identical self-CUDA; summing all rows inflates totals (~2x). Buckets use ATen + memcpy
    rows only."""
    k = str(key)
    if k.startswith("aten::"):
        return True
    if "memcpy" in k.lower():
        return True
    return False


def _aggregate_block_kpis(prof: Any, *, use_cuda: bool) -> tuple[dict[str, float], float]:
    buckets: dict[str, float] = {}
    total_ms = 0.0
    for e in prof.key_averages():
        key = str(e.key)
        if use_cuda and not _include_row_in_cuda_bucket_sum(key):
            continue
        if use_cuda:
            t_us = float(
                getattr(e, "self_cuda_time_total", 0)
                or getattr(e, "self_device_time_total", 0)
                or 0
            )
        else:
            t_us = float(getattr(e, "self_cpu_time_total", 0) or 0)
        if t_us <= 0:
            continue
        t_ms = t_us / 1000.0
        total_ms += t_ms
        b = _bucket_swin3d_block(key)
        buckets[b] = buckets.get(b, 0.0) + t_ms
    return buckets, total_ms


def _extract_sdpa_stats(prof: Any, *, use_cuda: bool) -> tuple[int, float]:
    """ATen-level SDPA / memory-efficient attention (matches table rows like ``_efficient_attention_forward``)."""
    calls = 0
    self_ms = 0.0
    for e in prof.key_averages():
        key = str(e.key)
        kl = key.lower()
        if not key.startswith("aten::"):
            continue
        if (
            "scaled_dot_product" not in kl
            and "sdpa" not in kl
            and "efficient_attention" not in kl
        ):
            continue
        calls += int(getattr(e, "count", 0) or 0)
        if use_cuda:
            t_us = float(
                getattr(e, "self_cuda_time_total", 0)
                or getattr(e, "self_device_time_total", 0)
                or 0
            )
        else:
            t_us = float(getattr(e, "self_cpu_time_total", 0) or 0)
        self_ms += t_us / 1000.0
    return calls, self_ms


def _print_bottleneck_summary(
    buckets: dict[str, float], total_ms: float, *, use_cuda: bool
) -> list[str]:
    lines = [
        "",
        "--- Swin3D block bottleneck (CUDA: ATen+memcpy rows only — no GEMM/FMHA kernel double-count) ---"
        if use_cuda
        else "--- Swin3D block bottleneck (aggregate self-time, all profiler rows) ---",
    ]
    if total_ms <= 0:
        lines.append("  (no CUDA/CPU self-time recorded)")
        return lines
    for b in sorted(buckets.keys(), key=lambda k: -buckets[k]):
        ms = buckets[b]
        lines.append(f"  {b}: {ms:.2f} ms ({100.0 * ms / total_ms:.1f}%)")
    lines.append(f"  total: {total_ms:.2f} ms")
    return lines


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from aurora.model.swin3d import Swin3DTransformerBlock

    p = argparse.ArgumentParser(
        description="Profile one Swin3DTransformerBlock (isolate block-level bottlenecks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=("small", "aurora", "none"),
        default="small",
        help=(
            "small: dim=256, heads=4 (matches AuroraSmall-like stage 0); "
            "aurora: dim=512, heads=8 (full Aurora 1.3B stage 0). "
            "'none' uses --dim / --num-heads explicitly."
        ),
    )
    p.add_argument("--dim", type=int, default=256, help="Block channel dim (ignored if preset≠none).")
    p.add_argument("--num-heads", type=int, default=4, help="Attention heads (ignored if preset≠none).")
    p.add_argument(
        "--time-dim",
        type=int,
        default=0,
        help="AdaLN context dim (default: same as dim).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--latent-levels", type=int, default=4, help="C in patch_res (C, H, W).")
    p.add_argument("--patch-h", type=int, default=32)
    p.add_argument("--patch-w", type=int, default=64)
    p.add_argument(
        "--window-size",
        type=int,
        nargs=3,
        default=(2, 6, 12),
        metavar=("Wc", "Wh", "Ww"),
        help="3D window size (default 2 6 12, Aurora).",
    )
    p.add_argument(
        "--shifted",
        action="store_true",
        help="Use shifted-window (SW-MSA) attention mask path; default is W-MSA (shift 0).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--table-rows", type=int, default=40)
    p.add_argument("--plot-out", type=str, default="")
    p.add_argument("--plot-top", type=int, default=30)
    p.add_argument("--report-out", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--autocast",
        action="store_true",
        help="Run block under BF16 autocast (not default FP32).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile this block only (fixed shape recommended).",
    )
    p.add_argument(
        "--use-triton-layout",
        action="store_true",
        help="Fused roll/pad/window Triton path (CUDA float32).",
    )
    p.add_argument(
        "--use-triton-adaln",
        action="store_true",
        help="Fused AdaLN Triton path (CUDA float32).",
    )
    p.add_argument(
        "--use-triton-mlp",
        action="store_true",
        help="Triton GELU in MLP (CUDA float32, eval).",
    )
    p.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA in attention (extra GEMMs vs baseline).",
    )
    p.add_argument(
        "--use-lora-merged",
        action="store_true",
        help="Merge LoRA into linear at inference (requires --use-lora).",
    )
    args = p.parse_args()

    if args.preset == "small":
        dim, num_heads = 256, 4
    elif args.preset == "aurora":
        dim, num_heads = 512, 8
    else:
        dim, num_heads = args.dim, args.num_heads

    time_dim = args.time_dim if args.time_dim > 0 else dim
    ws = tuple(args.window_size)

    C, H, W = args.latent_levels, args.patch_h, args.patch_w
    if C % ws[0] != 0:
        raise SystemExit(f"latent-levels ({C}) must be divisible by window[0] ({ws[0]}).")
    L = C * H * W

    shift = (ws[0] // 2, ws[1] // 2, ws[2] // 2) if args.shifted else (0, 0, 0)

    block = Swin3DTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        time_dim=time_dim,
        window_size=ws,
        shift_size=shift,
        mlp_ratio=4.0,
        drop_path=0.0,
        use_triton_layout=args.use_triton_layout,
        use_triton_adaln=args.use_triton_adaln,
        use_triton_mlp=args.use_triton_mlp,
        use_lora=args.use_lora,
        use_lora_merged_inference=args.use_lora_merged,
    )
    block.eval()
    block.to(args.device)

    if args.compile:
        block = torch.compile(block, dynamic=False)

    x = torch.randn(args.batch_size, L, dim, device=args.device, dtype=torch.float32)
    c = torch.randn(args.batch_size, time_dim, device=args.device, dtype=torch.float32)
    patch_res = (C, H, W)

    def run_once() -> None:
        ctx: Any
        if args.autocast and args.device.startswith("cuda"):
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif args.autocast and torch.xpu.is_available() and args.device == "xpu":
            ctx = torch.autocast(device_type="xpu", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            with torch.inference_mode():
                _ = block(x, c, patch_res, rollout_step=0, warped=True)

    print(
        f"[config] preset={args.preset}, dim={dim}, heads={num_heads}, time_dim={time_dim}, "
        f"patch_res={patch_res}, L={L}, window={ws}, shift={shift}, "
        f"autocast={args.autocast}, compile={args.compile}, "
        f"use_triton_layout={args.use_triton_layout}, use_triton_adaln={args.use_triton_adaln}, "
        f"use_triton_mlp={args.use_triton_mlp}, use_lora={args.use_lora}, "
        f"use_lora_merged={args.use_lora_merged}"
    )

    if not args.device.startswith("cuda") and torch.cuda.is_available():
        warnings.warn(f"CUDA is available but device={args.device!r}; profiling may be CPU-only.")

    for _ in range(args.warmup):
        run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    timing_line = ""
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(args.repeat):
            run_once()
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1)
        timing_line = f"GPU: {ms:.2f} ms for {args.repeat} iters -> {ms / args.repeat:.3f} ms/iter"
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"[timing] {timing_line}")
        print(
            f"[mem] peak CUDA allocated: {peak_alloc:.1f} MB, "
            f"peak reserved: {peak_reserved:.1f} MB"
        )
    else:
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            run_once()
        ms = (time.perf_counter() - t0) * 1e3
        timing_line = f"CPU: {ms:.2f} ms for {args.repeat} iters"
        print(f"[timing] {timing_line}")

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        acc_events=True,
    ) as prof:
        for _ in range(args.repeat):
            run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    sort_by = "self_cuda_time_total" if args.device.startswith("cuda") else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=args.table_rows)
    print("\n" + table)

    use_cuda = args.device.startswith("cuda")
    names, tms = _top_ops_ms(prof, use_cuda=use_cuda, top_k=args.plot_top)
    buckets, total_kpi_ms = _aggregate_block_kpis(prof, use_cuda=use_cuda)
    for line in _print_bottleneck_summary(buckets, total_kpi_ms, use_cuda=use_cuda):
        print(line)

    addmm_calls, addmm_self_ms = _extract_addmm_stats(prof, use_cuda=use_cuda)
    sdpa_calls, sdpa_self_ms = _extract_sdpa_stats(prof, use_cuda=use_cuda)
    st = "self_cuda" if use_cuda else "self_cpu"
    print(
        f"\n[focus] ATen SDPA / efficient_attention calls={sdpa_calls}, {st}≈{sdpa_self_ms:.3f} ms"
    )
    print(f"[focus] aten::addmm calls={addmm_calls}, {st}≈{addmm_self_ms:.3f} ms")

    if args.plot_out:
        timing_safe = timing_line.replace("\n", " ")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(names)
        fig_h = max(4.0, 0.38 * n + 1.2)
        fig, ax = plt.subplots(figsize=(10.5, fig_h), layout="constrained")
        y = range(n)
        ax.barh(y, tms, color="#8d4c2c", alpha=0.9)
        ax.set_yticks(list(y))
        ax.set_yticklabels([_shorten(x) for x in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Self time (ms)")
        ax.set_title("Swin3DTransformerBlock — top ops (self CUDA)")
        fig.text(0.02, 0.02, timing_safe, fontsize=8, family="monospace", color="#333333")
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {Path(args.plot_out).resolve()}")

    if args.report_out:
        lines = [
            "# Swin3DTransformerBlock profiling",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Torch: {torch.__version__}",
            f"- Config: preset={args.preset}, dim={dim}, heads={num_heads}, patch_res={patch_res}, "
            f"window={ws}, shift={shift}, autocast={args.autocast}, compile={args.compile}, "
            f"Triton layout/AdaLN/MLP={args.use_triton_layout}/{args.use_triton_adaln}/{args.use_triton_mlp}",
            "",
            "## Timer",
            "",
            timing_line,
            "",
            "## Bottleneck buckets (block-local)",
            "",
            "| Bucket | Self (ms) | % |",
            "| --- | ---: | ---: |",
        ]
        if total_kpi_ms > 0:
            for b in sorted(buckets.keys(), key=lambda k: -buckets[k]):
                ms = buckets[b]
                pct = 100.0 * ms / total_kpi_ms
                lines.append(f"| {b} | {ms:.3f} | {pct:.1f} |")
        lines.extend(
            [
                "",
                "## Focus",
                "",
                f"- ATen `scaled_dot_product*` / `_efficient_attention_forward`: calls≈{sdpa_calls}, "
                f"self-time≈{sdpa_self_ms:.3f} ms",
                f"- `aten::addmm`: calls={addmm_calls}, self-time≈{addmm_self_ms:.3f} ms",
                "",
                "## Top operators",
                "",
                "| Rank | Operator | Self (ms) |",
                "| ---: | --- | ---: |",
            ]
        )
        for i, (n, t) in enumerate(zip(names, tms, strict=True), start=1):
            lines.append(f"| {i} | {n.replace('|', '\\|')} | {t:.3f} |")
        lines.extend(["", "## Full profiler table", "", "```text", table.rstrip(), "```", ""])
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] {Path(args.report_out).resolve()}")


if __name__ == "__main__":
    main()
