#!/usr/bin/env python3
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Profile only :class:`Swin3DTransformerBackbone` to spot fusion opportunities.

Uses the same backbone defaults as :class:`AuroraSmallPretrained` unless overridden.
Synthetic tokens ``(B, L, D)`` match encoder output going into the backbone.

Run from the repository root::

    uv run python aurora/profiling_swin3d.py
    uv run python aurora/profiling_swin3d.py --preset baseline
    uv run python aurora/profiling_swin3d.py --batch-size 4 --patch-h 32 --patch-w 64
    uv run python aurora/profiling_swin3d.py --plot-out profiling/swin3d.png --report-out profiling/swin3d.md
    uv run python aurora/profiling_swin3d.py --compile --autocast-backbone
    uv run python aurora/profiling_swin3d.py --preset baseline --use-triton-layout --use-triton-adaln
    # D2+D3 three-way (baseline vs layout+AdaLN vs +workspace pool), stress load:
    uv run python aurora/profiling_swin3d.py --compare-d2d3 --preset stress --compare-report-out profiling/swin3d_d2d3_stress.md
    # If compile stalls or sympy warnings: omit --compile-dynamic (default fixed-shape).

``--preset baseline`` fixes batch=1 and patch_res=(4, 32, 64), matching :file:`profiling/swin3d.md`.
``--preset stress`` sets batch=16, patch_res=(4,16,32) (``L``=32768), warmup=8, repeat=16 - same tokens/step
as baseline (8192), heavier than default without the VRAM of ``stress-heavy``.
``--preset stress-heavy`` is batch=8 with patch_res=(4,32,64) (``L``=8192); needs very large VRAM.
Before each timed run, the script calls ``gc.collect`` and ``torch.cuda.empty_cache`` (disable with
``--no-empty-cache-between-runs``) so compare modes get fairer peak memory stats.
Triton flags require CUDA float32 (omit ``--autocast-backbone`` for a fair compare with hand kernels).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _top_ops_ms(
    prof: Any,
    *,
    use_cuda: bool,
    top_k: int,
) -> tuple[list[str], list[float]]:
    rows: list[tuple[str, float]] = []
    for e in prof.key_averages():
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
        rows.append((str(e.key), t_us / 1000.0))
    rows.sort(key=lambda x: -x[1])
    top = rows[:top_k]
    return [n for n, _ in top], [t for _, t in top]


def _shorten(s: str, max_len: int = 72) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _save_plot(
    names: list[str],
    times_ms: list[float],
    out_path: Path,
    timing_line: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    fig_h = max(4.0, 0.38 * n + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), layout="constrained")
    y = range(n)
    ax.barh(y, times_ms, color="#2c5f8d", alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_shorten(x) for x in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Self time (ms)")
    ax.set_title("Swin3DTransformerBackbone — top ops (Self CUDA)")
    fig.text(0.02, 0.02, timing_line, fontsize=8, family="monospace", color="#333333")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _categorize_for_fusion(names: list[str], times_ms: list[float]) -> None:
    """Print rough buckets to guide Triton / fusion work (library vs fuse-candidate)."""
    total = sum(times_ms) or 1.0

    def bucket(name: str) -> str:
        n = name.lower()
        if "efficient_attention" in n or "fmha" in n or "flash" in n:
            return "attention (library SDPA / FMHA)"
        if "addmm" in n or "magma" in n or "cutlass" in n or n.endswith("::mm") or "::mm " in n:
            return "GEMM (cuBLAS / CUTLASS — usually do not replace)"
        if "layer_norm" in n or "native_layer_norm" in n:
            return "LayerNorm"
        if "roll" in n:
            return "roll / shift (fusion candidate with neighbors)"
        if "copy_" in n or "copy" in n and "kernel" in n:
            return "copy / layout (reduce or fuse with compute)"
        if "command buffer" in n:
            return "Command Buffer Full (driver / many launches)"
        if "gelu" in n or "elementwise" in n or "::mul" in n or "::add" in n:
            return "elementwise / activations (fusion candidate chains)"
        return "other"

    sums: dict[str, float] = {}
    for name, t in zip(names, times_ms, strict=True):
        b = bucket(name)
        sums[b] = sums.get(b, 0.0) + t

    print("\n--- Fusion-oriented buckets (top-k rows only, approximate) ---")
    for b, s in sorted(sums.items(), key=lambda x: -x[1]):
        print(f"  {b}: {s:.2f} ms ({100.0 * s / total:.1f}% of top-k sum)")


def _bucket_full_profile(name: str) -> str:
    """Single bucket per operator row (full profiler table, for KPI ratios)."""
    n = name.lower()
    if "memcpy" in n or "dtoh" in n or "htod" in n or "memset" in n:
        return "memcpy"
    if "aten::copy_" in n or "copy_kernel" in n or "direct_copy" in n:
        return "copy_layout"
    if "efficient_attention" in n or "fmha" in n or "flash" in n or "sdpa" in n:
        return "attention"
    if (
        "addmm" in n
        or "magma" in n
        or "cutlass" in n
        or n.endswith("::mm")
        or "aten::mm" in n
        or "gemm" in n
        or "cublas" in n
    ):
        return "gemm"
    if "layer_norm" in n or "native_layer_norm" in n:
        return "layer_norm"
    if "roll" in n:
        return "roll_pad_layout"
    if "triton" in n or "compiledfxgraph" in n or ("compiled" in n and "fx" in n):
        return "triton_compile"
    if "gelu" in n or "elementwise" in n or "::mul" in n or "::add" in n or "silu" in n:
        return "elementwise"
    return "other"


def _aggregate_full_table_kpis(prof: Any, *, use_cuda: bool) -> tuple[dict[str, float], float]:
    """Sum self-time over *all* profiler rows into coarse buckets (not only top-k)."""
    buckets: dict[str, float] = {}
    total_ms = 0.0
    for e in prof.key_averages():
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
        b = _bucket_full_profile(str(e.key))
        buckets[b] = buckets.get(b, 0.0) + t_ms
    return buckets, total_ms


def _kpi_console_and_markdown(
    buckets: dict[str, float], total_ms: float, *, use_cuda: bool
) -> tuple[list[str], list[str]]:
    """Human-readable KPI lines for stdout and a markdown section for --report-out."""
    if total_ms <= 0:
        return (["(no aggregate self-time)"], ["(no aggregate self-time)"])
    label = "Self CUDA" if use_cuda else "Self CPU"
    console = [
        "",
        f"--- Full-table KPI buckets ({label} self-time, all rows) ---",
    ]
    md: list[str] = [
        "",
        f"## Aggregate KPIs ({label}, all profiler rows)",
        "",
        "| Bucket | Self (ms) | % of total |",
        "| --- | ---: | ---: |",
    ]
    for b in sorted(buckets.keys(), key=lambda k: -buckets[k]):
        ms = buckets[b]
        pct = 100.0 * ms / total_ms
        console.append(f"  {b}: {ms:.2f} ms ({pct:.1f}%)")
        md.append(f"| {b} | {ms:.3f} | {pct:.1f} |")
    console.append(f"  total: {total_ms:.2f} ms")
    md.extend(["", f"- **Total** {label.lower()}: {total_ms:.2f} ms", ""])
    return console, md


@dataclass
class _RunResult:
    """One profiling run. ``peak_*_mb`` are from the timed forward loop (after ``reset_peak_memory_stats``)."""

    timing_line: str
    names: list[str]
    tms: list[float]
    table: str
    buckets: dict[str, float]
    total_kpi_ms: float
    peak_mem_mb: float | None  # max_memory_allocated (MB)
    peak_reserved_mb: float | None  # max_memory_reserved, caching allocator (MB)
    use_cuda: bool
    addmm_calls: int
    addmm_self_ms: float


def _randomize_lora_weights(backbone: Any, *, seed: int) -> None:
    """Give non-zero LoRA delta (default init has ``lora_B`` = 0)."""
    import math

    import torch
    import torch.nn as nn

    from aurora.model.lora import LoRARollout

    torch.manual_seed(seed)
    for m in backbone.modules():
        if isinstance(m, LoRARollout):
            for lora in m.loras:
                nn.init.kaiming_uniform_(lora.lora_A, a=math.sqrt(5))
                nn.init.kaiming_uniform_(lora.lora_B, a=math.sqrt(5))


def _extract_addmm_stats(prof: Any, *, use_cuda: bool) -> tuple[int, float]:
    """Return total ``aten::addmm`` calls and summed self-time in ms."""
    calls = 0
    self_ms = 0.0
    for e in prof.key_averages():
        if str(e.key) != "aten::addmm":
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


def _cuda_reset_before_run(args: argparse.Namespace) -> None:
    """Release Python refs and return cached GPU blocks to the driver before each timed run."""
    if not args.device.startswith("cuda"):
        return
    if getattr(args, "no_empty_cache_between_runs", False):
        return
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[mem] reset: gc.collect + torch.cuda.empty_cache()")


def _run_profile_once(
    args: argparse.Namespace,
    *,
    use_triton_layout: bool,
    use_triton_adaln: bool,
    use_triton_mlp: bool,
    use_lora_merged_inference: bool,
    use_workspace_pool: bool = False,
    init_state_dict: dict[str, Any] | None = None,
    report_out: str = "",
    report_config_extra: str = "",
) -> _RunResult:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from aurora.model.swin3d import Swin3DTransformerBackbone
    from aurora.model.workspace_pool import InferenceWorkspacePool

    _cuda_reset_before_run(args)

    C, H, W = args.latent_levels, args.patch_h, args.patch_w
    L = C * H * W

    workspace_pool = InferenceWorkspacePool() if use_workspace_pool else None
    backbone = Swin3DTransformerBackbone(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode="single",
        use_triton_layout=use_triton_layout,
        use_triton_adaln=use_triton_adaln,
        use_triton_mlp=use_triton_mlp,
        use_lora_merged_inference=use_lora_merged_inference,
        workspace_pool=workspace_pool,
    )
    if init_state_dict is not None:
        backbone.load_state_dict(init_state_dict)
    backbone.eval()
    backbone.to(args.device)

    if getattr(args, "randomize_lora", False) and init_state_dict is None:
        _randomize_lora_weights(backbone, seed=int(getattr(args, "lora_random_seed", 0)))
        print(
            f"[lora] randomized LoRA lora_A/lora_B (non-zero delta), "
            f"seed={getattr(args, 'lora_random_seed', 0)}"
        )

    if args.compile:
        ck_kw: dict[str, Any] = {
            "dynamic": args.compile_dynamic,
            "mode": args.compile_mode,
        }
        backbone = torch.compile(backbone, **ck_kw)

    ws = backbone.window_size
    if C % ws[0] != 0:
        raise SystemExit(f"latent-levels ({C}) must be divisible by window_size[0] ({ws[0]}).")

    x = torch.randn(args.batch_size, L, 256, device=args.device, dtype=torch.float32)
    patch_res = (C, H, W)
    lead = timedelta(hours=6)

    def run_once() -> None:
        ctx: Any
        if args.autocast_backbone and args.device.startswith("cuda"):
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif args.autocast_backbone and torch.xpu.is_available() and args.device == "xpu":
            ctx = torch.autocast(device_type="xpu", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            _ = backbone(x, lead_time=lead, rollout_step=0, patch_res=patch_res)

    print(
        f"[config] AuroraSmall-like Swin3D, preset={args.preset}, batch={args.batch_size}, "
        f"patch_res={patch_res}, L={L}, autocast_backbone={args.autocast_backbone}, "
        f"compile={args.compile}"
        + (
            f", compile_dynamic={args.compile_dynamic}, compile_mode={args.compile_mode}"
            if args.compile
            else ""
        )
        + f", use_triton_layout={use_triton_layout}, use_triton_adaln={use_triton_adaln}, "
        f"use_triton_mlp={use_triton_mlp}, use_lora_merged_inference={use_lora_merged_inference}, "
        f"use_workspace_pool={use_workspace_pool}"
        + (
            f", randomize_lora=True, lora_random_seed={getattr(args, 'lora_random_seed', 0)}"
            if getattr(args, "randomize_lora", False)
            else ", randomize_lora=False"
        )
    )

    for _ in range(args.warmup):
        run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    timing_line = ""
    peak_mem_mb: float | None = None
    peak_reserved_mb: float | None = None
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
        timing_line = f"GPU: {ms:.2f} ms for {args.repeat} forwards -> {ms / args.repeat:.2f} ms/forward"
        print(f"[timing] {timing_line}")
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
        print(
            f"[mem] peak CUDA allocated: {peak_mem_mb:.1f} MB, "
            f"peak reserved: {peak_reserved_mb:.1f} MB"
        )
    else:
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            run_once()
        ms = (time.perf_counter() - t0) * 1e3
        timing_line = f"CPU: {ms:.2f} ms for {args.repeat} forwards"
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
    _categorize_for_fusion(names, tms)

    buckets, total_kpi_ms = _aggregate_full_table_kpis(prof, use_cuda=use_cuda)
    addmm_calls, addmm_self_ms = _extract_addmm_stats(prof, use_cuda=use_cuda)
    kpi_console, kpi_md = _kpi_console_and_markdown(buckets, total_kpi_ms, use_cuda=use_cuda)
    for line in kpi_console:
        print(line)
    print(f"  addmm: calls={addmm_calls}, self={addmm_self_ms:.3f} ms")

    if report_out:
        lines = [
            "# Swin3DTransformerBackbone profiling",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Torch: {torch.__version__}",
            f"- Config: preset={args.preset}, batch={args.batch_size}, patch_res=({C}, {H}, {W}), "
            f"L={L}, autocast_backbone={args.autocast_backbone}, "
            f"use_triton_layout={use_triton_layout}, use_triton_adaln={use_triton_adaln}, "
            f"use_triton_mlp={use_triton_mlp}, "
            f"use_lora_merged_inference={use_lora_merged_inference}, "
            f"use_workspace_pool={use_workspace_pool}"
            + (
                f", randomize_lora=True, lora_random_seed={getattr(args, 'lora_random_seed', 0)}"
                if getattr(args, "randomize_lora", False)
                else ""
            )
            + f"{report_config_extra}",
            "",
            "## Timer",
            "",
            timing_line,
            "",
        ]
        if peak_mem_mb is not None:
            assert peak_reserved_mb is not None
            lines.extend(
                [
                    "## CUDA memory (timed forward loop)",
                    "",
                    f"- Peak **allocated** (`max_memory_allocated`): **{peak_mem_mb:.1f} MB**",
                    f"- Peak **reserved** (`max_memory_reserved`): **{peak_reserved_mb:.1f} MB**",
                    "",
                ]
            )
        lines.extend(
            [
            "## Top operators",
            "",
            "| Rank | Operator | Self (ms) |",
            "| ---: | --- | ---: |",
            ]
        )
        for i, (n, t) in enumerate(zip(names, tms, strict=True), start=1):
            lines.append(f"| {i} | {n.replace('|', '\\|')} | {t:.3f} |")
        lines.extend(kpi_md)
        lines.extend(
            [
                "## GEMM call stats",
                "",
                f"- `aten::addmm` calls: {addmm_calls}",
                f"- `aten::addmm` self-time: {addmm_self_ms:.3f} ms",
                "",
            ]
        )
        lines.extend(["", "## Full profiler table", "", "```text", table.rstrip(), "```", ""])
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] {Path(report_out).resolve()}")

    return _RunResult(
        timing_line=timing_line,
        names=names,
        tms=tms,
        table=table,
        buckets=buckets,
        total_kpi_ms=total_kpi_ms,
        peak_mem_mb=peak_mem_mb,
        peak_reserved_mb=peak_reserved_mb,
        use_cuda=use_cuda,
        addmm_calls=addmm_calls,
        addmm_self_ms=addmm_self_ms,
    )


def _build_shared_init_state_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Build one reference state dict for fair A/B/C compare."""
    import torch

    from aurora.model.swin3d import Swin3DTransformerBackbone

    if getattr(args, "randomize_lora", False):
        torch.manual_seed(int(getattr(args, "lora_random_seed", 0)))

    backbone = Swin3DTransformerBackbone(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode="single",
        use_triton_layout=False,
        use_triton_adaln=False,
        use_triton_mlp=False,
        use_lora_merged_inference=False,
    ).eval()

    if getattr(args, "randomize_lora", False):
        _randomize_lora_weights(backbone, seed=int(getattr(args, "lora_random_seed", 0)))
        print(
            f"[lora] randomized once for shared compare state, "
            f"seed={getattr(args, 'lora_random_seed', 0)}"
        )

    # Clone to detach from module storage; reused by all compare runs.
    return {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}


def _fmt_mem_mb(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "—"


def _extract_ms_per_forward(timing_line: str) -> float | None:
    marker = "ms/forward"
    if marker not in timing_line:
        return None
    prefix = timing_line.split(marker, 1)[0].strip()
    try:
        return float(prefix.rsplit(" ", 1)[-1])
    except ValueError:
        return None


def _print_and_write_compare_summary(
    baseline: _RunResult,
    triton: _RunResult,
    *,
    report_out: str,
    triton_label: str,
) -> None:
    focus = ("copy_layout", "roll_pad_layout", "layer_norm")
    base_ms = _extract_ms_per_forward(baseline.timing_line)
    tri_ms = _extract_ms_per_forward(triton.timing_line)
    speedup = None
    if base_ms and tri_ms and tri_ms > 0:
        speedup = base_ms / tri_ms

    lines_console = [
        "",
        "=== baseline vs optimized compare ===",
        f"baseline timer: {baseline.timing_line}",
        f"{triton_label} timer: {triton.timing_line}",
    ]
    if speedup is not None:
        lines_console.append(
            f"speedup: {speedup:.3f}x ({(base_ms - tri_ms):.2f} ms/forward delta)"
        )
    if baseline.peak_mem_mb is not None and triton.peak_mem_mb is not None:
        lines_console.append(
            "peak CUDA allocated: "
            f"baseline={baseline.peak_mem_mb:.1f} MB, {triton_label}={triton.peak_mem_mb:.1f} MB"
        )
        if baseline.peak_reserved_mb is not None and triton.peak_reserved_mb is not None:
            lines_console.append(
                "peak CUDA reserved:  "
                f"baseline={baseline.peak_reserved_mb:.1f} MB, "
                f"{triton_label}={triton.peak_reserved_mb:.1f} MB"
            )

    lines_console.append("focus KPI buckets (all profiler rows, self-time):")
    for k in focus:
        b = baseline.buckets.get(k, 0.0)
        t = triton.buckets.get(k, 0.0)
        d = t - b
        rel = (100.0 * d / b) if b > 0 else float("nan")
        rel_s = f"{rel:+.1f}%" if b > 0 else "n/a"
        lines_console.append(f"  {k}: baseline={b:.3f} ms, triton={t:.3f} ms, delta={d:+.3f} ms ({rel_s})")

    for line in lines_console:
        print(line)

    if not report_out:
        return

    md = [
        "# Swin3D compare (baseline vs optimized)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Forward latency",
        "",
        "| Run | Timing |",
        "| --- | --- |",
        f"| Baseline | {baseline.timing_line} |",
        f"| {triton_label} | {triton.timing_line} |",
    ]
    if speedup is not None:
        md.append(f"- **Speedup:** {speedup:.3f}x")
        md.append(f"- **Delta:** {(base_ms - tri_ms):.2f} ms/forward")

    if baseline.peak_mem_mb is not None and triton.peak_mem_mb is not None:
        md.extend(
            [
                "",
                "## CUDA memory (timed forward loop)",
                "",
                "| Run | peak allocated (MB) | peak reserved (MB) |",
                "| --- | ---: | ---: |",
                f"| Baseline | {baseline.peak_mem_mb:.1f} | {_fmt_mem_mb(baseline.peak_reserved_mb)} |",
                f"| {triton_label} | {triton.peak_mem_mb:.1f} | {_fmt_mem_mb(triton.peak_reserved_mb)} |",
                "",
            ]
        )

    md.extend(["", "## Focus KPIs", "", "| Bucket | Baseline (ms) | Triton (ms) | Delta (ms) | Delta % |", "| --- | ---: | ---: | ---: | ---: |"])
    for k in focus:
        b = baseline.buckets.get(k, 0.0)
        t = triton.buckets.get(k, 0.0)
        d = t - b
        rel = (100.0 * d / b) if b > 0 else float("nan")
        rel_s = f"{rel:+.1f}" if b > 0 else "n/a"
        md.append(f"| {k} | {b:.3f} | {t:.3f} | {d:+.3f} | {rel_s} |")

    md.extend(["", "## Aggregate totals", "", "| Run | Total aggregate self-time (ms) |", "| --- | ---: |"])
    md.append(f"| Baseline | {baseline.total_kpi_ms:.3f} |")
    md.append(f"| {triton_label} | {triton.total_kpi_ms:.3f} |")
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[compare-report] {Path(report_out).resolve()}")


def _write_stagec_compare_report(
    report_out: str,
    *,
    baseline: _RunResult,
    stage_ab: _RunResult,
    stage_c: _RunResult,
    stage_ab_label: str,
    stage_c_label: str,
    lora_note: str,
) -> None:
    if not report_out:
        return
    b_ms = _extract_ms_per_forward(baseline.timing_line)
    ab_ms = _extract_ms_per_forward(stage_ab.timing_line)
    c_ms = _extract_ms_per_forward(stage_c.timing_line)
    md = [
        "# Swin3D Stage-C compare",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        lora_note,
        "",
        "## Forward latency",
        "",
        "| Run | Timing |",
        "| --- | --- |",
        f"| Baseline | {baseline.timing_line} |",
        f"| {stage_ab_label} | {stage_ab.timing_line} |",
        f"| {stage_c_label} | {stage_c.timing_line} |",
    ]
    if b_ms is not None and ab_ms is not None and c_ms is not None:
        md.extend(
            [
                f"- **Stage-A/B speedup vs baseline:** {b_ms / ab_ms:.3f}x",
                f"- **Stage-C speedup vs baseline:** {b_ms / c_ms:.3f}x",
                f"- **Stage-C vs Stage-A/B delta:** {ab_ms - c_ms:+.2f} ms/forward ({ab_ms / c_ms:.3f}x)",
            ]
        )
    if baseline.peak_mem_mb is not None:
        md.extend(
            [
                "",
                "## CUDA memory (timed forward loop)",
                "",
                "| Run | peak allocated (MB) | peak reserved (MB) |",
                "| --- | ---: | ---: |",
                f"| Baseline | {baseline.peak_mem_mb:.1f} | {_fmt_mem_mb(baseline.peak_reserved_mb)} |",
                f"| {stage_ab_label} | {stage_ab.peak_mem_mb:.1f} | {_fmt_mem_mb(stage_ab.peak_reserved_mb)} |",
                f"| {stage_c_label} | {stage_c.peak_mem_mb:.1f} | {_fmt_mem_mb(stage_c.peak_reserved_mb)} |",
                "",
            ]
        )
    md.extend(
        [
            "",
            "## Aggregate totals",
            "",
            "| Run | Total aggregate self-time (ms) |",
            "| --- | ---: |",
            f"| Baseline | {baseline.total_kpi_ms:.3f} |",
            f"| {stage_ab_label} | {stage_ab.total_kpi_ms:.3f} |",
            f"| {stage_c_label} | {stage_c.total_kpi_ms:.3f} |",
            "",
            f"- **Stage-C vs Stage-A/B aggregate delta:** "
            f"{stage_ab.total_kpi_ms - stage_c.total_kpi_ms:+.3f} ms",
            "",
            "## addmm stats",
            "",
            "| Run | `aten::addmm` calls | `aten::addmm` self-time (ms) |",
            "| --- | ---: | ---: |",
            f"| Baseline | {baseline.addmm_calls} | {baseline.addmm_self_ms:.3f} |",
            f"| {stage_ab_label} | {stage_ab.addmm_calls} | {stage_ab.addmm_self_ms:.3f} |",
            f"| {stage_c_label} | {stage_c.addmm_calls} | {stage_c.addmm_self_ms:.3f} |",
            "",
            f"- **Stage-C vs Stage-A/B addmm calls delta:** "
            f"{stage_ab.addmm_calls - stage_c.addmm_calls:+d}",
            f"- **Stage-C vs Stage-A/B addmm self-time delta:** "
            f"{stage_ab.addmm_self_ms - stage_c.addmm_self_ms:+.3f} ms",
        ]
    )
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stagec-report] {Path(report_out).resolve()}")


def _print_d2d3_compare_summary(
    baseline: _RunResult,
    d2: _RunResult,
    d2d3: _RunResult,
    *,
    report_out: str,
) -> None:
    """Console + optional markdown: baseline vs D2 (Triton layout+AdaLN) vs D2+D3 (+ workspace pool)."""
    b_ms = _extract_ms_per_forward(baseline.timing_line)
    d2_ms = _extract_ms_per_forward(d2.timing_line)
    d23_ms = _extract_ms_per_forward(d2d3.timing_line)

    lines = [
        "",
        "=== D2 + D3 compare (baseline vs D2 layout+AdaLN vs D2+D3 pool) ===",
        f"baseline: {baseline.timing_line}",
        f"D2:       {d2.timing_line}",
        f"D2+D3:    {d2d3.timing_line}",
    ]
    if baseline.peak_mem_mb is not None and d2.peak_mem_mb is not None and d2d3.peak_mem_mb is not None:
        lines.append(
            "peak CUDA allocated: "
            f"baseline={baseline.peak_mem_mb:.1f} MB, D2={d2.peak_mem_mb:.1f} MB, "
            f"D2+D3={d2d3.peak_mem_mb:.1f} MB"
        )
        if (
            baseline.peak_reserved_mb is not None
            and d2.peak_reserved_mb is not None
            and d2d3.peak_reserved_mb is not None
        ):
            lines.append(
                "peak CUDA reserved:  "
                f"baseline={baseline.peak_reserved_mb:.1f} MB, D2={d2.peak_reserved_mb:.1f} MB, "
                f"D2+D3={d2d3.peak_reserved_mb:.1f} MB"
            )
    if b_ms and d2_ms and d23_ms:
        lines.append(f"D2 vs baseline speedup: {b_ms / d2_ms:.3f}x")
        lines.append(f"D2+D3 vs baseline speedup: {b_ms / d23_ms:.3f}x")
        lines.append(
            f"D2+D3 vs D2 only: ms/forward delta {d2_ms - d23_ms:+.4f} "
            f"({(d2_ms / d23_ms):.4f}x)"
        )
    lines.extend(
        [
            "",
            "aten::addmm (full profiler table):",
            f"  baseline: calls={baseline.addmm_calls}, self={baseline.addmm_self_ms:.3f} ms",
            f"  D2:       calls={d2.addmm_calls}, self={d2.addmm_self_ms:.3f} ms",
            f"  D2+D3:    calls={d2d3.addmm_calls}, self={d2d3.addmm_self_ms:.3f} ms",
        ]
    )
    for line in lines:
        print(line)

    if not report_out:
        return
    md = [
        "# Swin3D D2 + D3 compare",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "- **D2:** `use_triton_layout` + `use_triton_adaln` (fused AdaLN + residual when eval FP32).",
        "- **D3:** `InferenceWorkspacePool` on final decoder `cat` (same math, fewer alloc on that buffer).",
        "",
        "## Forward latency",
        "",
        "| Run | Timing |",
        "| --- | --- |",
        f"| Baseline | {baseline.timing_line} |",
        f"| D2 (layout + AdaLN) | {d2.timing_line} |",
        f"| D2 + D3 (pool) | {d2d3.timing_line} |",
        "",
    ]
    if b_ms and d2_ms and d23_ms:
        md.extend(
            [
                f"- **D2 vs baseline:** {b_ms / d2_ms:.3f}x",
                f"- **D2+D3 vs baseline:** {b_ms / d23_ms:.3f}x",
                f"- **D2+D3 vs D2 ms/forward delta:** {d2_ms - d23_ms:+.4f} ms",
                "",
            ]
        )
    if baseline.peak_mem_mb is not None:
        md.extend(
            [
                "## CUDA memory (timed forward loop)",
                "",
                "Peak stats from `torch.cuda.reset_peak_memory_stats()` before the timed loop, "
                "then `max_memory_allocated` / `max_memory_reserved` after the loop (same window as `[timing]`).",
                "",
                "| Run | peak allocated (MB) | peak reserved (MB) |",
                "| --- | ---: | ---: |",
                f"| Baseline | {baseline.peak_mem_mb:.1f} | {_fmt_mem_mb(baseline.peak_reserved_mb)} |",
                f"| D2 | {d2.peak_mem_mb:.1f} | {_fmt_mem_mb(d2.peak_reserved_mb)} |",
                f"| D2+D3 | {d2d3.peak_mem_mb:.1f} | {_fmt_mem_mb(d2d3.peak_reserved_mb)} |",
                "",
            ]
        )
    md.extend(
        [
            "## aten::addmm",
            "",
            "| Run | calls | self (ms) |",
            "| --- | ---: | ---: |",
            f"| Baseline | {baseline.addmm_calls} | {baseline.addmm_self_ms:.3f} |",
            f"| D2 | {d2.addmm_calls} | {d2.addmm_self_ms:.3f} |",
            f"| D2+D3 | {d2d3.addmm_calls} | {d2d3.addmm_self_ms:.3f} |",
            "",
        ]
    )
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[compare-d2d3-report] {Path(report_out).resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile Swin3DTransformerBackbone only.",
        epilog=(
            "Optional deeper GPU analysis: run under Nsight Compute, e.g.\n"
            "  ncu --set full -o profiling/swin3d_ncu python aurora/profiling_swin3d.py --preset baseline"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=("none", "baseline", "stress", "stress-heavy"),
        default="none",
        help=(
            "baseline: force batch=1 and patch_res=(4,32,64) (L=8192) to match profiling/swin3d.md; "
            "ignores --batch-size / --latent-levels / --patch-h / --patch-w. "
            "stress: batch=4, patch_res=(4,16,32) → L=2048, warmup=8, repeat=16 (8192 tokens/step). "
            "stress-heavy: batch=8, patch_res=(4,32,64) → L=8192 (very large VRAM only)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--latent-levels",
        type=int,
        default=4,
        help="C in patch_res (C, H, W); must be divisible by window_size[0] (default 2).",
    )
    p.add_argument("--patch-h", type=int, default=32, help="Patch grid height H.")
    p.add_argument("--patch-w", type=int, default=64, help="Patch grid width W.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--plot-out", type=str, default="")
    p.add_argument("--plot-top", type=int, default=25)
    p.add_argument("--report-out", type=str, default="")
    p.add_argument(
        "--compare-triton",
        action="store_true",
        help=(
            "Run Stage-A compare in one command: baseline (no Triton) then Triton "
            "(layout+AdaLN), with ms/forward and KPI deltas."
        ),
    )
    p.add_argument(
        "--compare-report-out",
        type=str,
        default="",
        help="Optional markdown output for --compare-triton summary.",
    )
    p.add_argument(
        "--compare-stagec",
        action="store_true",
        help=(
            "Run three-way compare: baseline, Stage-A (layout+AdaLN[+MLP]), "
            "and Stage-C (+LoRA merged inference)."
        ),
    )
    p.add_argument(
        "--compare-d2d3",
        action="store_true",
        help=(
            "Run three-way compare: baseline (no Triton), D2 (layout+AdaLN), "
            "D2+D3 (same + InferenceWorkspacePool on decoder concat). "
            "Use --preset stress for higher batch/repeat (see --preset help)."
        ),
    )
    p.add_argument(
        "--no-empty-cache-between-runs",
        action="store_true",
        help=(
            "Skip gc.collect + torch.cuda.empty_cache before each profile run. "
            "By default we reset so peak reserved/allocated compare more fairly between runs."
        ),
    )
    p.add_argument("--table-rows", type=int, default=30)
    p.add_argument(
        "--autocast-backbone",
        action="store_true",
        help="Run backbone under BF16 autocast (matches Aurora forward when autocast=True).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap backbone with torch.compile. Default: dynamic=False, mode=default (stable for fixed patch_res).",
    )
    p.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Use dynamic=True (varying shapes; may hit slow Inductor/sympy on some GPUs).",
    )
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help="torch.compile mode (default, reduce-overhead, max-autotune). Use 'default' on laptops.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--use-triton-layout",
        action="store_true",
        help="Enable fused roll/pad/window Triton path (CUDA float32 only; no effect under autocast BF16).",
    )
    p.add_argument(
        "--use-triton-adaln",
        action="store_true",
        help="Enable fused AdaptiveLayerNorm Triton path (CUDA float32 only).",
    )
    p.add_argument(
        "--use-triton-mlp",
        action="store_true",
        help="Enable Triton GELU in MLP inference path (CUDA float32, dropout=0 only).",
    )
    p.add_argument(
        "--use-lora-merged-inference",
        action="store_true",
        help="Enable merged LoRA+Linear inference path in attention qkv/proj.",
    )
    p.add_argument(
        "--use-workspace-pool",
        action="store_true",
        help="Enable InferenceWorkspacePool on backbone final decoder concat (D3). Single-run profiling only.",
    )
    p.add_argument(
        "--randomize-lora",
        action="store_true",
        help=(
            "Reinitialize all LoRA lora_A/lora_B (non-zero B) so profiling reflects non-trivial "
            "LoRA delta (default checkpoint-free init has lora_B=0)."
        ),
    )
    p.add_argument(
        "--lora-random-seed",
        type=int,
        default=0,
        help="RNG seed for --randomize-lora.",
    )
    args = p.parse_args()

    if args.preset == "baseline":
        args.batch_size = 1
        args.latent_levels = 4
        args.patch_h = 32
        args.patch_w = 64
    elif args.preset == "stress":
        # Same tokens/step as baseline (8192): 4 x (4.16.32). Safer for compare-d2d3 baseline path.
        args.batch_size = 4
        args.latent_levels = 4
        args.patch_h = 32
        args.patch_w = 64
        args.warmup = 8
        args.repeat = 16
    elif args.preset == "stress-heavy":
        args.batch_size = 8
        args.latent_levels = 4
        args.patch_h = 32
        args.patch_w = 64
        args.warmup = 8
        args.repeat = 16

    import torch
    from torch.profiler import ProfilerActivity, profile

    from aurora.model.swin3d import Swin3DTransformerBackbone

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA required unless --device cpu.")

    C, H, W = args.latent_levels, args.patch_h, args.patch_w
    L = C * H * W
    if (args.use_triton_layout or args.use_triton_adaln) and args.autocast_backbone:
        print(
            "[warn] --autocast-backbone uses BF16; Triton layout/AdaLN kernels require float32 "
            "and will fall back to PyTorch inside the module."
        )
    if args.use_triton_mlp and args.autocast_backbone:
        print(
            "[warn] --autocast-backbone uses BF16; Triton MLP GELU requires float32 and "
            "will fall back to PyTorch GELU inside the module."
        )
    if args.compare_triton:
        if args.use_triton_layout or args.use_triton_adaln:
            print(
                "[warn] --compare-triton ignores --use-triton-layout/--use-triton-adaln and "
                "runs both modes explicitly."
            )
        if args.report_out:
            print("[warn] --report-out is ignored under --compare-triton; use --compare-report-out.")

        shared_init_state = _build_shared_init_state_dict(args)
        print("\n=== Run 1/2: baseline (no Triton layout/AdaLN) ===")
        baseline = _run_profile_once(
            args,
            use_triton_layout=False,
            use_triton_adaln=False,
            use_triton_mlp=False,
            use_lora_merged_inference=False,
            init_state_dict=shared_init_state,
        )
        print("\n=== Run 2/2: Triton (layout + AdaLN) ===")
        triton = _run_profile_once(
            args,
            use_triton_layout=True,
            use_triton_adaln=True,
            use_triton_mlp=args.use_triton_mlp,
            use_lora_merged_inference=args.use_lora_merged_inference,
            init_state_dict=shared_init_state,
        )
        triton_label = "Triton (layout+AdaLN)"
        if args.use_triton_mlp:
            triton_label = "Triton (layout+AdaLN+MLP-GELU)"
        _print_and_write_compare_summary(
            baseline,
            triton,
            report_out=args.compare_report_out,
            triton_label=triton_label,
        )
        return

    if args.compare_stagec:
        if args.report_out:
            print("[warn] --report-out is ignored under --compare-stagec; use --compare-report-out.")
        shared_init_state = _build_shared_init_state_dict(args)
        print("\n=== Run 1/3: baseline (no Triton / no merge) ===")
        baseline = _run_profile_once(
            args,
            use_triton_layout=False,
            use_triton_adaln=False,
            use_triton_mlp=False,
            use_lora_merged_inference=False,
            init_state_dict=shared_init_state,
        )
        print("\n=== Run 2/3: Stage-A/B (layout + AdaLN [+ MLP]) ===")
        stage_ab = _run_profile_once(
            args,
            use_triton_layout=True,
            use_triton_adaln=True,
            use_triton_mlp=args.use_triton_mlp,
            use_lora_merged_inference=False,
            init_state_dict=shared_init_state,
        )
        print("\n=== Run 3/3: Stage-C (layout + AdaLN [+ MLP] + LoRA merge) ===")
        stage_c = _run_profile_once(
            args,
            use_triton_layout=True,
            use_triton_adaln=True,
            use_triton_mlp=args.use_triton_mlp,
            use_lora_merged_inference=True,
            init_state_dict=shared_init_state,
        )
        _print_and_write_compare_summary(
            baseline,
            stage_c,
            report_out="",
            triton_label="Stage-C (layout+AdaLN+LoRA-merge)",
        )
        stage_ab_label = "Stage-A/B (layout+AdaLN)"
        if args.use_triton_mlp:
            stage_ab_label = "Stage-A/B (layout+AdaLN+MLP-GELU)"
        lora_note = (
            f"- **LoRA:** randomized A/B (seed={args.lora_random_seed}) for non-zero ΔW."
            if args.randomize_lora
            else "- **LoRA:** default init (lora_B=0 until finetuned); add `--randomize-lora` for Stage-C stress."
        )
        _write_stagec_compare_report(
            args.compare_report_out,
            baseline=baseline,
            stage_ab=stage_ab,
            stage_c=stage_c,
            stage_ab_label=stage_ab_label,
            stage_c_label="Stage-C (layout+AdaLN+LoRA-merge)",
            lora_note=lora_note,
        )
        print("\n[stagec-delta]")
        ab_ms = _extract_ms_per_forward(stage_ab.timing_line)
        c_ms = _extract_ms_per_forward(stage_c.timing_line)
        if ab_ms is not None and c_ms is not None:
            print(
                f"  Stage-C vs Stage-A/B ms/forward delta: {ab_ms - c_ms:+.2f} "
                f"({(ab_ms / c_ms):.3f}x)"
            )
        print(
            f"  Stage-C vs Stage-A/B aggregate self-time delta: "
            f"{stage_ab.total_kpi_ms - stage_c.total_kpi_ms:+.3f} ms"
        )
        return

    if args.compare_d2d3:
        if args.report_out:
            print("[warn] --report-out is ignored under --compare-d2d3; use --compare-report-out.")
        shared_init_state = _build_shared_init_state_dict(args)
        print("\n=== Run 1/3: baseline (no Triton layout/AdaLN, no pool) ===")
        baseline = _run_profile_once(
            args,
            use_triton_layout=False,
            use_triton_adaln=False,
            use_triton_mlp=False,
            use_lora_merged_inference=False,
            use_workspace_pool=False,
            init_state_dict=shared_init_state,
        )
        print("\n=== Run 2/3: D2 (layout + AdaLN), no pool ===")
        d2 = _run_profile_once(
            args,
            use_triton_layout=True,
            use_triton_adaln=True,
            use_triton_mlp=args.use_triton_mlp,
            use_lora_merged_inference=args.use_lora_merged_inference,
            use_workspace_pool=False,
            init_state_dict=shared_init_state,
        )
        print("\n=== Run 3/3: D2 + D3 (layout + AdaLN + InferenceWorkspacePool) ===")
        d2d3 = _run_profile_once(
            args,
            use_triton_layout=True,
            use_triton_adaln=True,
            use_triton_mlp=args.use_triton_mlp,
            use_lora_merged_inference=args.use_lora_merged_inference,
            use_workspace_pool=True,
            init_state_dict=shared_init_state,
        )
        _print_d2d3_compare_summary(
            baseline,
            d2,
            d2d3,
            report_out=args.compare_report_out,
        )
        return

    result = _run_profile_once(
        args,
        use_triton_layout=args.use_triton_layout,
        use_triton_adaln=args.use_triton_adaln,
        use_triton_mlp=args.use_triton_mlp,
        use_lora_merged_inference=args.use_lora_merged_inference,
        use_workspace_pool=args.use_workspace_pool,
        report_out=args.report_out,
    )

    if args.plot_out:
        try:
            _save_plot(result.names, result.tms, Path(args.plot_out), result.timing_line)
            print(f"\n[plot] {Path(args.plot_out).resolve()}")
        except ImportError:
            print("matplotlib missing; skip --plot-out")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Profiler.*", category=UserWarning)
    main()
