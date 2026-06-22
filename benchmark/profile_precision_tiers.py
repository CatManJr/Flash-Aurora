#!/usr/bin/env python3
"""Profile ``tf32`` vs ``bf16_mixed`` on AuroraSmallPretrained (400x800).

Default scope is **Swin backbone only**: encoder runs once to build tokens; the timed
loop is ``_run_backbone`` only. Perceiver (encoder/decoder) is identical across
tiers and excluded. Use ``--full-model`` to profile end-to-end ``forward``.

Aggregates torch.profiler self-CUDA into buckets (GEMM, CuTe window attn, casts,
layout, etc.) and prints a side-by-side table plus optional Markdown report.

Example::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/profile_precision_tiers.py
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/profile_precision_tiers.py \\
        --report-out profiling/tf32_vs_bf16_mixed.md
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import _bootstrap  # noqa: F401, E402


from profiling_swin3d import (  # noqa: E402
    _aggregate_full_table_kpis,
    _extract_addmm_stats,
    _kpi_console_and_markdown,
    _top_ops_ms,
)

from bench_small_pretrained import _load_batch, _purge_gpu  # noqa: E402


def _bucket_tier_profile(name: str) -> str:
    """Finer buckets for tf32 vs bf16 comparison (all profiler rows)."""
    n = name.lower()
    if "memcpy" in n or "dtoh" in n or "htod" in n or "memset" in n:
        return "memcpy"
    if "aten::to" in n or "convert_element_type" in n or "_to_copy" in n:
        return "cast_dtype"
    if "aten::copy_" in n or "direct_copy" in n:
        return "copy_tensor"
    # CuTe shifted-window attention (before generic "cutlass" -> gemm_other).
    if "windowattnfwd" in n:
        return "attention_cute_window"
    if (
        "efficient_attention" in n
        or "fmha" in n
        or "flash" in n
        or "scaled_dot_product" in n
        or "sdpa" in n
    ):
        return "attention_sdpa"
    if "aten::linear" in n:
        return "linear"
    if "aten::addmm" in n:
        return "addmm"
    if (
        "magma" in n
        or "cutlass" in n
        or "cublas" in n
        or "aten::mm" in n
        or n.endswith("::mm")
        or "gemm" in n
    ):
        return "gemm_other"
    if "layer_norm" in n or "native_layer_norm" in n:
        return "layer_norm"
    if "roll" in n or "triton" in n and "layout" in n:
        return "layout_triton"
    if "triton" in n or "compiled" in n:
        return "triton_other"
    if "gelu" in n or "silu" in n:
        return "gelu_act"
    if "elementwise" in n or "::mul" in n or "::add" in n:
        return "elementwise"
    return "other"


def _aggregate_buckets(prof: Any, *, use_cuda: bool) -> tuple[dict[str, float], float]:
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
        b = _bucket_tier_profile(str(e.key))
        buckets[b] = buckets.get(b, 0.0) + t_ms
    return buckets, total_ms


@dataclass
class ProfileResult:
    tier: str
    scope: str
    ms_forward: float
    buckets: dict[str, float]
    total_kpi_ms: float
    addmm_calls: int
    addmm_ms: float
    top_names: list[str]
    top_ms: list[float]


def _prepare_backbone_input(
    model: Any, batch: Any
) -> tuple[tuple[int, int, int], Any, int]:
    """Run encoder once; return ``(patch_res, tokens, rollout_step)`` for backbone loops."""
    import torch

    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_autocast

    _, transformed, patch_res = model._prepare_encoder_batch(batch)
    with torch.inference_mode():
        x = run_with_encoder_decoder_autocast(
            model.encoder,
            transformed,
            enabled=model.autocast_encoder_decoder,
            lead_time=model.timestep,
        )
    return patch_res, x, batch.metadata.rollout_step


def _profile_tier(
    tier: str,
    batch: Any,
    *,
    device: str,
    warmup: int,
    repeat: int,
    data_dir: Path,
    top_k: int,
    backbone_only: bool,
) -> ProfileResult:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from flash_aurora.aurora import AuroraSmallPretrained

    dev = torch.device(device)
    model = AuroraSmallPretrained(use_lora=False, inference_precision=tier)
    model.load_checkpoint_local(
        str(data_dir / "aurora-0.25-small-pretrained.ckpt"),
        strict=True,
    )
    model.eval()
    model.to(dev)
    batch_gpu = batch.to(dev)
    scope = "backbone" if backbone_only else "full_model"

    patch_res: tuple[int, int, int] | None = None
    backbone_x: Any = None
    rollout_step = 0
    if backbone_only:
        patch_res, backbone_x, rollout_step = _prepare_backbone_input(model, batch_gpu)

    def run_once() -> None:
        with torch.inference_mode():
            if backbone_only:
                assert patch_res is not None and backbone_x is not None
                _ = model._run_backbone(
                    backbone_x,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                    rollout_step=rollout_step,
                )
            else:
                _ = model.forward(batch_gpu)

    for _ in range(warmup):
        run_once()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    start.record()
    for _ in range(repeat):
        run_once()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    ms_forward = start.elapsed_time(end) / repeat

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(repeat):
            run_once()
        if dev.type == "cuda":
            torch.cuda.synchronize()

    buckets, total_kpi_ms = _aggregate_buckets(prof, use_cuda=dev.type == "cuda")
    addmm_calls, addmm_ms = _extract_addmm_stats(prof, use_cuda=dev.type == "cuda")
    top_names, top_ms = _top_ops_ms(prof, use_cuda=dev.type == "cuda", top_k=top_k)

    _purge_gpu(model, batch_gpu)
    return ProfileResult(
        tier=tier,
        scope=scope,
        ms_forward=ms_forward,
        buckets=buckets,
        total_kpi_ms=total_kpi_ms,
        addmm_calls=addmm_calls,
        addmm_ms=addmm_ms,
        top_names=top_names,
        top_ms=top_ms,
    )


def _print_comparison(results: list[ProfileResult]) -> None:
    scope = results[0].scope if results else "backbone"
    print(f"\n=== Scope: {scope} (encoder/decoder excluded when backbone) ===")
    all_buckets = sorted(
        {b for r in results for b in r.buckets},
        key=lambda k: -max(r.buckets.get(k, 0.0) for r in results),
    )
    print("\n=== GPU time (CUDA events, no profiler overhead) ===")
    for r in results:
        print(f"  {r.tier:<14} {r.ms_forward:7.1f} ms/forward")

    print("\n=== Self-CUDA buckets (ms, % of bucket total per tier) ===")
    hdr = f"{'bucket':<18}" + "".join(f"{r.tier:>14}" for r in results)
    if len(results) == 2:
        hdr += f"{'Δ ms':>10}{'Δ%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for b in all_buckets:
        cols = [f"{r.buckets.get(b, 0.0):13.2f}" for r in results]
        line = f"{b:<18}" + "".join(cols)
        if len(results) == 2:
            a, t = results[0].buckets.get(b, 0.0), results[1].buckets.get(b, 0.0)
            line += f"{t - a:10.2f}{(100.0 * (t - a) / a) if a > 1e-6 else 0.0:7.1f}%"
        print(line)

    print("\n=== aten::addmm (proxy for hooked linear GEMM) ===")
    for r in results:
        pct = 100.0 * r.addmm_ms / r.total_kpi_ms if r.total_kpi_ms else 0.0
        print(
            f"  {r.tier:<14} calls={r.addmm_calls:5d}  self_ms={r.addmm_ms:8.2f}  "
            f"({pct:.1f}% of profiled self-cuda)"
        )
    if len(results) == 2:
        d_calls = results[1].addmm_calls - results[0].addmm_calls
        d_ms = results[1].addmm_ms - results[0].addmm_ms
        print(f"  {'Δ bf16-tf32':<14} calls={d_calls:+5d}  self_ms={d_ms:+8.2f}")

    for r in results:
        print(f"\n--- Top-{len(r.top_names)} ops: {r.tier} ---")
        for name, ms in zip(r.top_names, r.top_ms, strict=True):
            print(f"  {ms:8.2f} ms  {name[:90]}")


def _write_markdown(path: Path, results: list[ProfileResult]) -> None:
    scope = results[0].scope if results else "backbone"
    lines = [
        "# tf32 vs bf16_mixed profiler comparison",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Scope: **{scope}** (Perceiver encoder/decoder identical across tiers)",
        f"- Batch: AuroraSmallPretrained HF test input (400×800), `use_lora=False`",
        "",
        "## Forward time (CUDA events, no profiler in timed loop)",
        "",
        "| Tier | ms/forward |",
        "| --- | ---: |",
    ]
    for r in results:
        lines.append(f"| {r.tier} | {r.ms_forward:.2f} |")
    if len(results) == 2:
        lines.append(
            f"| Δ (bf16 − tf32) | {results[1].ms_forward - results[0].ms_forward:+.2f} |"
        )

    all_buckets = sorted(
        {b for r in results for b in r.buckets},
        key=lambda k: -max(r.buckets.get(k, 0.0) for r in results),
    )
    lines.extend(["", "## Self-CUDA buckets (ms)", "", "| Bucket | " + " | ".join(r.tier for r in results) + " |", "| --- | " + " | ".join("---:" for _ in results) + " |"])
    for b in all_buckets:
        row = f"| {b} | " + " | ".join(f"{r.buckets.get(b, 0.0):.2f}" for r in results) + " |"
        lines.append(row)

    lines.extend(["", "## aten::addmm", "", "| Tier | calls | self ms | % of total |", "| --- | ---: | ---: | ---: |"])
    for r in results:
        pct = 100.0 * r.addmm_ms / r.total_kpi_ms if r.total_kpi_ms else 0.0
        lines.append(f"| {r.tier} | {r.addmm_calls} | {r.addmm_ms:.2f} | {pct:.1f} |")

    for r in results:
        lines.extend(["", f"## Top ops — {r.tier}", "", "| ms | op |", "| ---: | --- |"])
        for name, ms in zip(r.top_names, r.top_ms, strict=True):
            lines.append(f"| {ms:.3f} | `{name}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-tmp/aurora"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Profile end-to-end forward (includes Perceiver). Default is backbone only.",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    data_dir = args.data_dir.expanduser().resolve()
    batch = _load_batch(data_dir)

    tiers = ("tf32", "bf16_mixed")
    results: list[ProfileResult] = []
    for tier in tiers:
        print(f"\n[profile] {tier} ...", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        results.append(
            _profile_tier(
                tier,
                batch,
                device="cuda",
                warmup=args.warmup,
                repeat=args.repeat,
                data_dir=data_dir,
                top_k=args.top_k,
                backbone_only=not args.full_model,
            )
        )

    _print_comparison(results)
    if args.report_out is not None:
        _write_markdown(args.report_out.expanduser().resolve(), results)


if __name__ == "__main__":
    main()
