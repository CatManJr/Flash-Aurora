#!/usr/bin/env python3
"""Copyright (c) Catman Jr. Licensed under the MIT license.

Full-model Aurora benchmark: compare six Swin3D optimization tiers on throughput and VRAM.

Tiers (accuracy reference = ``fp32``):

1. ``fp32``             — PyTorch FP32
2. ``pytorch_autocast`` — PyTorch backbone BF16 autocast
3. ``fast_fp32``        — Triton Swin + native Perceiver
4. ``tf32_1x``          — ``fast_fp32`` + CuTe 1×TF32 window attention
5. ``bf16_mixed``       — ``fast_fp32`` + explicit BF16 CuTe window attention
6. ``full_bf16``        — full-model BF16 mixed precision + Perceiver FlashAttention

All tiers except ``full_bf16`` use native Perceiver (PyTorch SDPA). Each tier rebuilds
the model and fully purges GPU state before timing.

Examples::

    uv run python benchmark/bench_aurora_precision_matrix.py
    uv run python benchmark/bench_aurora_precision_matrix.py --preset medium
    uv run python benchmark/bench_aurora_precision_matrix.py --batch-size 4 --repeat 50

Unless ``--batch-size`` is set, batch size is auto-probed to target ``--vram-fraction`` (default
90%) of total GPU memory using a conservative fp32 forward before the tier matrix runs.
Use ``--preset smoke`` (32×64) only for quick sanity checks; ``bf16_mixed`` may crash there
(N<32 CuTe limitation).

Default grid is ``--preset production`` (721×1440, ERA5 0.25° global, patch_res 4×180×360).

Uses ``AuroraSmallPretrained`` (debug-scale embed_dim=256). Swin/CuTe wins are much larger on
the full Aurora checkpoint and ``--preset production`` grid; see script output section breakdown.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import torch

_REPO = Path(__file__).resolve().parents[1]
_AURORA_PKG = _REPO / "aurora"
if _AURORA_PKG.is_dir():
    sys.path.insert(0, str(_AURORA_PKG))

_DEFAULT_CHECKPOINT_DIR = "/root/autodl-tmp/aurora"

# ERA5 0.25° global (721×1440) → patch_res (4, 180, 360), window N=144 at all stages.
_GRID_PRESETS: dict[str, tuple[int, int]] = {
    "production": (721, 1440),
    "medium": (128, 256),  # patch_res (4, 32, 64), L=8192
    "smoke": (32, 64),  # tiny; bf16_mixed may hit N<32 CuTe limitation
}

_DEFAULT_VRAM_FRACTION = 0.90
_DEFAULT_BATCH_CAP = 512
_PROBE_PRECISION = "fp32"  # conservative vs all tiers (weights + activations)

_BENCH_TIERS: tuple[tuple[str, str, str], ...] = (
    ("fp32", "fp32", "PyTorch FP32"),
    ("pytorch_autocast", "pytorch_autocast", "PyTorch backbone BF16 autocast"),
    ("fast_fp32", "fast_fp32", "Triton Swin + native Perceiver"),
    ("tf32_1x", "tf32_1x", "fast_fp32 + CuTe 1×TF32 attention"),
    ("bf16_mixed", "bf16_mixed", "fast_fp32 + CuTe BF16 attention"),
    ("full_bf16", "full_bf16", "Full-model BF16 + Perceiver FlashAttention"),
)


@dataclass
class TierResult:
    key: str
    precision: str
    label: str
    ms_per_forward: float
    forwards_per_sec: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    max_abs_diff_vs_baseline: float | None
    mean_abs_diff_vs_baseline: float | None
    max_rel_diff_vs_baseline: float | None


def _purge_gpu(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _load_batch_synthetic(
    *,
    batch_size: int,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
    device: torch.device,
) -> Any:
    from aurora import Batch, Metadata

    batch = Batch(
        surf_vars={k: torch.randn(batch_size, history, h, w) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z", "slt")},
        atmos_vars={
            k: torch.randn(batch_size, history, len(levels), h, w) for k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    )
    return batch.to(device)


def _cuda_oom_like(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "outofmemory" in msg.replace(" ", "") or "out of memory" in msg:
        return True
    if "cudaerrormemoryallocation" in msg.replace(" ", ""):
        return True
    if type(exc).__name__ in {"OutOfMemoryError", "AcceleratorError"} and "memory" in msg:
        return True
    return False


def _recover_cuda_after_oom() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _gpu_total_mb(device: torch.device) -> float:
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 1e6


def _forward_peak_mb(
    model: Any,
    batch: Any,
    device: torch.device,
) -> float:
    _purge_gpu()
    with torch.inference_mode():
        _ = model.forward(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1e6
    return float("nan")


def _forward_peak_fits_target(
    *,
    state_dict: dict[str, torch.Tensor],
    batch: Any,
    device: torch.device,
    target_peak_mb: float,
    precision: str,
) -> tuple[bool, float]:
    model = _build_model(precision, state_dict, device)
    try:
        peak_mb = _forward_peak_mb(model, batch, device)
        return peak_mb <= target_peak_mb, peak_mb
    except Exception as exc:
        if _cuda_oom_like(exc):
            _recover_cuda_after_oom()
            return False, float("inf")
        raise
    finally:
        _purge_gpu(model, batch)


def _auto_batch_size_for_vram(
    *,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
    target_fraction: float,
    cap: int,
) -> tuple[int, float, float, float]:
    """Pick the largest batch whose fp32 forward peak stays within ``target_fraction`` of total VRAM."""
    total_mb = _gpu_total_mb(device)
    target_peak_mb = total_mb * target_fraction

    if not _forward_peak_fits_target(
        state_dict=state_dict,
        batch=_load_batch_synthetic(
            batch_size=1,
            h=h,
            w=w,
            history=history,
            levels=levels,
            device=device,
        ),
        device=device,
        target_peak_mb=target_peak_mb,
        precision=_PROBE_PRECISION,
    )[0]:
        print(
            f"[auto-vram] batch=1 peak exceeds {target_fraction:.0%} of total "
            f"({total_mb:.0f} MB); using batch_size=1."
        )
        peak_mb = _forward_peak_fits_target(
            state_dict=state_dict,
            batch=_load_batch_synthetic(
                batch_size=1, h=h, w=w, history=history, levels=levels, device=device
            ),
            device=device,
            target_peak_mb=float("inf"),
            precision=_PROBE_PRECISION,
        )[1]
        return 1, peak_mb, total_mb, target_peak_mb

    lo, hi = 1, max(1, cap)
    best_batch = 1
    best_peak = float("inf")

    # Expand hi until OOM or cap when still below target.
    while hi < cap:
        fits, peak_mb = _forward_peak_fits_target(
            state_dict=state_dict,
            batch=_load_batch_synthetic(
                batch_size=hi, h=h, w=w, history=history, levels=levels, device=device
            ),
            device=device,
            target_peak_mb=target_peak_mb,
            precision=_PROBE_PRECISION,
        )
        if fits:
            best_batch, best_peak = hi, peak_mb
            if hi >= cap:
                break
            hi = min(cap, hi * 2)
            continue
        break

    if hi > lo and best_batch < hi:
        # Binary search between last good batch and first failing bound.
        fail_hi = hi
        lo = best_batch
        while lo < fail_hi:
            mid = (lo + fail_hi + 1) // 2
            fits, peak_mb = _forward_peak_fits_target(
                state_dict=state_dict,
                batch=_load_batch_synthetic(
                    batch_size=mid, h=h, w=w, history=history, levels=levels, device=device
                ),
                device=device,
                target_peak_mb=target_peak_mb,
                precision=_PROBE_PRECISION,
            )
            if fits:
                lo = mid
                best_batch, best_peak = mid, peak_mb
            else:
                fail_hi = mid - 1

    return best_batch, best_peak, total_mb, target_peak_mb


def _prediction_tensors(pred: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, tensor in getattr(pred, group).items():
            out[f"{group}.{name}"] = tensor.detach().float().cpu()
    return out


def _diff_vs_reference(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, float, float]:
    max_diff = 0.0
    max_rel = 0.0
    total = 0.0
    count = 0
    for key, ref in reference.items():
        cand = candidate[key]
        diff = (cand - ref).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        denom = ref.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        total += float(diff.mean().item())
        count += 1
    mean_diff = total / max(count, 1)
    return max_diff, mean_diff, max_rel


def _load_shared_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    from aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=True, lora_mode="single")
    model.load_checkpoint_local(str(checkpoint_path), strict=False)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _build_model(precision: str, state_dict: dict[str, torch.Tensor], device: torch.device) -> Any:
    from aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(
        use_lora=True,
        lora_mode="single",
        inference_precision=precision,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device)


def _time_forward(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                _ = model.forward(batch)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = model.forward(batch)
            ms_total = (time.perf_counter() - t0) * 1e3
            peak_alloc = float("nan")
            peak_reserved = float("nan")

    ms_per = ms_total / repeat
    fps = 1000.0 / ms_per if ms_per > 0 else float("inf")
    return ms_per, peak_alloc, peak_reserved


def _run_tier(
    *,
    key: str,
    precision: str,
    label: str,
    state_dict: dict[str, torch.Tensor],
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
    baseline_pred: dict[str, torch.Tensor] | None,
) -> TierResult:
    print(f"\n{'=' * 72}\n[tier] {key} ({precision}) — {label}\n{'=' * 72}")
    _purge_gpu()

    model = _build_model(precision, state_dict, device)
    ms_per, peak_alloc, peak_reserved = _time_forward(
        model, batch, warmup=warmup, repeat=repeat, device=device
    )

    max_diff: float | None = None
    mean_diff: float | None = None
    max_rel: float | None = None
    if baseline_pred is not None:
        with torch.inference_mode():
            pred = model.forward(batch)
        cand = _prediction_tensors(pred)
        max_diff, mean_diff, max_rel = _diff_vs_reference(baseline_pred, cand)
        print(
            f"[accuracy] max_abs_diff={max_diff:.6e} mean_abs_diff={mean_diff:.6e} "
            f"max_rel_diff={max_rel:.6e} vs baseline"
        )

    print(
        f"[timing] {ms_per:.3f} ms/forward ({1000.0 / ms_per:.2f} forwards/s)\n"
        f"[mem] peak allocated={peak_alloc:.1f} MB, peak reserved={peak_reserved:.1f} MB"
    )

    _purge_gpu(model, batch)
    return TierResult(
        key=key,
        precision=precision,
        label=label,
        ms_per_forward=ms_per,
        forwards_per_sec=1000.0 / ms_per,
        peak_alloc_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        max_abs_diff_vs_baseline=max_diff,
        mean_abs_diff_vs_baseline=mean_diff,
        max_rel_diff_vs_baseline=max_rel,
    )


def _print_summary(results: list[TierResult]) -> None:
    baseline_ms = results[0].ms_per_forward if results else 1.0
    print(f"\n{'=' * 72}\nSummary (baseline = {results[0].key if results else 'n/a'})\n{'=' * 72}")
    header = (
        f"{'tier':<14}{'ms/fwd':>10}{'thrpt':>10}{'peak MB':>10}{'reserved':>10}"
        f"{'speedup':>9}{'max|Δ|':>12}{'mean|Δ|':>12}{'max rel':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        speedup = baseline_ms / row.ms_per_forward if row.ms_per_forward > 0 else float("nan")
        max_d = (
            f"{row.max_abs_diff_vs_baseline:.3e}"
            if row.max_abs_diff_vs_baseline is not None
            else "   (ref)"
        )
        mean_d = (
            f"{row.mean_abs_diff_vs_baseline:.3e}"
            if row.mean_abs_diff_vs_baseline is not None
            else "   (ref)"
        )
        rel_d = (
            f"{row.max_rel_diff_vs_baseline:.3e}"
            if row.max_rel_diff_vs_baseline is not None
            else "   (ref)"
        )
        print(
            f"{row.key:<14}"
            f"{row.ms_per_forward:>10.3f}"
            f"{row.forwards_per_sec:>10.2f}"
            f"{row.peak_alloc_mb:>10.1f}"
            f"{row.peak_reserved_mb:>10.1f}"
            f"{speedup:>8.2f}x"
            f"{max_d:>12}"
            f"{mean_d:>12}"
            f"{rel_d:>10}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Aurora full-model precision/throughput matrix benchmark.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fixed batch size. Default: auto-probe from GPU memory (--vram-fraction).",
    )
    p.add_argument(
        "--vram-fraction",
        type=float,
        default=_DEFAULT_VRAM_FRACTION,
        help="Target peak allocated fraction of total GPU memory for auto batch probe (default 0.90).",
    )
    p.add_argument(
        "--batch-cap",
        type=int,
        default=_DEFAULT_BATCH_CAP,
        help="Upper bound for auto batch-size binary search.",
    )
    p.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="Disable VRAM auto-probe; use batch_size=1.",
    )
    p.add_argument(
        "--preset",
        choices=tuple(_GRID_PRESETS),
        default="production",
        help="Synthetic grid preset (overridden by explicit --synthetic-h/--synthetic-w).",
    )
    p.add_argument(
        "--synthetic-h",
        type=int,
        default=None,
        help="Latitude grid height (default: from --preset).",
    )
    p.add_argument(
        "--synthetic-w",
        type=int,
        default=None,
        help="Longitude grid width (default: from --preset).",
    )
    p.add_argument("--history", type=int, default=2)
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[100, 250, 500, 850],
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint-dir", type=str, default=_DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--report-out", type=str, default="")
    args = p.parse_args()

    preset_h, preset_w = _GRID_PRESETS[args.preset]
    synthetic_h = preset_h if args.synthetic_h is None else args.synthetic_h
    synthetic_w = preset_w if args.synthetic_w is None else args.synthetic_w

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    if synthetic_w % 4 != 0:
        raise SystemExit("--synthetic-w must be a multiple of 4.")
    if synthetic_h % 4 not in (0, 1):
        raise SystemExit("--synthetic-h must satisfy H%4==0 or H%4==1.")
    if not (0.0 < args.vram_fraction <= 1.0):
        raise SystemExit("--vram-fraction must be in (0, 1].")
    if args.batch_cap < 1:
        raise SystemExit("--batch-cap must be >= 1.")

    from aurora.model.checkpoint_local import resolve_checkpoint_path

    device = torch.device(args.device)
    ckpt_path = resolve_checkpoint_path(
        filename="aurora-0.25-small-pretrained.ckpt",
        checkpoint_dir=args.checkpoint_dir,
        explicit_path=args.checkpoint or None,
        allow_hub_download=False,
    )

    _purge_gpu()
    state_dict = _load_shared_state_dict(ckpt_path)
    _purge_gpu()

    levels = tuple(args.levels)
    if args.batch_size is not None:
        batch_size = args.batch_size
        auto_vram_note = ""
    elif args.no_auto_batch:
        batch_size = 1
        auto_vram_note = ""
    else:
        batch_size, probe_peak_mb, total_mb, target_mb = _auto_batch_size_for_vram(
            state_dict=state_dict,
            device=device,
            h=synthetic_h,
            w=synthetic_w,
            history=args.history,
            levels=levels,
            target_fraction=args.vram_fraction,
            cap=args.batch_cap,
        )
        auto_vram_note = (
            f" auto_vram={args.vram_fraction:.0%} target={target_mb:.0f}MB "
            f"probe_peak={probe_peak_mb:.0f}MB/{total_mb:.0f}MB"
        )
        print(
            f"[auto-vram] batch_size={batch_size} "
            f"(target {args.vram_fraction:.0%} of {total_mb:.0f} MB → {target_mb:.0f} MB, "
            f"probe peak {probe_peak_mb:.0f} MB, precision={_PROBE_PRECISION})"
        )

    print(f"[config] device={torch.cuda.get_device_name(device)}")
    print(
        f"[config] batch={batch_size} preset={args.preset} "
        f"grid={synthetic_h}x{synthetic_w} warmup={args.warmup} repeat={args.repeat}"
        f"{auto_vram_note}"
    )
    print(f"[checkpoint] {ckpt_path}")

    batch = _load_batch_synthetic(
        batch_size=batch_size,
        h=synthetic_h,
        w=synthetic_w,
        history=args.history,
        levels=levels,
        device=device,
    )

    results: list[TierResult] = []
    baseline_pred: dict[str, torch.Tensor] | None = None

    _purge_gpu()
    ref_model = _build_model("fp32", state_dict, device)
    with torch.inference_mode():
        baseline_pred = _prediction_tensors(ref_model.forward(batch))
    _purge_gpu(ref_model)

    for key, precision, label in _BENCH_TIERS:
        result = _run_tier(
            key=key,
            precision=precision,
            label=label,
            state_dict=state_dict,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
            baseline_pred=None if key == "fp32" else baseline_pred,
        )
        results.append(result)
        batch = _load_batch_synthetic(
            batch_size=batch_size,
            h=synthetic_h,
            w=synthetic_w,
            history=args.history,
            levels=levels,
            device=device,
        )

    _print_summary(results)

    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Aurora precision matrix benchmark",
            "",
            f"- device: {torch.cuda.get_device_name(device)}",
            f"- checkpoint: {ckpt_path}",
            f"- batch_size: {batch_size}",
            f"- vram_fraction: {args.vram_fraction}",
            f"- preset: {args.preset}",
            f"- grid: {synthetic_h}x{synthetic_w}",
            f"- warmup/repeat: {args.warmup}/{args.repeat}",
            "",
            "| tier | precision | ms/forward | forwards/s | peak alloc MB | peak reserved MB | speedup vs baseline | max abs diff | mean abs diff |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        baseline_ms = results[0].ms_per_forward
        for row in results:
            speedup = baseline_ms / row.ms_per_forward
            max_d = "" if row.max_abs_diff_vs_baseline is None else f"{row.max_abs_diff_vs_baseline:.6e}"
            mean_d = "" if row.mean_abs_diff_vs_baseline is None else f"{row.mean_abs_diff_vs_baseline:.6e}"
            lines.append(
                f"| {row.key} | {row.precision} | {row.ms_per_forward:.3f} | "
                f"{row.forwards_per_sec:.2f} | {row.peak_alloc_mb:.1f} | {row.peak_reserved_mb:.1f} | "
                f"{speedup:.2f}x | {max_d} | {mean_d} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[report] {path.resolve()}")


if __name__ == "__main__":
    main()
