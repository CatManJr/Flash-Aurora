#!/usr/bin/env python3
"""Simple precision benchmark for AuroraSmallPretrained on HF test batch.

Loads ``aurora-0.25-small-pretrained-test-input.pickle`` + ``aurora-0.25-static.pickle``,
runs each inference tier, times end-to-end ``forward``, and compares outputs against ``fp32``.
Prints per-variable **official tolerances** from ``tests/test_model.py`` (mean|err|/mean|ref|).

Examples::

    uv run python benchmark/bench_small_pretrained.py
    uv run python benchmark/bench_small_pretrained.py --compare-hf
    uv run python benchmark/bench_small_pretrained.py --tiers fp32 fast_fp32 bf16_mixed
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_AURORA_PKG = _REPO / "aurora"
if _AURORA_PKG.is_dir():
    sys.path.insert(0, str(_AURORA_PKG))

_DEFAULT_DATA_DIR = "/root/autodl-tmp/aurora"
_CHECKPOINT_NAME = "aurora-0.25-small-pretrained.ckpt"
_INPUT_NAME = "aurora-0.25-small-pretrained-test-input.pickle"
_STATIC_NAME = "aurora-0.25-static.pickle"
_OUTPUT_NAME = "aurora-0.25-small-pretrained-test-output.pickle"

_DEFAULT_TIERS: tuple[tuple[str, str], ...] = (
    ("fp32", "PyTorch FP32"),
    ("fast_fp32", "Triton layout + AdaLN + PyTorch GELU"),
    ("tf32_1x", "fast_fp32 + TF32 matmul + CuTe TF32 attn"),
    ("bf16_mixed", "fast_fp32 + BF16 backbone + CuTe BF16 attn"),
    ("pytorch_autocast", "PyTorch backbone BF16 autocast"),
)

# Same relative mean error gates as aurora/tests/test_model.py::test_aurora_small
_OFFICIAL_TOLERANCES: dict[str, float] = {
    "2t": 1e-4,
    "10u": 5e-3,
    "10v": 5e-3,
    "msl": 1e-4,
    "u": 5e-3,
    "v": 5e-3,
    "t": 1e-4,
    "q": 5e-3,
}

_OFFICIAL_VAR_ORDER: tuple[tuple[str, str], ...] = (
    ("surf_vars", "2t"),
    ("surf_vars", "10u"),
    ("surf_vars", "10v"),
    ("surf_vars", "msl"),
    ("atmos_vars", "u"),
    ("atmos_vars", "v"),
    ("atmos_vars", "t"),
    ("atmos_vars", "q"),
)


def _purge_gpu(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _load_batch(data_dir: Path) -> Any:
    from aurora import Batch, Metadata
    from aurora.batch import interpolate_numpy

    input_path = data_dir / _INPUT_NAME
    static_path = data_dir / _STATIC_NAME
    if not input_path.is_file():
        raise FileNotFoundError(f"missing test input: {input_path}")
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static vars: {static_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(input_path, "rb") as f:
            test_input = pickle.load(f)
        with open(static_path, "rb") as f:
            static_vars = pickle.load(f)

    if os.name == "nt":

        class PatchedDateTime(datetime):
            def timestamp(self) -> float:
                return -631134000.0

        test_input["metadata"]["time"] = [PatchedDateTime(1950, 1, 1, 6, 0)]

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    return Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )


def _load_hf_output_tensors(data_dir: Path) -> dict[str, torch.Tensor] | None:
    output_path = data_dir / _OUTPUT_NAME
    if not output_path.is_file():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(output_path, "rb") as f:
            test_output = pickle.load(f)
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, arr in test_output[group].items():
            out[f"{group}.{name}"] = torch.from_numpy(np.asarray(arr)).float()
    return out


def _prediction_tensors(pred: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, tensor in getattr(pred, group).items():
            out[f"{group}.{name}"] = tensor.detach().float().cpu()
    return out


def _diff_vs_reference(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, float, float, float]:
    max_diff = 0.0
    max_rel = 0.0
    total = 0.0
    cos_total = 0.0
    count = 0
    for key, ref in reference.items():
        cand = candidate[key]
        diff = (cand - ref).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        denom = ref.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        total += float(diff.mean().item())

        ref_flat = ref.flatten().double()
        cand_flat = cand.flatten().double()
        ref_norm = ref_flat.norm()
        cand_norm = cand_flat.norm()
        if ref_norm.item() == 0.0 and cand_norm.item() == 0.0:
            cos = 1.0
        elif ref_norm.item() == 0.0 or cand_norm.item() == 0.0:
            cos = 0.0
        else:
            cos = float(torch.dot(ref_flat, cand_flat).item() / (ref_norm.item() * cand_norm.item()))
        cos_total += cos
        count += 1
    return max_diff, total / max(count, 1), max_rel, cos_total / max(count, 1)


def _official_tol_rows(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> list[tuple[str, float, float, float, bool]]:
    """Per-variable mean(|err|)/mean(|ref|) vs official tol (test_model.py)."""
    rows: list[tuple[str, float, float, float, bool]] = []
    for group, name in _OFFICIAL_VAR_ORDER:
        key = f"{group}.{name}"
        ref = reference[key]
        cand = candidate[key]
        err = (cand - ref).abs()
        mean_rel = float(err.mean().item() / ref.abs().mean().clamp_min(1e-8).item())
        tol = _OFFICIAL_TOLERANCES[name]
        max_abs = float(err.max().item())
        rows.append((name, mean_rel, tol, max_abs, mean_rel <= tol))
    return rows


def _print_official_tol_table(title: str, rows: list[tuple[str, float, float, float, bool]]) -> None:
    print(f"\n{title}")
    print("  metric: mean(|out-ref|) / mean(|ref|)  (aurora/tests/test_model.py)")
    print(f"  {'var':<6} {'mean_rel':>10} {'tol':>10} {'max_abs':>10} {'ok':>4}")
    print("  " + "-" * 44)
    for name, mean_rel, tol, max_abs, ok in rows:
        mark = "yes" if ok else "NO"
        print(f"  {name:<6} {mean_rel:10.4e} {tol:10.4e} {max_abs:10.4g} {mark:>4}")
    passed = sum(1 for r in rows if r[4])
    print(f"  summary: {passed}/{len(rows)} variables within official tolerance")


def _build_model(precision: str, checkpoint: Path, device: torch.device) -> Any:
    from aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision=precision)
    model.load_checkpoint_local(str(checkpoint), strict=True)
    model.eval()
    return model.to(device)


def _time_forward(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[Any, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
        else:
            import time

            t0 = time.perf_counter()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            ms_total = (time.perf_counter() - t0) * 1e3

    return pred, ms_total / repeat


def _run_tier(
    *,
    precision: str,
    checkpoint: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, torch.Tensor], float]:
    model = _build_model(precision, checkpoint, device)
    try:
        pred, ms_per = _time_forward(model, batch, warmup=warmup, repeat=repeat, device=device)
        return _prediction_tensors(pred), ms_per
    finally:
        _purge_gpu(model)


def _print_table(
    rows: list[tuple[str, str, float, float, float, float, float, float | None]],
) -> None:
    print(
        f"\n{'tier':<18} {'ms':>8} {'speedup':>8} {'max_abs':>10} {'mean_abs':>10} "
        f"{'max_rel':>10} {'cos_sim':>8}"
    )
    print("-" * 88)
    for key, _label, ms, speedup, max_abs, mean_abs, max_rel, cos in rows:
        speedup_s = f"{speedup:.2f}x" if speedup is not None else "  base"
        print(
            f"{key:<18} {ms:8.1f} {speedup_s:>8} {max_abs:10.4g} {mean_abs:10.4g} "
            f"{max_rel:10.4g} {cos:8.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(_DEFAULT_DATA_DIR),
        help=f"Directory with HF pickles (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint path (default: <data-dir>/{_CHECKPOINT_NAME})",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=[t[0] for t in _DEFAULT_TIERS],
        help="Inference precision tiers to run (default: all five)",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Also run official tol table for fp32 vs HF test-output.pickle",
    )
    parser.add_argument(
        "--no-official-tol",
        action="store_true",
        help="Skip per-variable official tolerance tables (default: print for each tier vs fp32)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before timing (default: 1)")
    parser.add_argument("--repeat", type=int, default=3, help="Timed forwards per tier (default: 3)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or data_dir / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    batch = _load_batch(data_dir).to(device)
    lat = batch.metadata.lat.numel()
    lon = batch.metadata.lon.numel()
    print(f"[config] AuroraSmallPretrained @ {lat}x{lon}, batch={next(iter(batch.surf_vars.values())).shape[0]}")
    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] data_dir={data_dir}")
    print(f"[config] timing warmup={args.warmup} repeat={args.repeat}")

    tier_labels = dict(_DEFAULT_TIERS)
    unknown = [t for t in args.tiers if t not in tier_labels]
    if unknown:
        raise SystemExit(f"unknown tier(s): {unknown}; expected one of: {', '.join(tier_labels)}")

    if "fp32" not in args.tiers:
        print("[warn] fp32 not in --tiers; using first tier as baseline for relative diffs")

    baseline_key = "fp32" if "fp32" in args.tiers else args.tiers[0]
    baseline: dict[str, torch.Tensor] | None = None
    baseline_ms: float | None = None
    all_preds: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[tuple[str, str, float, float, float, float, float, float | None]] = []

    for tier in args.tiers:
        print(f"[run] {tier}...", flush=True)
        pred, ms_per = _run_tier(
            precision=tier,
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_preds[tier] = pred
        print(f"[run] {tier} e2e forward={ms_per:.1f} ms ({1000.0 / ms_per:.2f} fwd/s)", flush=True)
        if tier == baseline_key:
            baseline = pred
            baseline_ms = ms_per
            rows.append((tier, tier_labels.get(tier, tier), ms_per, None, 0.0, 0.0, 0.0, 1.0))
            continue
        assert baseline is not None and baseline_ms is not None
        max_abs, mean_abs, max_rel, cos = _diff_vs_reference(baseline, pred)
        speedup = baseline_ms / ms_per
        rows.append((tier, tier_labels.get(tier, tier), ms_per, speedup, max_abs, mean_abs, max_rel, cos))

    _print_table(rows)

    if not args.no_official_tol and baseline is not None:
        for tier, pred in all_preds.items():
            if tier == baseline_key:
                continue
            tol_rows = _official_tol_rows(baseline, pred)
            _print_official_tol_table(f"[official tol] {tier} vs {baseline_key}", tol_rows)

    if args.compare_hf:
        hf_ref = _load_hf_output_tensors(data_dir)
        if hf_ref is None:
            print(f"\n[compare-hf] skipped: {_OUTPUT_NAME} not found under {data_dir}")
        elif baseline is None:
            print("\n[compare-hf] skipped: no fp32 baseline in this run")
        else:
            max_abs, mean_abs, max_rel, cos = _diff_vs_reference(hf_ref, baseline)
            print(
                f"\n[compare-hf] pooled fp32 vs HF test-output: "
                f"max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} max_rel={max_rel:.4g} cos_sim={cos:.6f}"
            )
            tol_rows = _official_tol_rows(hf_ref, baseline)
            _print_official_tol_table("[official tol] fp32 vs HF test-output", tol_rows)


if __name__ == "__main__":
    main()
