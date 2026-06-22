#!/usr/bin/env python3
"""End-to-end precision benchmark for AuroraPretrained on real ERA5 ingress data.

Loads cached CDS NetCDF under ``<asset-root>/era5/``, runs each inference tier,
times ``forward``, and compares outputs against the PyTorch FP32 baseline.

**Default suite (10 tiers)** - same grid as ``bench_small_pretrained.py``:

1. PyTorch: ``backbone=fp32, encoder/decoder=fp32`` (accuracy baseline)
2. PyTorch: ``backbone=autocast_bf16, encoder/decoder=fp32``
3. Eight explicit custom combos ``{fp32,tf32,bf16_mixed,bf16}@{fp32,tf32}``

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_pretrained.py \\
        --asset-root /path/to/assets

    uv run python benchmark/bench_aurora_pretrained.py --suite legacy --warmup 1 --repeat 1
    uv run python benchmark/bench_aurora_pretrained.py --tiers tf32 bf16_mixed@fp32

    # CuTe window attn vs SDPA (per-variable max_abs table)
    uv run python benchmark/bench_aurora_pretrained.py --ablate-cute --warmup 1 --repeat 1
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402


import torch

from _pretrained_era5 import (  # noqa: E402
    _CHECKPOINT_NAME,
    _DEFAULT_ASSET_ROOT,
    _PYTORCH_BASELINE_KEY,
    diff_vs_reference,
    format_peak_memory,
    load_era5_batch,
    official_tol_rows,
    print_official_tol_table,
    print_per_variable_table,
    print_startup_gpu_state,
    print_summary_table,
    purge_gpu,
    run_ablate_cute,
    run_tier,
    tiers_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=_DEFAULT_ASSET_ROOT,
        help=f"Asset directory with checkpoint + era5/ cache (default: {_DEFAULT_ASSET_ROOT})",
    )
    parser.add_argument(
        "--era5-cache",
        type=Path,
        default=None,
        help="ERA5 NetCDF cache (default: <asset-root>/era5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint path (default: <asset-root>/{_CHECKPOINT_NAME})",
    )
    parser.add_argument(
        "--valid-time",
        type=str,
        default="2023-01-01T06:00:00",
        help="IC valid time matching cached NetCDF day (default: 2023-01-01 06:00 UTC)",
    )
    parser.add_argument("--time-index", type=int, default=1, help="Time index in daily NetCDF (default: 1)")
    parser.add_argument("--tiers", nargs="+", default=None, help="Explicit tier names or combo strings")
    parser.add_argument(
        "--suite",
        choices=("full", "legacy", "combos"),
        default="full",
        help="full (default): 2 PyTorch refs + 8 custom combos; legacy: named presets",
    )
    parser.add_argument("--combo-matrix", action="store_true", help="Override combo level lists")
    parser.add_argument("--combos-only", action="store_true", help="Skip PyTorch reference tiers")
    parser.add_argument(
        "--backbone-levels",
        nargs="+",
        default=["fp32", "tf32", "bf16_mixed", "bf16"],
    )
    parser.add_argument("--encoder-decoder-levels", nargs="+", default=["fp32", "tf32"])
    parser.add_argument(
        "--no-official-tol",
        action="store_true",
        help="Skip per-variable tolerance tables vs fp32 baseline",
    )
    parser.add_argument(
        "--no-per-var",
        action="store_true",
        help="Skip per-variable max_abs table after the summary (default: print)",
    )
    parser.add_argument(
        "--ablate-cute",
        action="store_true",
        help="CuTe window attn vs SDPA ablation on tf32/bf16_mixed (instead of tier suite)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before timing (default: 1)")
    parser.add_argument("--repeat", type=int, default=3, help="Timed forwards per tier (default: 3)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    checkpoint = (args.checkpoint or asset_root / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    print_startup_gpu_state(device=device)

    valid_time = datetime.fromisoformat(args.valid_time)
    batch = load_era5_batch(
        asset_root,
        era5_cache=args.era5_cache,
        valid_time=valid_time,
        time_index=args.time_index,
    ).to(device)

    lat = batch.metadata.lat.numel()
    lon = batch.metadata.lon.numel()
    bs = next(iter(batch.surf_vars.values())).shape[0]
    print(f"[config] AuroraPretrained @ {lat}x{lon}, batch={bs}")
    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] valid_time={valid_time} time_index={args.time_index}")
    print(f"[config] timing warmup={args.warmup} repeat={args.repeat}")

    if args.ablate_cute:
        run_ablate_cute(
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        purge_gpu(batch)
        return

    try:
        tier_list = tiers_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"{exc}\nUse --suite full, combo strings, or --suite legacy.") from exc

    print(f"[config] suite={args.suite} tiers={len(tier_list)}")
    for key, spec, label in tier_list:
        print(f"  [{key}] inference_precision={spec!r}")
        print(f"           {label}")

    baseline_key = _PYTORCH_BASELINE_KEY
    if baseline_key not in {t[0] for t in tier_list}:
        baseline_key = tier_list[0][0]
        print(f"[warn] PyTorch FP32 baseline not in run; using {baseline_key!r} for diffs")

    baseline = None
    baseline_ms: float | None = None
    all_preds: dict[str, dict] = {}
    rows: list[tuple[str, str, float, float, float, float, float, float | None]] = []

    for key, precision, label in tier_list:
        print(f"[run] {key} ({label})...", flush=True)
        pred, ms_per, peak_alloc, peak_reserved = run_tier(
            precision=precision,
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_preds[key] = pred
        print(
            f"[run] {key} forward={ms_per:.1f} ms ({1000.0 / ms_per:.2f} fwd/s) "
            f"{format_peak_memory(peak_alloc, peak_reserved)}",
            flush=True,
        )
        if key == baseline_key:
            baseline = pred
            baseline_ms = ms_per
            rows.append((key, label, ms_per, None, 0.0, 0.0, 0.0, 1.0))
            continue
        assert baseline is not None and baseline_ms is not None
        max_abs, mean_abs, max_rel, cos = diff_vs_reference(baseline, pred)
        rows.append((key, label, ms_per, baseline_ms / ms_per, max_abs, mean_abs, max_rel, cos))

    print_summary_table(rows)

    if baseline is not None and not args.no_per_var:
        non_baseline = tuple(k for k in all_preds if k != baseline_key)
        print_per_variable_table(
            f"Per-variable max_abs vs {baseline_key}",
            baseline,
            all_preds,
            tier_order=non_baseline,
        )

    if not args.no_official_tol and baseline is not None:
        for tier_key, pred in all_preds.items():
            if tier_key == baseline_key:
                continue
            tol_rows = official_tol_rows(baseline, pred)
            print_official_tol_table(f"[official tol] {tier_key} vs {baseline_key}", tol_rows)

    purge_gpu(batch)


if __name__ == "__main__":
    main()
