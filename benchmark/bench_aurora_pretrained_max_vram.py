#!/usr/bin/env python3
"""Maximum batch probe for AuroraPretrained on real ERA5 ingress data.

Binary-searches the largest batch size in ``[1, cap]`` for which one forward
(or rollout) completes without CUDA OOM, per inference precision tier.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_pretrained_max_vram.py \\
        --asset-root /path/to/assets

    uv run python benchmark/bench_aurora_pretrained_max_vram.py --suite legacy --cap 8
    uv run python benchmark/bench_aurora_pretrained_max_vram.py --tiers tf32 bf16 --report-peak-mb
    uv run python benchmark/bench_aurora_pretrained_max_vram.py --rollout --rollout-steps 2
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
    build_model,
    load_era5_batch,
    peak_mb_forward,
    probe_max_batch,
    purge_gpu,
    repeat_batch,
    tiers_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=_DEFAULT_ASSET_ROOT)
    parser.add_argument("--era5-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--valid-time", type=str, default="2023-01-01T06:00:00")
    parser.add_argument("--time-index", type=int, default=1)
    parser.add_argument("--cap", type=int, default=16, help="Upper bound for batch binary search (default: 16)")
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Probe rollout instead of a single forward (uses more VRAM)",
    )
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument(
        "--report-peak-mb",
        action="store_true",
        help="After probing, run once at max batch and print peak CUDA allocated (MB)",
    )
    parser.add_argument("--tiers", nargs="+", default=None)
    parser.add_argument("--suite", choices=("full", "legacy", "combos"), default="legacy")
    parser.add_argument("--combo-matrix", action="store_true")
    parser.add_argument("--combos-only", action="store_true")
    parser.add_argument("--backbone-levels", nargs="+", default=["fp32", "tf32", "bf16_mixed", "bf16"])
    parser.add_argument("--encoder-decoder-levels", nargs="+", default=["fp32", "tf32"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    checkpoint = (args.checkpoint or asset_root / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if args.cap < 1:
        raise SystemExit("--cap must be >= 1")

    valid_time = datetime.fromisoformat(args.valid_time)
    batch_b1 = load_era5_batch(
        asset_root,
        era5_cache=args.era5_cache,
        valid_time=valid_time,
        time_index=args.time_index,
    ).to(device)

    lat = batch_b1.metadata.lat.numel()
    lon = batch_b1.metadata.lon.numel()
    forward_only = not args.rollout
    mode = "forward" if forward_only else f"rollout({args.rollout_steps})"

    print(f"[config] device={torch.cuda.get_device_name(device)}")
    print(f"[config] AuroraPretrained @ {lat}x{lon}, IC batch=1")
    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] mode={mode} cap={args.cap}")

    try:
        tier_list = tiers_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"{exc}") from exc

    print(f"[config] tiers={len(tier_list)}")
    for key, spec, label in tier_list:
        print(f"  [{key}] {spec!r} — {label}")

    print(f"\n{'tier':<42} {'max_batch':>10} {'peak_MB':>10}")
    print("-" * 64)

    for key, precision, label in tier_list:
        print(f"[probe] {key}...", flush=True)
        purge_gpu()
        model = build_model(precision, checkpoint, device)
        try:
            max_batch = probe_max_batch(
                model=model,
                batch_b1=batch_b1,
                cap=args.cap,
                device=device,
                forward_only=forward_only,
                rollout_steps=args.rollout_steps,
            )
            peak_mb: float | str = "n/a"
            if args.report_peak_mb and max_batch >= 1:
                batch_max = repeat_batch(batch_b1, max_batch).to(device)
                peak_mb = peak_mb_forward(model, batch_max, device)
                purge_gpu(batch_max)
        finally:
            purge_gpu(model)

        peak_s = f"{peak_mb:.0f}" if isinstance(peak_mb, float) else peak_mb
        print(f"{key:<42} {max_batch:>10} {peak_s:>10}")
        if max_batch == 0:
            print(f"  [warn] {key}: batch=1 failed ({label})")

    purge_gpu(batch_b1)


if __name__ == "__main__":
    main()
