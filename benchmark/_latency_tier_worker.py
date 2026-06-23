#!/usr/bin/env python3
"""Run one preset × tier in a fresh process (subprocess worker for fair latency)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _latency_bench import run_tier_lora_modes  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--tier-label", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    batch, config = load_preset_batch(args.preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    timings = run_tier_lora_modes(
        config=config,
        ckpt=ckpt,
        precision=args.precision,
        batch=batch,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    out = {
        "preset": args.preset,
        "tier_label": args.tier_label,
        "precision": args.precision,
        "use_lora": config.variant.use_lora,
        "grid": f"{batch.spatial_shape[0]}x{batch.spatial_shape[1]}",
        "timings": {
            key: {"ms": ms, "peak_alloc_mb": peak_alloc, "peak_reserved_mb": peak_reserved}
            for key, (ms, peak_alloc, peak_reserved) in timings.items()
        },
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
