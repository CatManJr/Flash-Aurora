#!/usr/bin/env python3
"""One closed-loop precision tier in a fresh process (VRAM isolation)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402
from bench_rollout_drift import (  # noqa: E402
    _atomic_save,
    _synchronize,
    build_model,
    require_perceiver_fp32,
    rollout_tensors,
)

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--step-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    require_perceiver_fp32(args.precision)
    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    step_dir = args.step_dir.expanduser().resolve()
    step_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    batch, config = load_preset_batch(args.preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    model = build_model(config, ckpt, precision=args.precision, device=device)
    _synchronize(device)
    load_s = time.perf_counter() - t_load

    t_forecast = time.perf_counter()
    preds, peak_gib, _inner_s = rollout_tensors(
        model, batch, steps=args.steps, device=device
    )
    _synchronize(device)
    forecast_s = time.perf_counter() - t_forecast
    del model

    for index, tensors in enumerate(preds, start=1):
        path = step_dir / f"step_{index:04d}.pt"
        _atomic_save(tensors, path)
        print(json.dumps({"event": "step", "index": index, "path": str(path)}), flush=True)
    del preds

    per_step_s = forecast_s / args.steps if args.steps else 0.0
    print(
        json.dumps(
            {
                "event": "done",
                "peak_gib": peak_gib,
                "load_s": load_s,
                "forecast_s": forecast_s,
                "per_step_s": per_step_s,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
