#!/usr/bin/env python3
"""One closed-loop tier in its own process: a single model on the GPU."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from flash_aurora.engine.core.asset_root import resolve_asset_root  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402
from bench_rollout_drift import (  # noqa: E402
    _synchronize,
    build_model,
    pin_temp_to_data_disk,
    require_perceiver_fp32,
    rollout_variable,
)

import torch


def stream_tier(queue, preset: str, precision: str, steps: int, asset_root: str, variable: str) -> None:
    """Roll one model. Each step sends only ``variable`` to the parent, then frees the GPU."""
    try:
        root = Path(asset_root)
        os.environ["AURORA_ASSET_ROOT"] = str(root)
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        pin_temp_to_data_disk(root)
        require_perceiver_fp32(precision)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        device = torch.device("cuda")
        import time

        started = time.perf_counter()
        batch, config = load_preset_batch(preset, root)
        ckpt = checkpoint_path(config, root)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt}")
        model = build_model(config, ckpt, precision=precision, device=device)
        _synchronize(device)
        load_s = time.perf_counter() - started

        def send_field(index: int, tensor: torch.Tensor) -> None:
            queue.put(("field", tensor))

        forecast_started = time.perf_counter()
        peak_gib, _inner = rollout_variable(
            model,
            batch,
            steps=steps,
            device=device,
            on_field=send_field,
            variable=variable,
        )
        _synchronize(device)
        forecast_s = time.perf_counter() - forecast_started
        del model, batch
        torch.cuda.empty_cache()
        queue.put(
            (
                "done",
                {
                    "peak_gib": peak_gib,
                    "load_s": load_s,
                    "forecast_s": forecast_s,
                    "per_step_s": forecast_s / steps if steps else 0.0,
                },
            )
        )
    except Exception:
        queue.put(("error", traceback.format_exc()))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    args = parser.parse_args()
    if args.asset_root is None:
        args.asset_root = resolve_asset_root()
    if args.asset_root is None:
        raise SystemExit("pass --asset-root or export AURORA_ASSET_ROOT")
    import multiprocessing as mp

    queue = mp.get_context("spawn").Queue()
    stream_tier(queue, args.preset, args.precision, args.steps, str(args.asset_root), args.variable)


if __name__ == "__main__":
    main()
