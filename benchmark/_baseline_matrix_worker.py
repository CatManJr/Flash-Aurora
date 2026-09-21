#!/usr/bin/env python3
"""Run one Group A baseline-matrix row in a fresh process."""

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
from _baseline_matrix import get_row, row_ids, run_baseline_row  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, choices=row_ids())
    parser.add_argument("--preset", required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    row = get_row(args.row)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    print(f"[baseline-worker] row={row.row_id} preset={args.preset}", file=sys.stderr, flush=True)
    batch, config = load_preset_batch(args.preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    payload = run_baseline_row(
        row=row,
        config=config,
        ckpt=ckpt,
        batch=batch,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    payload["preset"] = args.preset
    print(json.dumps(payload))
    if not payload.get("ok", False):
        raise SystemExit(payload.get("error", "row failed"))


if __name__ == "__main__":
    main()
