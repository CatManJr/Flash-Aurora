#!/usr/bin/env python3
"""Freeze one-step native PyTorch unfused-FP32 dumps (Group A).

Each invocation loads one preset, runs the isolate-process FP32 twin, and
writes tensors plus a manifest. Existing preset files are left in place.

    export AURORA_ASSET_ROOT=/path/to/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/freeze_pytorch_ref.py \\
        --preset era5_pretrained --dump-dir /path/to/dumps/pytorch-ref-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _pretrained_era5 import (  # noqa: E402
    _PYTORCH_BASELINE_KEY,
    prediction_tensors,
    purge_gpu,
)
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402
from bench_aurora_precision_all import (  # noqa: E402
    build_model,
    run_forward_tensors,
    set_benchmark_seed,
)

import torch

SEED = 42
TIER_LABEL = _PYTORCH_BASELINE_KEY
TIER_PRECISION = "fp32"


def _tensor_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors.values()))


def _ic_stamp(batch: Any) -> str:
    times = getattr(batch.metadata, "time", ())
    if not times:
        return "unknown"
    first = times[0]
    if hasattr(first, "isoformat"):
        return first.isoformat()
    return str(first)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing preset dump. Default is to skip.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dump_dir = args.dump_dir.expanduser().resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = dump_dir / f"{args.preset}.pt"
    manifest_path = dump_dir / "manifest.json"
    if tensor_path.is_file() and not args.overwrite:
        print(f"[skip] {tensor_path} exists; pass --overwrite to replace", flush=True)
        return

    set_benchmark_seed(args.seed)
    device = torch.device("cuda")
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    gpu = torch.cuda.get_device_name(device)
    print(f"[gpu] {gpu}", flush=True)
    print(f"[preset] {args.preset}", flush=True)

    batch, config = load_preset_batch(args.preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    height, width = batch.spatial_shape
    print(f"  grid={height}x{width} ckpt={ckpt.name}", flush=True)

    model = build_model(config, ckpt, precision=TIER_PRECISION, device=device)
    try:
        tensors = run_forward_tensors(model, batch, device=device)
    finally:
        purge_gpu(model)

    payload = {
        "preset": args.preset,
        "tier": TIER_LABEL,
        "precision": TIER_PRECISION,
        "seed": args.seed,
        "tensors": tensors,
    }
    torch.save(payload, tensor_path)
    nbytes = _tensor_nbytes(tensors)
    preset_record = {
        "ic": _ic_stamp(batch),
        "checkpoint": ckpt.name,
        "grid": [int(height), int(width)],
        "n_tensors": len(tensors),
        "nbytes": nbytes,
        "keys": sorted(tensors),
        "path": tensor_path.name,
    }
    manifest = _load_manifest(manifest_path)
    if not manifest:
        manifest = {
            "dump_id": dump_dir.name,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "gpu": gpu,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH", ""),
            "seed": args.seed,
            "tier": TIER_LABEL,
            "precision": TIER_PRECISION,
            "harness": "one-step isolate process per preset",
            "presets": {},
        }
    manifest["presets"][args.preset] = preset_record
    manifest["updated"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {tensor_path} ({nbytes / (1024 ** 3):.3f} GiB)", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
