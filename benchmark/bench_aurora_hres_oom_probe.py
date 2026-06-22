#!/usr/bin/env python3
"""Probe AuroraHighRes (0.1°) forward with synthetic IC — OOM smoke test.

Uses fake Gaussian fields + official static pickle (1801×3600). Does not
require HRES NetCDF ingress.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_hres_oom_probe.py
    uv run python benchmark/bench_aurora_hres_oom_probe.py --use-lora --precision fp32
    uv run python benchmark/bench_aurora_hres_oom_probe.py --batch 2
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import numpy as np
import torch

from _pretrained_era5 import cuda_oom_like, purge_gpu, recover_cuda_after_oom

_DEFAULT_ASSET_ROOT = Path("/root/autodl-tmp/aurora")
_CHECKPOINT_NAME = "aurora-0.1-finetuned.ckpt"
_STATIC_NAME = "aurora-0.1-static.pickle"
_STANDARD_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def make_synthetic_hres_batch(
    asset_root: Path,
    *,
    height: int = 1801,
    width: int = 3600,
    batch: int = 1,
    history: int = 2,
) -> "Batch":
    from aurora import Batch, Metadata

    lat = torch.from_numpy(np.linspace(90, -90, height)).float()
    lon = torch.from_numpy(np.linspace(0, 360, width, endpoint=False)).float()
    surf = {k: torch.randn(batch, history, height, width) for k in ("2t", "10u", "10v", "msl")}
    atmos = {
        k: torch.randn(batch, history, len(_STANDARD_LEVELS), height, width)
        for k in ("z", "u", "v", "t", "q")
    }
    static_path = asset_root / _STATIC_NAME
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static pickle: {static_path}")
    with open(static_path, "rb") as f:
        static = pickle.load(f)
    static_vars = {k: torch.as_tensor(v).float() for k, v in static.items()}
    sh, sw = next(iter(static_vars.values())).shape
    if (sh, sw) != (height, width):
        raise ValueError(f"static grid {sh}x{sw} != requested {height}x{width}")

    times = (datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6))[:history]
    return Batch(
        surf_vars=surf,
        static_vars=static_vars,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=times,
            atmos_levels=_STANDARD_LEVELS,
        ),
    )


def build_hres_model(
    checkpoint: Path,
    *,
    precision: str,
    use_lora: bool,
    device: torch.device,
):
    from aurora.model.aurora import AuroraHighRes

    model = AuroraHighRes(use_lora=use_lora, inference_precision=precision)
    model.load_checkpoint_local(str(checkpoint), strict=False)
    model.eval()
    return model.to(device)


def ic_fp32_gb(batch) -> float:
    n = sum(v.numel() for v in batch.surf_vars.values())
    n += sum(v.numel() for v in batch.atmos_vars.values())
    n += sum(v.numel() for v in batch.static_vars.values())
    return n * 4 / 1e9


def try_forward(
    model,
    batch,
    device: torch.device,
) -> tuple[str, float | None, float | None, str | None]:
    """Return (status, peak_alloc_gb, peak_reserved_gb, error_message)."""
    purge_gpu()
    batch = batch.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            _ = model.forward(batch)
        torch.cuda.synchronize(device)
    except Exception as exc:
        if cuda_oom_like(exc):
            recover_cuda_after_oom()
            return "OOM", None, None, str(exc).split("\n")[0]
        recover_cuda_after_oom()
        return "ERROR", None, None, f"{type(exc).__name__}: {exc}"

    alloc = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.max_memory_reserved(device) / 1e9
    purge_gpu(model, batch)
    return "OK", alloc, reserved, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=_DEFAULT_ASSET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--precision",
        default="bf16_mixed",
        choices=("bf16_mixed", "fp32", "tf32", "bf16"),
    )
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA (default finetuned path)")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    checkpoint = (args.checkpoint or asset_root / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    print(f"[gpu] {torch.cuda.get_device_name(device)} total={total_mb:.0f} MiB")

    batch = make_synthetic_hres_batch(asset_root, batch=args.batch)
    h, w = batch.spatial_shape
    print(f"[ic] synthetic fake data {h}x{w} batch={args.batch} ic_fp32≈{ic_fp32_gb(batch):.2f} GB")
    print(f"[model] AuroraHighRes precision={args.precision} use_lora={args.use_lora}")
    print(f"[ckpt] {checkpoint}")

    model = build_hres_model(
        checkpoint,
        precision=args.precision,
        use_lora=args.use_lora,
        device=device,
    )
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"[weights] {weights_gb:.2f} GB on GPU")

    print("[run] forward...", flush=True)
    status, alloc, reserved, err = try_forward(model, batch, device)

    if status == "OK":
        headroom = total_mb / 1024 - reserved
        print(f"[result] OK — peak allocated={alloc:.1f} GB reserved={reserved:.1f} GB")
        print(f"  96GB headroom (physical − reserved): {headroom:.1f} GB")
        if reserved > total_mb / 1024 * 0.92:
            print("  [warn] reserved >92% of physical VRAM — fragile on 96GB cards")
    elif status == "OOM":
        print(f"[result] OOM — {err}")
    else:
        print(f"[result] ERROR (not OOM) — {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
