#!/usr/bin/env python3
"""ROI egress bytes and host-transfer time versus a global dump (Group A).

One GPU-to-CPU copy per step, then CPU clip. Named masks from docs/roi and
``RoiBatch.example_batch()`` are timed independently of isolate-tiers latency.

    export AURORA_ASSET_ROOT=/path/to/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_roi_io.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _pretrained_era5 import purge_gpu  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402
from bench_aurora_precision_all import build_model, set_benchmark_seed  # noqa: E402
from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout  # noqa: E402
from flash_aurora.engine.egress.mask import Mask  # noqa: E402
from flash_aurora.engine.egress.offload import owned_cpu_copy  # noqa: E402
from flash_aurora.engine.egress.roi import apply_mask  # noqa: E402
from flash_aurora.engine.egress.roi_batch import RoiBatch  # noqa: E402

import torch

SEED = 42
DEFAULT_PRESET = "era5_pretrained"
DEFAULT_PRECISION = "bf16_mixed@fp32"
DEFAULT_WARMUP = 2
DEFAULT_REPEAT = 5
ROI_DIR = Path(_REPO) / "docs" / "roi"
DOC_ROI_NAMES: tuple[str, ...] = ("california", "georgia", "michigan", "texas")


def _iter_batch_tensors(batch: Any):
    for group in (batch.surf_vars, batch.static_vars, batch.atmos_vars):
        yield from group.values()
    yield batch.metadata.lat
    yield batch.metadata.lon


def batch_nbytes(batch: Any) -> int:
    return int(sum(t.numel() * t.element_size() for t in _iter_batch_tensors(batch)))


def _forward_gpu(model: Any, batch: Any, device: torch.device) -> Any:
    set_benchmark_seed(SEED)
    dev_batch = batch.to(device)
    with torch.inference_mode():
        if model_uses_v1p5_rollout(model):
            batch_size = next(iter(dev_batch.surf_vars.values())).shape[0]
            param = next(model.parameters())
            lead_hours = model.timestep.total_seconds() / 3600.0
            lead_times = torch.full(
                (batch_size,),
                lead_hours,
                device=param.device,
                dtype=param.dtype,
            )
            return model.forward(dev_batch, lead_times=lead_times)
        return model.forward(dev_batch)


def _cuda_ms(fn, *, warmup: int, repeat: int) -> tuple[float, float, Any]:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    last = None
    for _ in range(repeat):
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    arr = np.asarray(samples, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), last


def _doc_roi_batch(roi_dir: Path) -> RoiBatch:
    regions = {
        name: Mask.from_geojson(roi_dir / f"{name}.geojson") for name in DOC_ROI_NAMES
    }
    return RoiBatch.from_mapping(regions)


def _clip_rows(cpu_batch: Any, roi_batch: RoiBatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, mask in roi_batch.regions:
        clipped = apply_mask(cpu_batch, mask)
        rows.append({"name": name, "nbytes": batch_nbytes(clipped)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--roi-dir", type=Path, default=ROI_DIR)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(_BENCH_DIR) / "roi_io_latest.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(_BENCH_DIR) / "roi_io_latest.md",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    set_benchmark_seed(SEED)
    device = torch.device("cuda")
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    gpu = torch.cuda.get_device_name(device)
    print(f"[gpu] {gpu}", flush=True)
    print(f"[preset] {args.preset} precision={args.precision}", flush=True)

    batch, config = load_preset_batch(args.preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    model = build_model(config, ckpt, precision=args.precision, device=device)
    try:
        pred = _forward_gpu(model, batch, device)
        mean_ms, std_ms, cpu_pred = _cuda_ms(
            lambda: owned_cpu_copy(pred),
            warmup=args.warmup,
            repeat=args.repeat,
        )
    finally:
        purge_gpu(model)

    global_bytes = batch_nbytes(cpu_pred)
    packs = {
        "example_batch": RoiBatch.example_batch(),
        "docs_roi": _doc_roi_batch(args.roi_dir.expanduser().resolve()),
    }
    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "torch": torch.__version__,
        "preset": args.preset,
        "precision": args.precision,
        "seed": SEED,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "global": {
            "nbytes": global_bytes,
            "d2h_mean_ms": mean_ms,
            "d2h_std_ms": std_ms,
        },
        "packs": {},
    }
    lines = [
        "# ROI bytes versus global dump",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {gpu}",
        f"- Preset: `{args.preset}` / `{args.precision}`",
        f"- Isolate process, seed {SEED}, D2H warmup {args.warmup}, n={args.repeat}",
        "",
        f"Global dump: {global_bytes / (1024 ** 2):.1f} MiB, "
        f"D2H {mean_ms:.2f}±{std_ms:.2f} ms.",
        "",
        "| pack | region | MiB | vs global |",
        "| --- | --- | ---: | ---: |",
    ]
    print(
        f"  global {global_bytes / (1024 ** 2):.1f} MiB  "
        f"d2h={mean_ms:.2f}±{std_ms:.2f} ms",
        flush=True,
    )
    for pack_name, roi_batch in packs.items():
        rows = _clip_rows(cpu_pred, roi_batch)
        payload["packs"][pack_name] = {
            "nbytes_sum": int(sum(row["nbytes"] for row in rows)),
            "regions": rows,
        }
        for row in rows:
            ratio = row["nbytes"] / global_bytes if global_bytes else 0.0
            lines.append(
                f"| `{pack_name}` | `{row['name']}` | "
                f"{row['nbytes'] / (1024 ** 2):.2f} | {ratio:.3f} |"
            )
            print(
                f"  {pack_name}/{row['name']}: {row['nbytes'] / (1024 ** 2):.2f} MiB "
                f"({ratio:.3f} of global)",
                flush=True,
            )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)
    print(f"wrote {args.md_out}", flush=True)


if __name__ == "__main__":
    main()
