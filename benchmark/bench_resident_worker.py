#!/usr/bin/env python3
"""Resident-worker check: one engine serves consecutive jobs (Group B).

A scheduler worker keeps its model on the GPU between jobs. This runs N jobs
through one engine and reports per-job prepare time, rollout time and peak GiB.
Exit status is non-zero when the model was rebuilt, the peak grew, or a rollout
slowed down (thresholds in ``_resident_worker_checks``).

    export AURORA_ASSET_ROOT=/path/to/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_resident_worker.py \
        --preset era5_pretrained --jobs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
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
from _preset_ic import load_preset_ingest_request  # noqa: E402
from _resident_worker_checks import JobMeasurement, find_violations  # noqa: E402
from flash_aurora.engine.core.engine import AuroraEngine  # noqa: E402

import torch  # noqa: E402

DEFAULT_PRESET = "era5_pretrained"
DEFAULT_PRECISION = "bf16_mixed@fp32"
DEFAULT_JOBS = 5
DEFAULT_STEPS = 1
DEFAULT_DEVICE = "cuda:0"
REPORT_DIR = Path(_REPO) / "groupB" / "reports"
_BYTES_PER_GIB = 1024**3


def measure_job(engine: AuroraEngine, request: Any, *, job_index: int, steps: int) -> JobMeasurement:
    device = torch.device(engine.config.device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    started_s = time.perf_counter()
    batch = engine.prepare(request, rollout_steps=steps)
    torch.cuda.synchronize(device)
    prepared_s = time.perf_counter()

    for _ in engine.rollout_stream(batch, steps):
        pass
    torch.cuda.synchronize(device)
    finished_s = time.perf_counter()

    return JobMeasurement(
        job_index=job_index,
        prepare_s=prepared_s - started_s,
        rollout_s=finished_s - prepared_s,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / _BYTES_PER_GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB,
        model_id=id(engine.model),
    )


def run_jobs(engine: AuroraEngine, request: Any, *, jobs: int, steps: int) -> list[JobMeasurement]:
    return [measure_job(engine, request, job_index=index, steps=steps) for index in range(jobs)]


def format_table(measurements: list[JobMeasurement]) -> str:
    header = f"{'job':>3}  {'prepare_s':>9}  {'rollout_s':>9}  {'peak_alloc_GiB':>14}  {'peak_resv_GiB':>13}"
    rows = [
        f"{m.job_index:>3}  {m.prepare_s:>9.2f}  {m.rollout_s:>9.2f}  "
        f"{m.peak_allocated_gib:>14.1f}  {m.peak_reserved_gib:>13.1f}"
        for m in measurements
    ]
    return "\n".join([header, *rows])


def write_report(path: Path, preset: str, measurements: list[JobMeasurement], violations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": preset,
        "device_name": torch.cuda.get_device_name(0),
        "jobs": [asdict(m) for m in measurements],
        "violations": violations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    request, _config = load_preset_ingest_request(args.preset, asset_root)
    engine = AuroraEngine.from_preset(
        args.preset,
        asset_root=asset_root,
        inference_precision=args.precision,
        allow_hub_download=False,
    )
    engine.config.device = args.device
    try:
        measurements = run_jobs(engine, request, jobs=args.jobs, steps=args.steps)
    finally:
        engine.close()

    violations = find_violations(measurements)
    print(format_table(measurements))
    for violation in violations:
        print(f"VIOLATION: {violation}")
    write_report(args.out_dir / f"resident_worker_{args.preset}.json", args.preset, measurements, violations)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
