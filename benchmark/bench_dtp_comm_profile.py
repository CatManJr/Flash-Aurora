#!/usr/bin/env python3
"""Where does a domain-tensor-parallel forward spend its time, and how much is communication?

Loads a real checkpoint, applies the 4D mesh (or runs a single GPU as the baseline),
and profiles forward passes on a synthetic input of the preset's grid. Step time does
not depend on the field values, so no initial condition is needed.

Collectives are blocking in the current implementation, so the compute stream idles
while an NCCL kernel runs; the NCCL kernel time per step is therefore the exposed
communication that overlapping could hide. It includes waiting for the slowest peer.

Examples::

    export AURORA_ASSET_ROOT=/path/to/aurora
    uv run python benchmark/bench_dtp_comm_profile.py --scheme single --report-json single.json
    uv run torchrun --standalone --nproc-per-node 4 benchmark/bench_dtp_comm_profile.py \\
        --scheme dtp --mesh-channel 2 --mesh-spatial 2 --report-json dtp_2x2.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
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
from bench_rollout_drift import build_model, set_benchmark_seed  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

DEFAULT_VARIANT = "aurora-0.25-finetuned"
NATIVE_FP32_TIER = "fp32"
SCHEMES = ("single", "dtp")
WARMUP_STEPS = 1
BYTES_PER_GIB = 1024**3
MICROSECONDS_PER_MILLISECOND = 1000.0
COMPUTE_CATEGORY = "compute_kernels"
TOP_KERNEL_COUNT = 15
NCCL_ANNOTATION_PREFIX = "nccl:"
# Substrings of NCCL kernel names, in matching order, with the label they report under.
NCCL_KERNEL_CATEGORIES = (
    ("AllGather", "all_gather"),
    ("ReduceScatter", "reduce_scatter"),
    ("AllReduce", "all_reduce"),
    ("SendRecv", "send_recv"),
    ("Broadcast", "broadcast"),
    ("nccl", "other_nccl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--precision", default=NATIVE_FP32_TIER)
    parser.add_argument("--mesh-channel", type=int, default=2)
    parser.add_argument("--mesh-spatial", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--profiled-steps", type=int, default=2)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def synthetic_batch(variant: Any, device: torch.device) -> Any:
    from flash_aurora.models.aurora import Batch, Metadata

    height, width = variant.resolution
    generator = torch.Generator().manual_seed(0)

    def field(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device)

    num_history_steps = 2
    batch = Batch(
        surf_vars={name: field(1, num_history_steps, height, width) for name in variant.surf_vars},
        static_vars={name: field(height, width) for name in variant.static_vars},
        atmos_vars={
            name: field(1, num_history_steps, len(variant.levels), height, width) for name in variant.atmos_vars
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, height),
            lon=torch.arange(width) * (360.0 / width),
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=tuple(variant.levels),
        ),
    )
    return batch.to(device)


def build_placed_model(args: argparse.Namespace, asset_root: Path) -> tuple[Any, Any, torch.device, Any]:
    """Return ``(model, variant, device, mesh)`` with the scheme applied and the model on its device."""
    from types import SimpleNamespace

    from flash_aurora.engine.core.presets import VARIANTS

    variant = VARIANTS[args.variant]
    config = SimpleNamespace(variant=variant)
    checkpoint = asset_root / variant.checkpoint_filename
    mesh = None
    if args.scheme == "dtp":
        from flash_aurora.engine.distributed.dtp import apply_domain_tensor_parallel
        from flash_aurora.engine.distributed.process_mesh import (
            MeshShape,
            build_process_mesh,
            init_distributed_from_env,
        )

        device = init_distributed_from_env()
        mesh = build_process_mesh(MeshShape(channel=args.mesh_channel, spatial=args.mesh_spatial))
        model = build_model(config, checkpoint, precision=args.precision, device=torch.device("cpu"))
        apply_domain_tensor_parallel(model, mesh)
    else:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        model = build_model(config, checkpoint, precision=args.precision, device=torch.device("cpu"))
    return model.to(device), variant, device, mesh


def synchronize(device: torch.device, distributed: bool) -> None:
    torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()


def timed_step_milliseconds(model: Any, batch: Any, device: torch.device, distributed: bool) -> float:
    synchronize(device, distributed)
    start = time.perf_counter()
    with torch.inference_mode():
        model(batch)
    synchronize(device, distributed)
    return (time.perf_counter() - start) * MICROSECONDS_PER_MILLISECOND


def nccl_category(kernel_name: str) -> str | None:
    for needle, label in NCCL_KERNEL_CATEGORIES:
        if needle.lower() in kernel_name.lower():
            return label
    return None


def gpu_events(profiler: Any) -> list[Any]:
    """GPU-side events without the ``nccl:`` annotations.

    Each collective appears twice on the GPU timeline: as the NCCL kernel and as an
    annotation range of the same duration. Counting both doubles the communication time.
    """
    return [
        event
        for event in profiler.key_averages()
        if event.device_type == DeviceType.CUDA and not event.key.startswith(NCCL_ANNOTATION_PREFIX)
    ]


def kernel_milliseconds_by_category(profiler: Any, num_steps: int) -> dict[str, float]:
    """Per-step GPU milliseconds of NCCL kernels by category, and of all other kernels."""
    totals: dict[str, float] = defaultdict(float)
    for event in gpu_events(profiler):
        totals[nccl_category(event.key) or COMPUTE_CATEGORY] += event.self_device_time_total
    return {name: total / MICROSECONDS_PER_MILLISECOND / num_steps for name, total in totals.items()}


def top_kernel_milliseconds(profiler: Any, num_steps: int, count: int) -> list[dict[str, Any]]:
    """The ``count`` most expensive GPU kernels, to audit the category matching."""
    kernels = [(event.key, event.self_device_time_total, event.count) for event in gpu_events(profiler)]
    kernels.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "name": name,
            "ms_per_step": total / MICROSECONDS_PER_MILLISECOND / num_steps,
            "calls_per_step": calls / num_steps,
        }
        for name, total, calls in kernels[:count]
    ]


def profiled_breakdown(model: Any, batch: Any, device: torch.device, distributed: bool, steps: int) -> dict[str, Any]:
    synchronize(device, distributed)
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with torch.inference_mode():
            for _ in range(steps):
                model(batch)
        synchronize(device, distributed)
    wall_ms = (time.perf_counter() - start) * MICROSECONDS_PER_MILLISECOND / steps
    return {
        "profiled_wall_ms": wall_ms,
        "kernel_ms_per_step": kernel_milliseconds_by_category(profiler, steps),
        "top_kernels": top_kernel_milliseconds(profiler, steps, TOP_KERNEL_COUNT),
    }


def gather_reports(report: dict[str, Any], distributed: bool) -> list[dict[str, Any]]:
    if not distributed:
        return [report]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, report)
    return gathered


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    step_ms = [statistics.median(report["timed_step_ms"]) for report in reports]
    categories = sorted({name for report in reports for name in report["kernel_ms_per_step"]})
    mean_kernel = {
        name: statistics.mean(report["kernel_ms_per_step"].get(name, 0.0) for report in reports)
        for name in categories
    }
    communication = sum(value for name, value in mean_kernel.items() if name != COMPUTE_CATEGORY)
    median_step = statistics.mean(step_ms)
    return {
        "step_ms_mean_over_ranks": median_step,
        "nccl_ms_per_step_mean_over_ranks": communication,
        "nccl_share_of_step": communication / median_step if median_step > 0 else None,
        "kernel_ms_per_step_mean_over_ranks": mean_kernel,
        "peak_memory_gib_per_rank": [report["peak_memory_gib"] for report in reports],
    }


def main() -> None:
    args = parse_args()
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    set_benchmark_seed()
    model, variant, device, mesh = build_placed_model(args, asset_root)
    distributed = mesh is not None
    batch = synthetic_batch(variant, device)

    for _ in range(WARMUP_STEPS):
        timed_step_milliseconds(model, batch, device, distributed)
    torch.cuda.reset_peak_memory_stats(device)
    timed = [timed_step_milliseconds(model, batch, device, distributed) for _ in range(args.timed_steps)]
    breakdown = profiled_breakdown(model, batch, device, distributed, args.profiled_steps)

    rank_report = {
        "timed_step_ms": timed,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
        **breakdown,
    }
    reports = gather_reports(rank_report, distributed)
    if not distributed or dist.get_rank() == 0:
        result = {
            "scheme": args.scheme,
            "variant": args.variant,
            "precision": args.precision,
            "mesh": None if mesh is None else {"channel": mesh.shape.channel, "spatial": mesh.shape.spatial},
            "gpu": torch.cuda.get_device_name(device),
            "summary": summarize(reports),
            "ranks": reports,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
