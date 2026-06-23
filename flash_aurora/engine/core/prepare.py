"""Overlap CPU-heavy ingress with model initialization."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora


@dataclass(frozen=True)
class LoadTiming:
    build_model_ms: float
    load_ckpt_ms: float
    model_h2d_ms: float


@dataclass(frozen=True)
class PrepareTiming:
    build_ic_ms: float
    build_model_ms: float
    load_ckpt_ms: float
    model_h2d_ms: float
    prepare_wall_ms: float


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def overlap_ic_and_load(
    build_ic: Callable[[], Batch],
    load: Callable[[], tuple[Aurora, LoadTiming]],
) -> tuple[Batch, Aurora, PrepareTiming]:
    """Build IC on a worker thread while the model loads on the caller thread."""
    ic_holder: dict[str, Batch | float | None] = {"batch": None, "ms": None}

    def _build_ic_worker() -> None:
        batch, ms = _timed(build_ic)
        ic_holder["batch"] = batch
        ic_holder["ms"] = ms

    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="aurora-ic") as pool:
        ic_future = pool.submit(_build_ic_worker)
        model, load_timing = load()
        ic_future.result()
    prepare_wall_ms = (time.perf_counter() - wall_t0) * 1000.0

    batch = ic_holder["batch"]
    build_ic_ms = ic_holder["ms"]
    if batch is None or build_ic_ms is None:
        raise RuntimeError("IC build did not produce a batch")

    return batch, model, PrepareTiming(
        build_ic_ms=float(build_ic_ms),
        build_model_ms=load_timing.build_model_ms,
        load_ckpt_ms=load_timing.load_ckpt_ms,
        model_h2d_ms=load_timing.model_h2d_ms,
        prepare_wall_ms=prepare_wall_ms,
    )


def serial_ic_then_load(
    build_ic: Callable[[], Batch],
    load: Callable[[], tuple[Aurora, LoadTiming]],
) -> tuple[Batch, Aurora, PrepareTiming]:
    """Build IC, then load the model (legacy serial order)."""
    wall_t0 = time.perf_counter()
    batch, build_ic_ms = _timed(build_ic)
    model, load_timing = load()
    prepare_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    return batch, model, PrepareTiming(
        build_ic_ms=build_ic_ms,
        build_model_ms=load_timing.build_model_ms,
        load_ckpt_ms=load_timing.load_ckpt_ms,
        model_h2d_ms=load_timing.model_h2d_ms,
        prepare_wall_ms=prepare_wall_ms,
    )
