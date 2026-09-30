"""Acceptance criteria for a worker that keeps its model resident between jobs.

A long-lived worker pays checkpoint load and CUDA graph capture once. From the
second job on, that shows up as three things: the same model instance, no more
peak memory than the first job needed, and a rollout no slower than the first.
Rebuilding the model per job breaks all three, the last two by roughly 2x.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PEAK_GROWTH_TOLERANCE = 1.05
ROLLOUT_SLOWDOWN_TOLERANCE = 1.25


@dataclass(frozen=True)
class JobMeasurement:
    job_index: int
    prepare_s: float
    rollout_s: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    model_id: int


def find_violations(measurements: Sequence[JobMeasurement]) -> list[str]:
    """Every way jobs after the first departed from the first job's footprint."""
    if len(measurements) < 2:
        raise ValueError("a residency check needs at least two consecutive jobs")
    first, *later = measurements
    return [
        *_rebuilt_model_violations(first, later),
        *_peak_growth_violations(first, later),
        *_rollout_slowdown_violations(first, later),
    ]


def _rebuilt_model_violations(first: JobMeasurement, later: Sequence[JobMeasurement]) -> list[str]:
    distinct = {first.model_id, *(job.model_id for job in later)}
    if len(distinct) == 1:
        return []
    return [f"model was rebuilt: {len(distinct)} distinct instances across {len(later) + 1} jobs"]


def _peak_growth_violations(first: JobMeasurement, later: Sequence[JobMeasurement]) -> list[str]:
    ceiling_gib = first.peak_allocated_gib * PEAK_GROWTH_TOLERANCE
    return [
        f"job {job.job_index}: peak {job.peak_allocated_gib:.1f} GiB exceeds the first job's "
        f"{first.peak_allocated_gib:.1f} GiB by more than {PEAK_GROWTH_TOLERANCE - 1:.0%}"
        for job in later
        if job.peak_allocated_gib > ceiling_gib
    ]


def _rollout_slowdown_violations(first: JobMeasurement, later: Sequence[JobMeasurement]) -> list[str]:
    ceiling_s = first.rollout_s * ROLLOUT_SLOWDOWN_TOLERANCE
    return [
        f"job {job.job_index}: rollout {job.rollout_s:.2f} s is slower than the first job's "
        f"{first.rollout_s:.2f} s by more than {ROLLOUT_SLOWDOWN_TOLERANCE - 1:.0%}"
        for job in later
        if job.rollout_s > ceiling_s
    ]
