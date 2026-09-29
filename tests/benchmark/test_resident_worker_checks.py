"""Acceptance criteria for a worker that keeps its model resident."""

from __future__ import annotations

import pytest

from _resident_worker_checks import JobMeasurement, find_violations


def _job(
    index: int,
    *,
    rollout_s: float = 0.8,
    peak_allocated_gib: float = 30.0,
    model_id: int = 1,
) -> JobMeasurement:
    return JobMeasurement(
        job_index=index,
        prepare_s=20.0,
        rollout_s=rollout_s,
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_allocated_gib + 2.0,
        model_id=model_id,
    )


def test_resident_jobs_with_a_steady_footprint_pass() -> None:
    jobs = [_job(0), _job(1, peak_allocated_gib=28.0), _job(2, rollout_s=0.82)]

    assert find_violations(jobs) == []


def test_a_rebuilt_model_is_reported() -> None:
    jobs = [_job(0, model_id=1), _job(1, model_id=2), _job(2, model_id=3)]

    [violation] = find_violations(jobs)

    assert "3 distinct instances" in violation


def test_doubled_peak_memory_is_reported_per_job() -> None:
    jobs = [_job(0), _job(1, peak_allocated_gib=60.0), _job(2, peak_allocated_gib=29.0)]

    [violation] = find_violations(jobs)

    assert violation.startswith("job 1: peak")


def test_a_rollout_that_doubles_after_the_first_job_is_reported() -> None:
    jobs = [_job(0, rollout_s=0.78), _job(1, rollout_s=1.69)]

    [violation] = find_violations(jobs)

    assert violation.startswith("job 1: rollout")


def test_growth_within_tolerance_is_not_reported() -> None:
    jobs = [_job(0, peak_allocated_gib=30.0, rollout_s=1.0), _job(1, peak_allocated_gib=31.0, rollout_s=1.2)]

    assert find_violations(jobs) == []


def test_every_departure_is_listed_together() -> None:
    jobs = [_job(0), _job(1, model_id=2, peak_allocated_gib=60.0, rollout_s=1.7)]

    assert len(find_violations(jobs)) == 3


def test_a_single_job_cannot_show_residency() -> None:
    with pytest.raises(ValueError, match="at least two"):
        find_violations([_job(0)])
