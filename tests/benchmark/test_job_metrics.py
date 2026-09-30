"""Job-level metrics: latency split, makespan against the lower bound, balance, deadlines."""

from __future__ import annotations

import json

import pytest

from _job_metrics import (
    burst_makespan_lower_bound_s,
    busy_seconds_per_worker,
    deadline_hit_ratio,
    jobs_per_worker,
    makespan_s,
    percentile,
    summarize_distribution,
    summarize_run,
)
from _job_timeline import JobRecord, JobStatus


def _completed(
    request_id: str,
    *,
    preset: str = "era5_pretrained",
    worker_id: str = "gpu-0",
    submitted_s: float = 0.0,
    started_s: float = 0.0,
    service_s: float = 2.0,
    deadline_s: float | None = None,
) -> JobRecord:
    """A job whose service time splits evenly into prepare and rollout."""
    running_s = started_s + service_s / 2
    return JobRecord(
        request_id=request_id,
        preset=preset,
        submitted_s=submitted_s,
        deadline_s=deadline_s,
        accepted_s=submitted_s,
        preparing_s=started_s,
        running_s=running_s,
        finished_s=started_s + service_s,
        status=JobStatus.COMPLETED,
        worker_id=worker_id,
        worker_device=None,
        error=None,
    )


def _failed(request_id: str, *, deadline_s: float | None = None) -> JobRecord:
    return JobRecord(
        request_id=request_id,
        preset="era5_pretrained",
        submitted_s=0.0,
        deadline_s=deadline_s,
        accepted_s=None,
        preparing_s=None,
        running_s=None,
        finished_s=1.0,
        status=JobStatus.FAILED,
        worker_id="gpu-0",
        worker_device=None,
        error="boom",
    )


def test_percentile_interpolates_between_closest_ranks() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_of_a_hundred_values_puts_p99_just_below_the_maximum() -> None:
    assert percentile([float(v) for v in range(1, 101)], 99) == pytest.approx(99.01)


def test_percentile_bounds_are_the_minimum_and_maximum() -> None:
    values = [5.0, 1.0, 9.0]

    assert (percentile(values, 0), percentile(values, 100)) == (1.0, 9.0)


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.0], 99) == 7.0


def test_percentile_rejects_an_empty_sample_and_a_bad_quantile() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="within"):
        percentile([1.0], 101)


def test_distribution_summary_reports_count_mean_median_tail_and_max() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0, 10.0])

    assert (summary.count, summary.mean_s, summary.p50_s, summary.max_s) == (4, 4.0, 2.5, 10.0)


def test_makespan_runs_from_first_submission_to_last_completion() -> None:
    records = [_completed("a", submitted_s=1.0, started_s=1.0), _completed("b", submitted_s=2.0, started_s=8.0)]

    assert makespan_s(records) == 9.0


def test_lower_bound_divides_a_presets_work_across_its_workers() -> None:
    records = [_completed(f"a{i}", service_s=2.0) for i in range(4)]

    assert burst_makespan_lower_bound_s(records, {"era5_pretrained": 2}) == 4.0


def test_lower_bound_is_set_by_the_busiest_preset() -> None:
    records = [
        *[_completed(f"a{i}", preset="era5_pretrained", service_s=2.0) for i in range(4)],
        _completed("b", preset="cams", service_s=3.0),
    ]

    assert burst_makespan_lower_bound_s(records, {"era5_pretrained": 2, "cams": 1}) == 4.0


def test_lower_bound_never_drops_below_the_longest_single_job() -> None:
    records = [_completed("long", service_s=10.0)]

    assert burst_makespan_lower_bound_s(records, {"era5_pretrained": 4}) == 10.0


def test_lower_bound_needs_a_declared_worker_for_every_preset() -> None:
    with pytest.raises(ValueError, match="no worker declared"):
        burst_makespan_lower_bound_s([_completed("a")], {"cams": 1})


def test_deadline_hit_ratio_counts_failed_jobs_as_misses() -> None:
    records = [
        _completed("on-time", service_s=2.0, deadline_s=5.0),
        _completed("late", service_s=8.0, deadline_s=5.0),
        _failed("failed", deadline_s=5.0),
        _completed("unconstrained"),
    ]

    assert deadline_hit_ratio(records) == pytest.approx(1 / 3)


def test_deadline_hit_ratio_needs_at_least_one_deadline() -> None:
    with pytest.raises(ValueError, match="no job carries a deadline"):
        deadline_hit_ratio([_completed("a")])


def test_jobs_and_busy_time_are_tallied_per_worker() -> None:
    records = [
        _completed("a", worker_id="gpu-0", service_s=2.0),
        _completed("b", worker_id="gpu-0", service_s=3.0),
        _completed("c", worker_id="gpu-1", service_s=4.0),
        _failed("d"),
    ]

    assert jobs_per_worker(records) == {"gpu-0": 2, "gpu-1": 1}
    assert busy_seconds_per_worker(records) == {"gpu-0": 5.0, "gpu-1": 4.0}


def test_run_summary_is_json_serializable_and_counts_every_status() -> None:
    records = [
        _completed("a", worker_id="gpu-0", deadline_s=10.0),
        _completed("b", worker_id="gpu-1", deadline_s=10.0),
        _failed("c"),
    ]

    summary = summarize_run(records, {"era5_pretrained": 2})

    assert json.loads(json.dumps(summary)) == summary
    assert summary["jobs"] == {"completed": 2, "failed": 1, "incomplete": 0, "submitted": 3}
    assert summary["deadline_hit_ratio"] == 1.0


def test_run_summary_reports_makespan_against_the_lower_bound() -> None:
    records = [_completed(f"j{i}", worker_id="gpu-0", started_s=2.0 * i, service_s=2.0) for i in range(4)]

    summary = summarize_run(records, {"era5_pretrained": 2})

    assert summary["makespan_s"] == 8.0
    assert summary["burst_makespan_lower_bound_s"] == 4.0
    assert summary["makespan_over_lower_bound"] == 2.0


def test_run_summary_utilization_is_busy_time_over_makespan() -> None:
    records = [_completed(f"j{i}", worker_id="gpu-0", started_s=2.0 * i, service_s=2.0) for i in range(2)]

    summary = summarize_run(records, {"era5_pretrained": 1})

    assert summary["per_worker"] == {"gpu-0": {"jobs": 2, "busy_s": 4.0, "utilization": 1.0}}


def test_run_summary_leaves_deadline_ratio_null_without_deadlines() -> None:
    summary = summarize_run([_completed("a")], {"era5_pretrained": 1})

    assert summary["deadline_hit_ratio"] is None


def test_run_summary_refuses_a_run_with_no_completed_job() -> None:
    with pytest.raises(ValueError, match="no completed jobs"):
        summarize_run([_failed("a")], {"era5_pretrained": 1})
