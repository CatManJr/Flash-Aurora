"""Job-level metrics over JobRecords: latency split, makespan, balance, deadlines.

These are the numbers the paper reports for the scheduler. They differ from
LLM-serving metrics on purpose: a job is minute-scale, non-preemptible,
exclusive on one GPU and not batchable, so the meaningful quantities are the
wait / prepare / rollout split, makespan against a lower bound, and the
fraction of jobs that beat their deadline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from _job_timeline import JobRecord, JobStatus

_P50 = 50.0
_P99 = 99.0


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean_s: float
    p50_s: float
    p99_s: float
    max_s: float


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation between closest ranks.

    With few samples p99 is effectively the maximum; the report carries ``count``
    so a reader can see how much weight the tail figure has.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within [0, 100], got {q}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    return DistributionSummary(
        count=len(values),
        mean_s=sum(values) / len(values),
        p50_s=percentile(values, _P50),
        p99_s=percentile(values, _P99),
        max_s=max(values),
    )


def completed_jobs(records: Sequence[JobRecord]) -> list[JobRecord]:
    return [record for record in records if record.status is JobStatus.COMPLETED]


def makespan_s(records: Sequence[JobRecord]) -> float:
    """First submission to last completion."""
    done = _require_completed(records)
    return max(record.finished_s for record in done) - min(record.submitted_s for record in done)


def burst_makespan_lower_bound_s(
    records: Sequence[JobRecord],
    workers_per_preset: Mapping[str, int],
) -> float:
    """Best makespan any scheduler could reach if every job were submitted at once.

    Jobs of a preset only run on that preset's workers and never split across
    GPUs, so the bound is the larger of the busiest preset's total service time
    divided by its worker count and the single longest job. It ignores arrival
    times, so it is meaningful for burst traces only.
    """
    done = _require_completed(records)
    service_by_preset: dict[str, float] = defaultdict(float)
    for record in done:
        service_by_preset[record.preset] += record.service_s
    per_preset_bound = max(
        total / _worker_count(workers_per_preset, preset)
        for preset, total in service_by_preset.items()
    )
    return max(per_preset_bound, max(record.service_s for record in done))


def deadline_hit_ratio(records: Sequence[JobRecord]) -> float:
    """Share of deadline-bearing jobs that completed in time; failed or unfinished ones miss."""
    with_deadline = [record for record in records if record.has_deadline]
    if not with_deadline:
        raise ValueError("no job carries a deadline")
    return sum(1 for record in with_deadline if record.met_deadline) / len(with_deadline)


def jobs_per_worker(records: Sequence[JobRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in completed_jobs(records):
        counts[_worker_of(record)] += 1
    return dict(counts)


def busy_seconds_per_worker(records: Sequence[JobRecord]) -> dict[str, float]:
    busy: dict[str, float] = defaultdict(float)
    for record in completed_jobs(records):
        busy[_worker_of(record)] += record.service_s
    return dict(busy)


def summarize_run(
    records: Sequence[JobRecord],
    workers_per_preset: Mapping[str, int],
) -> dict[str, object]:
    """JSON-ready summary of one benchmark run."""
    done = _require_completed(records)
    run_makespan_s = makespan_s(records)
    lower_bound_s = burst_makespan_lower_bound_s(records, workers_per_preset)
    return {
        "jobs": _status_counts(records),
        "latency_s": asdict(summarize_distribution([r.latency_s for r in done])),
        "wait_s": asdict(summarize_distribution([r.wait_s for r in done])),
        "prepare_s": asdict(summarize_distribution([r.prepare_s for r in done])),
        "rollout_s": asdict(summarize_distribution([r.rollout_s for r in done])),
        "makespan_s": run_makespan_s,
        "burst_makespan_lower_bound_s": lower_bound_s,
        "makespan_over_lower_bound": run_makespan_s / lower_bound_s,
        "deadline_hit_ratio": _deadline_hit_ratio_or_none(records),
        "per_worker": _per_worker_summary(records, run_makespan_s),
    }


def _require_completed(records: Sequence[JobRecord]) -> list[JobRecord]:
    done = completed_jobs(records)
    if not done:
        raise ValueError("no completed jobs to measure")
    return done


def _worker_count(workers_per_preset: Mapping[str, int], preset: str) -> int:
    count = workers_per_preset.get(preset, 0)
    if count < 1:
        raise ValueError(f"no worker declared for preset {preset!r}")
    return count


def _worker_of(record: JobRecord) -> str:
    if record.worker_id is None:
        raise ValueError(f"completed job {record.request_id!r} carries no worker_id")
    return record.worker_id


def _status_counts(records: Sequence[JobRecord]) -> dict[str, int]:
    counts = {status.value: 0 for status in JobStatus}
    for record in records:
        counts[record.status.value] += 1
    counts["submitted"] = len(records)
    return counts


def _deadline_hit_ratio_or_none(records: Sequence[JobRecord]) -> float | None:
    return deadline_hit_ratio(records) if any(r.has_deadline for r in records) else None


def _per_worker_summary(
    records: Sequence[JobRecord],
    run_makespan_s: float,
) -> dict[str, dict[str, float]]:
    counts = jobs_per_worker(records)
    busy = busy_seconds_per_worker(records)
    return {
        worker_id: {
            "jobs": counts[worker_id],
            "busy_s": busy[worker_id],
            "utilization": busy[worker_id] / run_makespan_s,
        }
        for worker_id in sorted(counts)
    }
