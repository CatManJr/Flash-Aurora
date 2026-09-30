"""Cached ingress must not fall through to a download.

These checks belong to the scheduler benchmark, not the production test run.
The default pytest selection is ``not benchmark``. Run them with ``pytest -m benchmark``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _trace_generator import (
    EXPENSIVE_PRESETS,
    SCHEDULER_TRACE_PRESETS,
    ROTATING_PRESETS,
    TraceJob,
    group_b_trace,
)
from _trace_replayer import CachedIngress, forecast_request
from _job_timeline import JobRecord, JobStatus
from bench_scheduler_trace import (
    SCHEDULER_TRACE_WORKERS,
    _job_row,
    _read_memory_report,
    memory_high_water,
    runtime_environment,
)

pytestmark = pytest.mark.benchmark


def _job(preset: str = "era5_pretrained") -> TraceJob:
    return TraceJob(
        job_id="job-1",
        product="ens-0.25",
        preset=preset,
        steps=4,
        arrival_s=1.0,
        valid_time="2024-06-01T00:00:00",
        deadline_s=10.0,
    )


def test_cached_ingress_requests_the_local_analysis_and_disables_download() -> None:
    ingress = CachedIngress(
        valid_time="2023-01-01T06:00:00",
        cache_dir="/data/aurora/era5",
        time_index=1,
    )

    request = forecast_request(
        _job(),
        output_mode="metadata_only",
        export_dir=None,
        ingress=ingress,
    )

    assert request.download is False
    assert request.valid_time == "2023-01-01T06:00:00"
    assert request.cache_dir == "/data/aurora/era5"
    assert request.output_mode == "metadata_only"


def test_without_ingress_the_trace_keeps_its_own_valid_time() -> None:
    request = forecast_request(
        _job(),
        output_mode="metadata_only",
        export_dir=None,
        ingress=None,
    )

    assert request.valid_time == "2024-06-01T00:00:00"
    assert request.download is True


def test_scheduler_trace_workers_cover_every_production_preset() -> None:
    covered = {preset for worker in SCHEDULER_TRACE_WORKERS for preset in worker.presets}

    assert covered == set(SCHEDULER_TRACE_PRESETS)
    assert len({worker.device for worker in SCHEDULER_TRACE_WORKERS}) == 4


def test_expensive_jobs_arrive_before_the_rotating_presets() -> None:
    trace = group_b_trace(100.0, ensemble_members=2, rotation_rounds=2)
    first_rotation_s = min(job.arrival_s for job in trace if job.preset in ROTATING_PRESETS)

    assert {job.preset for job in trace if job.arrival_s == 0.0} == set(EXPENSIVE_PRESETS)
    assert first_rotation_s == 100.0
    assert all(job.arrival_s >= first_rotation_s for job in trace if job.preset in ROTATING_PRESETS)
    assert {job.preset for job in trace if job.product == "rotate"} == set(ROTATING_PRESETS)


def test_runtime_environment_names_the_stack_and_the_gpus() -> None:
    environment = runtime_environment()

    assert environment["torch"]
    assert environment["cuda"]
    assert environment["python"]
    assert environment["cpu_model"]
    assert environment["logical_cpus"] > 0
    assert environment["gpus"]
    assert environment["gpus"][0]["name"]
    assert environment["driver"]


def test_job_row_keeps_the_times_a_bubble_chart_needs() -> None:
    record = JobRecord(
        request_id="rotate-r0-cams",
        preset="cams",
        submitted_s=120.0,
        deadline_s=240.0,
        accepted_s=120.1,
        preparing_s=121.0,
        running_s=140.0,
        finished_s=144.0,
        status=JobStatus.COMPLETED,
        worker_id="gpu-3",
        worker_device="cuda:3",
        error=None,
    )

    row = _job_row(record)

    assert row["submitted_s"] == 120.0
    assert row["wait_s"] == 1.0
    assert row["prepare_s"] == 19.0
    assert row["rollout_s"] == 4.0
    assert row["service_s"] == 23.0
    assert row["latency_s"] == 24.0
    assert row["met_deadline"] is True
    assert row["worker_id"] == "gpu-3"


def test_memory_report_file_is_preferred_over_the_log(tmp_path: Path) -> None:
    report = tmp_path / "gpu-0.memory.json"
    report.write_text('{"peak_allocated_gib": 12.5, "peak_reserved_gib": 40.0}', encoding="utf-8")

    assert _read_memory_report(report) == (12.5, 40.0)


def test_memory_high_water_reads_allocated_and_reserved() -> None:
    log = "listening\nmemory_high_water allocated_gib=10.500000 reserved_gib=30.250000\n"

    assert memory_high_water(log) == (10.5, 30.25)
