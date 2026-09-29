"""Trace -> coordinator -> workers -> timeline -> metrics, end to end over ZMQ."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

from _job_metrics import deadline_hit_ratio, makespan_s, summarize_run
from _job_timeline import JobStatus
from _trace_generator import default_operational_spec, generate_cycle_trace, generate_homogeneous_burst
from _trace_replayer import TraceReplayer
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.coordinator import (
    ForecastCoordinator,
    ForecastCoordinatorConfig,
    WorkerEndpoint,
)
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig

_PREPARE_S = 0.05
_REPLAY_TIMEOUT_S = 60.0
_THREAD_JOIN_TIMEOUT_S = 10.0
_PREDICTION_TIME = datetime(2024, 6, 1, 6)


def _mock_engine(prepare_s: float) -> MagicMock:
    engine = MagicMock()

    def prepare(*_args, **_kwargs):
        time.sleep(prepare_s)
        return MagicMock()

    def rollout_stream(_batch, steps, **_kwargs):
        for _ in range(steps):
            prediction = MagicMock()
            prediction.metadata.time = (_PREDICTION_TIME,)
            yield prediction

    engine.prepare.side_effect = prepare
    engine.rollout_stream.side_effect = rollout_stream
    return engine


def _worker(context: zmq.Context, tmp_path: Path, index: int, preset: str) -> ForecastWorker:
    worker_id = f"gpu-{index}"
    return ForecastWorker(
        ForecastWorkerConfig(
            preset=preset,
            asset_root=tmp_path,
            command_addr=f"ipc://{tmp_path / f'{worker_id}-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / f'{worker_id}-events.ipc'}",
            worker_id=worker_id,
            device=f"cuda:{index}",
            poll_timeout_ms=20,
        ),
        engine=_mock_engine(_PREPARE_S),
        downloader=MagicMock(),
        context=context,
    )


def _endpoint(worker: ForecastWorker) -> WorkerEndpoint:
    return WorkerEndpoint(
        worker_id=worker.worker_id,
        preset=worker.preset,
        command_addr=worker.command_addr,
        event_addr=worker.event_addr,
        device=worker.device,
    )


def _serve_in_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@contextmanager
def _node(tmp_path: Path, presets: Sequence[str]) -> Iterator[tuple[ForecastClient, dict[str, str]]]:
    """A coordinator over one worker per entry of ``presets``; yields a client and worker->preset."""
    context = zmq.Context()
    workers = [_worker(context, tmp_path, index, preset) for index, preset in enumerate(presets)]
    threads = [_serve_in_thread(worker.serve_forever) for worker in workers]
    coordinator = ForecastCoordinator(
        ForecastCoordinatorConfig(
            command_addr=f"ipc://{tmp_path / 'front-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'front-events.ipc'}",
            workers=tuple(_endpoint(worker) for worker in workers),
            poll_timeout_ms=20,
            worker_health_timeout_ms=500,
        ),
        context=context,
    )
    threads.append(_serve_in_thread(coordinator.serve_forever))
    client = ForecastClient(
        ForecastClientConfig(
            command_addr=coordinator.command_addr,
            event_addr=coordinator.event_addr,
            recv_timeout_ms=int(_REPLAY_TIMEOUT_S * 1000),
        ),
        context=context,
    )
    try:
        yield client, {worker.worker_id: worker.preset for worker in workers}
    finally:
        client.shutdown_worker()
        for thread in threads:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        client.close()
        context.term()


def test_homogeneous_burst_is_shared_by_every_worker_of_the_preset(tmp_path: Path) -> None:
    trace = generate_homogeneous_burst("era5_pretrained", jobs=8, steps=2)

    with _node(tmp_path, ["era5_pretrained"] * 2) as (client, _worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    records = timeline.records()
    assert all(record.status is JobStatus.COMPLETED for record in records)
    assert {record.worker_id for record in records} == {"gpu-0", "gpu-1"}


def test_burst_makespan_is_bounded_below_by_the_ideal_schedule(tmp_path: Path) -> None:
    trace = generate_homogeneous_burst("era5_pretrained", jobs=8, steps=2)

    with _node(tmp_path, ["era5_pretrained"] * 2) as (client, _worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    summary = summarize_run(timeline.records(), {"era5_pretrained": 2})
    assert summary["makespan_over_lower_bound"] >= 1.0


def test_heterogeneous_cycle_trace_routes_every_job_to_a_worker_of_its_preset(tmp_path: Path) -> None:
    presets = ["era5_pretrained", "hres_t0_finetuned", "hres_0.1", "cams"]
    trace = generate_cycle_trace(default_operational_spec(cycle_interval_s=0.5, ensemble_members=2))

    with _node(tmp_path, presets) as (client, worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    records = timeline.records()
    assert len(records) == len(trace)
    assert all(record.status is JobStatus.COMPLETED for record in records)
    assert all(worker_presets[record.worker_id] == record.preset for record in records)


def test_cycle_trace_jobs_are_submitted_no_earlier_than_their_arrival(tmp_path: Path) -> None:
    trace = generate_cycle_trace(default_operational_spec(cycle_interval_s=0.5, ensemble_members=2))
    arrival_by_job = {job.job_id: job.arrival_s for job in trace}
    presets = ["era5_pretrained", "hres_t0_finetuned", "hres_0.1", "cams"]

    with _node(tmp_path, presets) as (client, _worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    assert all(record.submitted_s >= arrival_by_job[record.request_id] for record in timeline.records())


def test_cycle_trace_summary_reports_deadlines_and_per_worker_load(tmp_path: Path) -> None:
    presets = ["era5_pretrained", "hres_t0_finetuned", "hres_0.1", "cams"]
    trace = generate_cycle_trace(default_operational_spec(cycle_interval_s=0.5, ensemble_members=2))

    with _node(tmp_path, presets) as (client, _worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    records = timeline.records()
    summary = summarize_run(records, {preset: 1 for preset in presets})
    assert summary["deadline_hit_ratio"] == deadline_hit_ratio(records)
    assert set(summary["per_worker"]) == {"gpu-0", "gpu-1", "gpu-2", "gpu-3"}
    assert makespan_s(records) > 0.0


def test_job_for_a_preset_without_a_worker_fails_at_the_coordinator(tmp_path: Path) -> None:
    trace = generate_homogeneous_burst("cams", jobs=1, steps=1)

    with _node(tmp_path, ["era5_pretrained"]) as (client, _worker_presets):
        timeline = TraceReplayer(client).replay(trace, timeout_s=_REPLAY_TIMEOUT_S)

    [record] = timeline.records()
    assert record.status is JobStatus.FAILED
    assert "no worker registered" in (record.error or "")


def test_replay_raises_when_the_trace_outlasts_the_timeout(tmp_path: Path) -> None:
    trace = generate_homogeneous_burst("era5_pretrained", jobs=4, steps=1)

    with _node(tmp_path, ["era5_pretrained"]) as (client, _worker_presets):
        with pytest.raises(TimeoutError, match="unfinished"):
            TraceReplayer(client).replay(trace, timeout_s=0.001)
