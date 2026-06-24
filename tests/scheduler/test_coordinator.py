from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import zmq

from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.coordinator import (
    ForecastCoordinator,
    ForecastCoordinatorConfig,
    WorkerEndpoint,
)
from flash_aurora.scheduler.protocol import ForecastRequest
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind


def _start_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _blocking_engine(
    export_path: Path,
    started: threading.Event,
    release: threading.Event,
) -> MagicMock:
    engine = MagicMock()
    batch = MagicMock()
    engine.prepare.return_value = batch

    def rollout_and_export(*_args, **_kwargs):
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release mocked rollout")
        export_path.write_text("nc")
        yield export_path

    engine.rollout_and_export.side_effect = rollout_and_export
    return engine


def _worker(
    *,
    tmp_path: Path,
    context: zmq.Context,
    worker_id: str,
    command_addr: str,
    event_addr: str,
    device: str,
    started: threading.Event,
    release: threading.Event,
) -> ForecastWorker:
    downloader = MagicMock()
    downloader.ingest_request.return_value = MagicMock()
    return ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            worker_id=worker_id,
            device=device,
            capacity=1,
            poll_timeout_ms=50,
        ),
        engine=_blocking_engine(tmp_path / f"{worker_id}.nc", started, release),
        downloader=downloader,
        context=context,
    )


def test_coordinator_dispatches_two_jobs_to_two_workers(tmp_path: Path) -> None:
    context = zmq.Context.instance()
    front_command_addr = f"ipc://{tmp_path / 'front-commands.ipc'}"
    front_event_addr = f"ipc://{tmp_path / 'front-events.ipc'}"
    worker_0_command_addr = f"ipc://{tmp_path / 'worker-0-commands.ipc'}"
    worker_0_event_addr = f"ipc://{tmp_path / 'worker-0-events.ipc'}"
    worker_1_command_addr = f"ipc://{tmp_path / 'worker-1-commands.ipc'}"
    worker_1_event_addr = f"ipc://{tmp_path / 'worker-1-events.ipc'}"

    started_0 = threading.Event()
    started_1 = threading.Event()
    release = threading.Event()

    worker_0 = _worker(
        tmp_path=tmp_path,
        context=context,
        worker_id="worker-0",
        command_addr=worker_0_command_addr,
        event_addr=worker_0_event_addr,
        device="cuda:0",
        started=started_0,
        release=release,
    )
    worker_1 = _worker(
        tmp_path=tmp_path,
        context=context,
        worker_id="worker-1",
        command_addr=worker_1_command_addr,
        event_addr=worker_1_event_addr,
        device="cuda:1",
        started=started_1,
        release=release,
    )
    wait_for_bind(worker_0_command_addr)
    wait_for_bind(worker_1_command_addr)
    worker_threads = [
        _start_thread(worker_0.serve_forever),
        _start_thread(worker_1.serve_forever),
    ]

    coordinator = ForecastCoordinator(
        ForecastCoordinatorConfig(
            command_addr=front_command_addr,
            event_addr=front_event_addr,
            workers=(
                WorkerEndpoint(
                    worker_id="worker-0",
                    preset="era5_pretrained",
                    command_addr=worker_0_command_addr,
                    event_addr=worker_0_event_addr,
                    device="cuda:0",
                ),
                WorkerEndpoint(
                    worker_id="worker-1",
                    preset="era5_pretrained",
                    command_addr=worker_1_command_addr,
                    event_addr=worker_1_event_addr,
                    device="cuda:1",
                ),
            ),
            poll_timeout_ms=50,
        ),
        context=context,
    )
    wait_for_bind(front_command_addr)
    coordinator_thread = _start_thread(coordinator.serve_forever)

    client = ForecastClient(
        ForecastClientConfig(
            command_addr=front_command_addr,
            event_addr=front_event_addr,
            recv_timeout_ms=5000,
        ),
        context=context,
    )

    requests = [
        ForecastRequest(
            request_id="req-0",
            preset="era5_pretrained",
            steps=1,
            valid_time="2024-06-01T06:00:00",
        ),
        ForecastRequest(
            request_id="req-1",
            preset="era5_pretrained",
            steps=1,
            valid_time="2024-06-01T06:00:00",
        ),
    ]
    for request in requests:
        client.submit(request)

    assert started_0.wait(timeout=2.0)
    assert started_1.wait(timeout=2.0)
    release.set()

    completed: set[str] = set()
    while completed != {"req-0", "req-1"}:
        event = client.recv_event()
        if event.kind == "completed" and event.request_id is not None:
            completed.add(event.request_id)

    client.shutdown_worker()
    coordinator_thread.join(timeout=5.0)
    for thread in worker_threads:
        thread.join(timeout=5.0)
    client.close()
