"""Coordinator dispatch and event routing across workers."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import zmq

from flash_aurora.scheduler.client import ForecastClient
from flash_aurora.scheduler.coordinator import ForecastCoordinator, ForecastCoordinatorConfig
from flash_aurora.scheduler.protocol import ForecastEvent
from tests.scheduler.stub_cluster import (
    IO_TIMEOUT_MS,
    PRESET,
    QUIET_MS,
    StubWorker,
    build_client,
    build_coordinator,
    forecast_request,
)

_POLL_SLICE_MS = 50


@pytest.fixture
def context():
    context = zmq.Context()
    yield context
    context.term()


def _wait_for_completed(client: ForecastClient, request_id: str) -> None:
    while True:
        event = client.recv_event()
        if event.kind == "completed" and event.request_id == request_id:
            return


def test_sequential_jobs_alternate_between_idle_workers(context: zmq.Context, tmp_path: Path) -> None:
    stubs = [StubWorker(context, tmp_path, "worker-0"), StubWorker(context, tmp_path, "worker-1")]
    coordinator = build_coordinator(context, tmp_path, stubs)
    thread = threading.Thread(target=coordinator.serve_forever, daemon=True)
    thread.start()
    client = build_client(coordinator, context)

    receivers: list[str] = []
    for index in range(3):
        request_id = f"req-{index}"
        client.submit(forecast_request(request_id))
        receiver = _worker_that_received(stubs, request_id)
        receivers.append(receiver.worker_id)
        receiver.push(ForecastEvent(kind="completed", request_id=request_id))
        _wait_for_completed(client, request_id)

    assert receivers == ["worker-0", "worker-1", "worker-0"]
    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()
    for stub in stubs:
        stub.close()


def _worker_that_received(stubs: list[StubWorker], request_id: str) -> StubWorker:
    for _ in range(IO_TIMEOUT_MS // _POLL_SLICE_MS):
        for stub in stubs:
            if stub.next_forecast_request_id(timeout_ms=_POLL_SLICE_MS) == request_id:
                return stub
    raise AssertionError(f"no worker received {request_id}")


def test_worker_health_reply_is_applied_but_not_forwarded_to_clients(
    context: zmq.Context, tmp_path: Path
) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    coordinator = build_coordinator(context, tmp_path, [stub])
    client = build_client(coordinator, context)
    health = ForecastEvent(
        kind="health",
        worker_preset=PRESET,
        worker_id="worker-0",
        worker_device="cuda:0",
        worker_capacity=3,
        message="ready",
    )

    coordinator._handle_worker_event(coordinator._workers["worker-0"], health)

    assert client.try_recv_event(timeout_ms=QUIET_MS) is None
    assert coordinator._workers["worker-0"].endpoint.capacity == 3
    client.close()
    coordinator.close()
    stub.close()


def test_refresh_reads_health_that_follows_a_job_event(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    coordinator = build_coordinator(context, tmp_path, [stub])
    client = build_client(coordinator, context)
    stub.push(ForecastEvent(kind="running", request_id="req-early"))
    stub.push(ForecastEvent(kind="health", worker_id="worker-0", worker_capacity=2, message="ready"))

    coordinator.refresh_worker_health()

    forwarded = client.recv_event()
    assert (forwarded.kind, forwarded.request_id) == ("running", "req-early")
    assert coordinator._workers["worker-0"].endpoint.capacity == 2
    assert client.try_recv_event(timeout_ms=QUIET_MS) is None
    client.close()
    coordinator.close()
    stub.close()


def test_owned_context_is_not_the_process_singleton(tmp_path: Path) -> None:
    shared = zmq.Context.instance()
    stub_context = zmq.Context()
    stub = StubWorker(stub_context, tmp_path, "worker-0")
    coordinator = ForecastCoordinator(
        ForecastCoordinatorConfig(
            command_addr=f"ipc://{tmp_path / 'front-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'front-events.ipc'}",
            workers=(stub.endpoint,),
        )
    )

    coordinator.close()

    assert not shared.closed
    stub.close()
    stub_context.term()
