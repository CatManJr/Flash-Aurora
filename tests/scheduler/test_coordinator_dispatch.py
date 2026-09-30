"""Coordinator dispatch and event routing across workers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import zmq

from flash_aurora.scheduler.client import ForecastClient
from flash_aurora.scheduler.coordinator import (
    ForecastCoordinator,
    ForecastCoordinatorConfig,
    WorkerEndpoint,
)
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


def test_worker_health_reply_updates_the_worker_and_announces_the_cluster(
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

    try:
        coordinator._handle_worker_event(coordinator._workers["worker-0"], health)

        announced = _flush_until_event(coordinator, client)
        assert announced is not None
        assert announced.kind == "ready"
        assert announced.message == "ready"
        assert announced.worker_id == "coordinator"
        assert client.try_recv_event(timeout_ms=QUIET_MS) is None
        assert coordinator._workers["worker-0"].endpoint.capacity == 3
    finally:
        client.close()
        coordinator.close()
        stub.close()


def test_refresh_reads_health_that_follows_a_job_event(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    coordinator = build_coordinator(context, tmp_path, [stub])
    client = build_client(coordinator, context)
    stub.push(ForecastEvent(kind="running", request_id="req-early"))
    stub.push(ForecastEvent(kind="health", worker_id="worker-0", worker_capacity=2, message="ready"))

    try:
        coordinator.refresh_worker_health()

        forwarded = _flush_until_event(coordinator, client)
        announced = _flush_until_event(coordinator, client)
        assert forwarded is not None and (forwarded.kind, forwarded.request_id) == ("running", "req-early")
        assert announced is not None
        assert (announced.kind, announced.message, announced.worker_id) == ("ready", "ready", "coordinator")
        assert coordinator._workers["worker-0"].endpoint.capacity == 2
        assert client.try_recv_event(timeout_ms=QUIET_MS) is None
    finally:
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


def test_client_events_are_queued_when_no_client_is_connected(
    context: zmq.Context, tmp_path: Path
) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    coordinator = build_coordinator(context, tmp_path, [stub])
    started_s = time.monotonic()

    for _ in range(8):
        coordinator._emit(ForecastEvent(kind="ready", message="ready", worker_id="coordinator"))

    assert time.monotonic() - started_s < 1.0
    client = build_client(coordinator, context)
    event = _flush_until_event(coordinator, client)
    assert event is not None and event.kind == "ready"
    client.close()
    coordinator.close()
    stub.close()


def _flush_until_event(coordinator: ForecastCoordinator, client: ForecastClient):
    deadline_s = time.monotonic() + 1.0
    while time.monotonic() < deadline_s:
        coordinator._flush_outbound()
        event = client.try_recv_event(timeout_ms=50)
        if event is not None:
            return event
    return None


def test_close_returns_when_no_worker_is_bound(tmp_path: Path) -> None:
    coordinator = ForecastCoordinator(
        ForecastCoordinatorConfig(
            command_addr=f"ipc://{tmp_path / 'front-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'front-events.ipc'}",
            workers=(
                WorkerEndpoint(
                    worker_id="missing",
                    preset=PRESET,
                    command_addr=f"ipc://{tmp_path / 'missing-commands.ipc'}",
                    event_addr=f"ipc://{tmp_path / 'missing-events.ipc'}",
                ),
            ),
        )
    )
    started_s = time.monotonic()

    coordinator.close()

    assert time.monotonic() - started_s < 1.0


def test_close_delivers_shutdown(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    coordinator = build_coordinator(context, tmp_path, [stub])

    try:
        coordinator.close()
        assert stub.next_command_kind(timeout_ms=IO_TIMEOUT_MS) == "shutdown"
    finally:
        stub.close()


def test_pool_dispatch_prefers_the_worker_that_already_holds_the_preset(
    context: zmq.Context, tmp_path: Path
) -> None:
    holding = StubWorker(context, tmp_path, "gpu-2")
    other = StubWorker(context, tmp_path, "gpu-3")
    rotating = ("era5_pretrained", "cams")
    coordinator = build_coordinator(context, tmp_path, [holding, other])
    coordinator._workers["gpu-2"].endpoint = WorkerEndpoint(
        worker_id="gpu-2",
        preset="era5_pretrained",
        command_addr=holding.command_addr,
        event_addr=holding.event_addr,
        accepted_presets=rotating,
    )
    coordinator._workers["gpu-3"].endpoint = WorkerEndpoint(
        worker_id="gpu-3",
        preset="cams",
        command_addr=other.command_addr,
        event_addr=other.event_addr,
        accepted_presets=rotating,
    )
    coordinator._workers["gpu-3"].dispatched_count = 5

    chosen = coordinator._choose_worker(forecast_request("req-cams", "cams"))

    assert chosen is not None
    assert chosen.endpoint.worker_id == "gpu-3"
    coordinator.close()
    holding.close()
    other.close()


def test_coordinator_rejects_two_workers_with_the_same_id(tmp_path: Path) -> None:
    def endpoint(name: str) -> WorkerEndpoint:
        return WorkerEndpoint(
            worker_id="gpu-0",
            preset=PRESET,
            command_addr=f"ipc://{tmp_path / f'{name}-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / f'{name}-events.ipc'}",
        )

    config = ForecastCoordinatorConfig(
        command_addr=f"ipc://{tmp_path / 'front-commands.ipc'}",
        event_addr=f"ipc://{tmp_path / 'front-events.ipc'}",
        workers=(endpoint("first"), endpoint("second")),
    )

    with pytest.raises(ValueError, match="unique"):
        ForecastCoordinator(config)
