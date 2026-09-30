"""Coordinator detection of workers that stop answering."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import zmq

from flash_aurora.scheduler.client import ForecastClient
from flash_aurora.scheduler.coordinator import ForecastCoordinatorConfig
from flash_aurora.scheduler.protocol import ForecastEvent
from tests.scheduler.stub_cluster import (
    IO_TIMEOUT_MS,
    StubWorker,
    build_client,
    build_coordinator,
    forecast_request,
)

_PROBE_INTERVAL_MS = 20
_SILENCE_LIMIT_MS = 400
_KEEP_ALIVE_PERIOD_S = 0.05
_DISPATCH_ATTEMPT_MS = 100
_THREAD_JOIN_TIMEOUT_S = 5.0


@pytest.fixture
def context():
    context = zmq.Context()
    yield context
    context.term()


class _KeepAlive:
    """Answers the coordinator's health probes on a stub's behalf until the block exits."""

    def __init__(self, stub: StubWorker) -> None:
        self._stub = stub
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._answer_until_stopped, daemon=True)

    def __enter__(self) -> _KeepAlive:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)

    def _answer_until_stopped(self) -> None:
        while not self._stop.wait(_KEEP_ALIVE_PERIOD_S):
            self._stub.answer_health()


@contextmanager
def _cluster(context: zmq.Context, tmp_path: Path, stubs: list[StubWorker]) -> Iterator[ForecastClient]:
    coordinator = build_coordinator(
        context,
        tmp_path,
        stubs,
        worker_probe_interval_ms=_PROBE_INTERVAL_MS,
        worker_silence_limit_ms=_SILENCE_LIMIT_MS,
    )
    thread = threading.Thread(target=coordinator.serve_forever, daemon=True)
    thread.start()
    client = build_client(coordinator, context)
    try:
        yield client
    finally:
        client.shutdown_worker()
        thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        client.close()
        for stub in stubs:
            stub.close()


def _next_failure(client: ForecastClient) -> ForecastEvent:
    while True:
        event = client.recv_event()
        if event.kind == "failed":
            return event


def _failed_request_ids(client: ForecastClient, *, count: int) -> dict[str, str]:
    failures: dict[str, str] = {}
    while len(failures) < count:
        event = _next_failure(client)
        failures[event.request_id or ""] = event.error or ""
    return failures


def _submit_until_dispatched(client: ForecastClient, stub: StubWorker) -> str:
    """Submit jobs until the stub receives one, riding out the coordinator's revival lag."""
    for attempt in range(IO_TIMEOUT_MS // _DISPATCH_ATTEMPT_MS):
        request_id = f"retry-{attempt}"
        client.submit(forecast_request(request_id))
        if stub.next_forecast_request_id(timeout_ms=_DISPATCH_ATTEMPT_MS) == request_id:
            return request_id
    raise AssertionError("the worker never received a job after it spoke again")


def test_silent_worker_has_its_running_job_failed(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    with _cluster(context, tmp_path, [stub]) as client:
        stub.announce_ready()
        client.submit(forecast_request("req-1"))
        assert stub.next_forecast_request_id(timeout_ms=IO_TIMEOUT_MS) == "req-1"

        failed = _next_failure(client)

    assert failed.request_id == "req-1"
    assert failed.worker_id == "worker-0"
    assert "stopped responding" in (failed.error or "")


def test_new_job_for_a_preset_with_only_silent_workers_fails_at_once(
    context: zmq.Context, tmp_path: Path
) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    with _cluster(context, tmp_path, [stub]) as client:
        stub.announce_ready()
        client.submit(forecast_request("req-1"))
        _next_failure(client)

        client.submit(forecast_request("req-2"))
        second = _next_failure(client)

    assert second.request_id == "req-2"
    assert "unresponsive" in (second.error or "")


def test_queued_jobs_fail_when_their_only_worker_goes_silent(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    with _cluster(context, tmp_path, [stub]) as client:
        stub.announce_ready()
        client.submit(forecast_request("running"))
        client.submit(forecast_request("queued"))

        failures = _failed_request_ids(client, count=2)

    assert "stopped responding" in failures["running"]
    assert "unresponsive" in failures["queued"]


def test_surviving_worker_takes_the_next_job(context: zmq.Context, tmp_path: Path) -> None:
    silent = StubWorker(context, tmp_path, "worker-0")
    alive = StubWorker(context, tmp_path, "worker-1")
    with _cluster(context, tmp_path, [silent, alive]) as client:
        silent.announce_ready()
        alive.announce_ready()
        with _KeepAlive(alive):
            client.submit(forecast_request("req-a"))
            assert silent.next_forecast_request_id(timeout_ms=IO_TIMEOUT_MS) == "req-a"
            _next_failure(client)

            client.submit(forecast_request("req-b"))
            received = alive.next_forecast_request_id(timeout_ms=IO_TIMEOUT_MS)

    assert received == "req-b"


def test_worker_that_speaks_again_rejoins_dispatch(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    with _cluster(context, tmp_path, [stub]) as client:
        stub.announce_ready()
        client.submit(forecast_request("req-1"))
        assert stub.next_forecast_request_id(timeout_ms=IO_TIMEOUT_MS) == "req-1"
        _next_failure(client)

        stub.answer_health()
        dispatched = _submit_until_dispatched(client, stub)

    assert dispatched.startswith("retry-")


def test_worker_never_heard_from_is_not_declared_dead(context: zmq.Context, tmp_path: Path) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")
    with _cluster(context, tmp_path, [stub]) as client:
        client.submit(forecast_request("req-1"))
        assert stub.next_forecast_request_id(timeout_ms=IO_TIMEOUT_MS) == "req-1"

        verdict = client.try_recv_event(timeout_ms=3 * _SILENCE_LIMIT_MS)

    assert verdict is None


@pytest.mark.parametrize("probe_interval_ms,silence_limit_ms", [(0, 100), (100, 100), (200, 100)])
def test_config_rejects_a_silence_limit_that_does_not_exceed_the_probe_interval(
    probe_interval_ms: int, silence_limit_ms: int
) -> None:
    with pytest.raises(ValueError):
        ForecastCoordinatorConfig(
            command_addr="ipc:///unused-commands",
            event_addr="ipc:///unused-events",
            workers=(),
            worker_probe_interval_ms=probe_interval_ms,
            worker_silence_limit_ms=silence_limit_ms,
        )
