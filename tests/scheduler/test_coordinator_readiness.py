"""Cluster readiness is one answer for every worker, not the first worker to speak."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import zmq

from flash_aurora.scheduler.client import ForecastClient
from flash_aurora.scheduler.protocol import (
    CLUSTER_STATUS_STARTING,
    WORKER_STATUS_LISTENING,
    WORKER_STATUS_READY,
    ForecastEvent,
)
from tests.scheduler.stub_cluster import StubWorker, build_client, build_coordinator

_THREAD_JOIN_TIMEOUT_S = 5.0
_NOT_READY_YET_S = 0.4
_READY_TIMEOUT_S = 2.0


@pytest.fixture
def context():
    context = zmq.Context()
    yield context
    context.term()


@contextmanager
def _cluster(context: zmq.Context, tmp_path: Path, stubs: list[StubWorker]) -> Iterator[ForecastClient]:
    coordinator = build_coordinator(context, tmp_path, stubs)
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


def test_wait_for_ready_blocks_until_every_worker_has_loaded(
    context: zmq.Context, tmp_path: Path
) -> None:
    first = StubWorker(context, tmp_path, "worker-0")
    second = StubWorker(context, tmp_path, "worker-1", preset="cams")

    with _cluster(context, tmp_path, [first, second]) as client:
        first.announce_ready()
        assert client.health().message == CLUSTER_STATUS_STARTING
        with pytest.raises(TimeoutError):
            client.wait_for_ready(timeout_s=_NOT_READY_YET_S)

        second.announce_ready()
        ready = client.wait_for_ready(timeout_s=_READY_TIMEOUT_S)
        assert client.health().message == WORKER_STATUS_READY

    assert ready.kind == "ready"
    assert ready.message == WORKER_STATUS_READY
    assert ready.worker_id == "coordinator"


def test_a_listening_worker_keeps_the_cluster_from_reporting_models_loaded(
    context: zmq.Context, tmp_path: Path
) -> None:
    loaded = StubWorker(context, tmp_path, "worker-0")
    bound = StubWorker(context, tmp_path, "worker-1")

    with _cluster(context, tmp_path, [loaded, bound]) as client:
        loaded.announce_ready()
        bound.push(
            ForecastEvent(
                kind="ready",
                worker_id=bound.worker_id,
                message=WORKER_STATUS_LISTENING,
            )
        )
        listening = client.wait_for_ready(timeout_s=_READY_TIMEOUT_S, require_model_loaded=False)
        with pytest.raises(TimeoutError):
            client.wait_for_ready(timeout_s=_NOT_READY_YET_S)
        assert client.health().message == WORKER_STATUS_LISTENING

    assert listening.message == WORKER_STATUS_LISTENING


def test_late_health_reply_recovers_a_missed_ready_event(
    context: zmq.Context, tmp_path: Path
) -> None:
    stub = StubWorker(context, tmp_path, "worker-0")

    with _cluster(context, tmp_path, [stub]) as client:
        stub.answer_health()
        ready = client.wait_for_ready(timeout_s=_READY_TIMEOUT_S)
        assert client.health().message == WORKER_STATUS_READY

    assert ready.message == WORKER_STATUS_READY
