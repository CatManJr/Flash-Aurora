"""Worker event attribution, failure handling, and socket ownership."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

from flash_aurora.engine.runtime.vram_preflight import InsufficientVramError
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.protocol import ForecastCommand, ForecastEvent, ForecastRequest
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind

_RECV_TIMEOUT_MS = 5000
_PRESET = "era5_pretrained"
_TERMINAL_KINDS = ("completed", "failed")


def _request(request_id: str) -> ForecastRequest:
    return ForecastRequest(
        request_id=request_id,
        preset=_PRESET,
        steps=1,
        valid_time="2024-06-01T06:00:00",
    )


def _forecast_command(request_id: str) -> ForecastCommand:
    return ForecastCommand(kind="forecast", request=_request(request_id))


def _mock_engine(tmp_path: Path) -> MagicMock:
    engine = MagicMock()
    export_path = tmp_path / "prediction-000.nc"
    export_path.write_text("nc")
    engine.prepare.return_value = MagicMock()
    engine.rollout_and_export.side_effect = lambda *_a, **_k: iter([export_path])
    return engine


def _worker(tmp_path: Path, engine: MagicMock, context: zmq.Context, **overrides) -> ForecastWorker:
    return ForecastWorker(
        ForecastWorkerConfig(
            preset=_PRESET,
            asset_root=tmp_path,
            command_addr=f"ipc://{tmp_path / 'commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'events.ipc'}",
            poll_timeout_ms=100,
            **overrides,
        ),
        engine=engine,
        downloader=MagicMock(),
        context=context,
    )


def _client(worker: ForecastWorker, context: zmq.Context) -> ForecastClient:
    return ForecastClient(
        ForecastClientConfig(
            command_addr=worker.command_addr,
            event_addr=worker.event_addr,
            recv_timeout_ms=_RECV_TIMEOUT_MS,
        ),
        context=context,
    )


def _events_until_terminal(client: ForecastClient, *, terminal_count: int = 1) -> list[ForecastEvent]:
    events: list[ForecastEvent] = []
    seen_terminal = 0
    while seen_terminal < terminal_count:
        event = client.recv_event()
        events.append(event)
        if event.kind in _TERMINAL_KINDS:
            seen_terminal += 1
    return events


@pytest.fixture
def context():
    context = zmq.Context()
    yield context
    context.term()


def test_every_job_event_carries_worker_id_and_device(tmp_path: Path, context: zmq.Context) -> None:
    worker = _worker(tmp_path, _mock_engine(tmp_path), context, worker_id="gpu-2", device="cuda:2")
    client = _client(worker, context)

    worker.handle_command(_forecast_command("req-1"))
    events = _events_until_terminal(client)

    assert [event.kind for event in events] == ["accepted", "preparing", "running", "step", "completed"]
    assert {(event.worker_id, event.worker_device) for event in events} == {("gpu-2", "cuda:2")}
    client.close()
    worker.close()


def test_failed_event_carries_worker_id(tmp_path: Path, context: zmq.Context) -> None:
    engine = _mock_engine(tmp_path)
    engine.prepare.side_effect = RuntimeError("ic build failed")
    worker = _worker(tmp_path, engine, context, worker_id="gpu-3", device="cuda:3")
    client = _client(worker, context)

    worker.handle_command(_forecast_command("req-1"))
    failed = _events_until_terminal(client)[-1]

    assert failed.kind == "failed"
    assert failed.worker_id == "gpu-3"
    assert "ic build failed" in (failed.error or "")
    client.close()
    worker.close()


def test_model_ready_resets_after_job_failure(tmp_path: Path, context: zmq.Context) -> None:
    engine = _mock_engine(tmp_path)
    engine.prepare.side_effect = RuntimeError("boom")
    worker = _worker(tmp_path, engine, context)
    worker.ensure_loaded()
    assert worker.model_ready

    worker._execute_forecast_job(_request("req-1"))

    assert not worker.model_ready
    engine.release_gpu.assert_called_once_with(move_model_to_cpu=True)
    worker.close()


def test_fatal_vram_error_fails_jobs_still_queued(tmp_path: Path, context: zmq.Context) -> None:
    engine = _mock_engine(tmp_path)
    engine.prepare.side_effect = InsufficientVramError("does not fit")
    worker = _worker(tmp_path, engine, context, worker_id="gpu-0")
    client = _client(worker, context)
    worker._job_queue.put(_request("req-running"))
    worker._job_queue.put(_request("req-queued"))

    worker._compute_loop()
    events = _events_until_terminal(client, terminal_count=2)

    failed_ids = {event.request_id for event in events if event.kind == "failed"}
    assert failed_ids == {"req-running", "req-queued"}
    client.close()
    worker.close()


def test_owned_context_is_not_the_process_singleton(tmp_path: Path) -> None:
    shared = zmq.Context.instance()
    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset=_PRESET,
            asset_root=tmp_path,
            command_addr=f"ipc://{tmp_path / 'commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'events.ipc'}",
        ),
        engine=MagicMock(),
        downloader=MagicMock(),
    )

    worker.close()

    assert not shared.closed


def test_wait_for_bind_returns_once_ipc_socket_file_exists(tmp_path: Path) -> None:
    socket_path = tmp_path / "bound.ipc"
    socket_path.touch()

    wait_for_bind(f"ipc://{socket_path}", timeout_s=0.5)


def test_wait_for_bind_times_out_when_ipc_socket_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="was not bound"):
        wait_for_bind(f"ipc://{tmp_path / 'missing.ipc'}", timeout_s=0.05)


def test_wait_for_bind_does_not_wait_for_tcp_endpoints() -> None:
    wait_for_bind("tcp://127.0.0.1:65000", timeout_s=0.0)
