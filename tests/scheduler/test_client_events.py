"""ForecastClient socket ownership and non-blocking event receive."""

from __future__ import annotations

from pathlib import Path

import zmq

from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.protocol import ForecastEvent, encode_event

_QUIET_MS = 100
_IO_TIMEOUT_MS = 5000


def _config(tmp_path: Path) -> ForecastClientConfig:
    return ForecastClientConfig(
        command_addr=f"ipc://{tmp_path / 'commands.ipc'}",
        event_addr=f"ipc://{tmp_path / 'events.ipc'}",
        recv_timeout_ms=_IO_TIMEOUT_MS,
    )


def test_try_recv_event_returns_none_when_no_event_arrives(tmp_path: Path) -> None:
    with ForecastClient(_config(tmp_path)) as client:
        assert client.try_recv_event(timeout_ms=_QUIET_MS) is None


def test_try_recv_event_returns_the_pushed_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    context = zmq.Context()
    event_push = context.socket(zmq.PUSH)
    event_push.setsockopt(zmq.SNDTIMEO, _IO_TIMEOUT_MS)
    event_push.bind(config.event_addr)
    client = ForecastClient(config, context=context)

    event_push.send(encode_event(ForecastEvent(kind="running", request_id="req-1")))
    event = client.try_recv_event(timeout_ms=_IO_TIMEOUT_MS)

    assert event is not None and (event.kind, event.request_id) == ("running", "req-1")
    client.close()
    event_push.close(linger=0)
    context.term()


def test_owned_context_is_not_the_process_singleton(tmp_path: Path) -> None:
    shared = zmq.Context.instance()
    client = ForecastClient(_config(tmp_path))

    client.close()

    assert not shared.closed
