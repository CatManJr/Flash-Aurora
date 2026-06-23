"""ZMQ client for the single-worker forecast scheduler (P1)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import zmq

from flash_aurora.scheduler.protocol import (
    ForecastCommand,
    ForecastEvent,
    ForecastRequest,
    SchedulerError,
    decode_event,
    encode_command,
)


@dataclass
class ForecastClientConfig:
    command_addr: str
    event_addr: str
    recv_timeout_ms: int = 3_600_000


class ForecastClient:
    """Send forecast jobs to a long-lived worker and receive streaming events."""

    def __init__(
        self,
        config: ForecastClientConfig,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self._config = config
        self._owns_context = context is None
        self._context = context or zmq.Context.instance()
        self._command_socket = self._context.socket(zmq.PUSH)
        self._event_socket = self._context.socket(zmq.PULL)
        self._command_socket.connect(config.command_addr)
        self._event_socket.connect(config.event_addr)
        self._event_socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout_ms)

    def close(self) -> None:
        self._command_socket.close(linger=0)
        self._event_socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _send_command(self, command: ForecastCommand) -> None:
        self._command_socket.send(encode_command(command))

    def _recv_event(self) -> ForecastEvent:
        data = self._event_socket.recv()
        return decode_event(data)

    def submit(self, request: ForecastRequest) -> None:
        self._send_command(ForecastCommand(kind="forecast", request=request))

    def health(self) -> ForecastEvent:
        self._send_command(ForecastCommand(kind="health"))
        deadline = time.time() + 30.0
        while time.time() < deadline:
            event = self._recv_event()
            if event.kind == "health":
                return event
        raise TimeoutError("timed out waiting for health response")

    def shutdown_worker(self) -> None:
        self._send_command(ForecastCommand(kind="shutdown"))

    def events(self, request_id: str) -> Iterator[ForecastEvent]:
        while True:
            event = self._recv_event()
            if event.request_id not in (None, request_id):
                continue
            yield event
            if event.kind == "failed" and event.request_id == request_id:
                raise SchedulerError(event.error or "forecast failed")
            if event.kind == "completed" and event.request_id == request_id:
                break

    def forecast(self, request: ForecastRequest) -> list[ForecastEvent]:
        self.submit(request)
        return list(self.events(request.request_id))
