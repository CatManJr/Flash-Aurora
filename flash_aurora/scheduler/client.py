"""ZMQ client for the single-worker forecast scheduler (P1)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import zmq

from flash_aurora.scheduler.protocol import (
    WORKER_STATUS_LISTENING,
    WORKER_STATUS_READY,
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
        self._context = zmq.Context() if context is None else context
        self._command_socket = self._context.socket(zmq.PUSH)
        self._event_socket = self._context.socket(zmq.PULL)
        self._command_socket.connect(config.command_addr)
        self._event_socket.connect(config.event_addr)
        self._event_socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout_ms)
        self._closed = False

    def __enter__(self) -> ForecastClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._command_socket.close(linger=0)
        self._event_socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _send_command(self, command: ForecastCommand) -> None:
        self._command_socket.send(encode_command(command))

    def recv_event(self) -> ForecastEvent:
        """Receive the next scheduler event from the event socket."""
        data = self._event_socket.recv()
        return decode_event(data)

    def try_recv_event(self, *, timeout_ms: int) -> ForecastEvent | None:
        """Return the next event, or None if none arrives within ``timeout_ms``."""
        if not self._event_socket.poll(timeout=timeout_ms):
            return None
        return self.recv_event()

    def _recv_event(self) -> ForecastEvent:
        return self.recv_event()

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

    def wait_for_ready(
        self,
        *,
        timeout_s: float = 600.0,
        require_model_loaded: bool = True,
    ) -> ForecastEvent:
        """Block until the peer reports ready (event or health).

        Late-joining clients miss the startup ``ready`` PUSH, so this also polls
        ``health``. Against one worker, ``ready`` means that worker's model is
        loaded. Against a coordinator, the same word means every worker is loaded.
        ``listening`` means sockets are up and weights are not.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._event_socket.poll(timeout=50):
                event = self._recv_event()
                if self._is_ready_report(event, require_model_loaded=require_model_loaded):
                    return event
                continue
            self._send_command(ForecastCommand(kind="health"))
            try:
                # Temporarily tighten recv timeout for the health round-trip.
                previous = self._event_socket.getsockopt(zmq.RCVTIMEO)
                self._event_socket.setsockopt(zmq.RCVTIMEO, 1_000)
                try:
                    event = self._recv_event()
                finally:
                    self._event_socket.setsockopt(zmq.RCVTIMEO, previous)
            except zmq.Again:
                continue
            if self._is_ready_report(event, require_model_loaded=require_model_loaded):
                return event
        raise TimeoutError("timed out waiting for worker ready")

    @staticmethod
    def _is_ready_report(event: ForecastEvent, *, require_model_loaded: bool) -> bool:
        if event.kind not in ("ready", "health"):
            return False
        if event.message == WORKER_STATUS_READY:
            return True
        return not require_model_loaded and event.message == WORKER_STATUS_LISTENING

    def shutdown_worker(self) -> None:
        self._send_command(ForecastCommand(kind="shutdown"))

    def events(self, request_id: str) -> Iterator[ForecastEvent]:
        while True:
            event = self._recv_event()
            if event.kind in ("ready", "health"):
                continue
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
