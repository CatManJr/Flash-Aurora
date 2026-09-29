"""Stub workers and helpers for driving a ForecastCoordinator over real ZMQ sockets."""

from __future__ import annotations

import time
from pathlib import Path

import zmq

from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.coordinator import (
    ForecastCoordinator,
    ForecastCoordinatorConfig,
    WorkerEndpoint,
)
from flash_aurora.scheduler.protocol import (
    ForecastEvent,
    ForecastRequest,
    decode_command,
    encode_event,
)

PRESET = "era5_pretrained"
IO_TIMEOUT_MS = 5000
QUIET_MS = 200


class StubWorker:
    """Stands in for a ForecastWorker: receives commands, pushes whatever events a test scripts."""

    def __init__(self, context: zmq.Context, tmp_path: Path, worker_id: str, preset: str = PRESET) -> None:
        self.worker_id = worker_id
        self.preset = preset
        self.command_addr = f"ipc://{tmp_path / f'{worker_id}-commands.ipc'}"
        self.event_addr = f"ipc://{tmp_path / f'{worker_id}-events.ipc'}"
        self._command_pull = context.socket(zmq.PULL)
        self._command_pull.bind(self.command_addr)
        self._event_push = context.socket(zmq.PUSH)
        self._event_push.setsockopt(zmq.SNDTIMEO, IO_TIMEOUT_MS)
        self._event_push.bind(self.event_addr)

    @property
    def endpoint(self) -> WorkerEndpoint:
        return WorkerEndpoint(
            worker_id=self.worker_id,
            preset=self.preset,
            command_addr=self.command_addr,
            event_addr=self.event_addr,
        )

    def next_forecast_request_id(self, *, timeout_ms: int) -> str | None:
        """Return the next forecast request id, ignoring the coordinator's health probes."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while (remaining_s := deadline - time.monotonic()) > 0:
            if not self._command_pull.poll(timeout=int(remaining_s * 1000) + 1):
                return None
            command = decode_command(self._command_pull.recv())
            if command.kind == "forecast" and command.request is not None:
                return command.request.request_id
        return None

    def push(self, event: ForecastEvent) -> None:
        self._event_push.send(encode_event(event))

    def announce_ready(self) -> None:
        self.push(ForecastEvent(kind="ready", worker_id=self.worker_id, message="ready"))

    def answer_health(self) -> None:
        self.push(ForecastEvent(kind="health", worker_id=self.worker_id, message="ready"))

    def close(self) -> None:
        self._command_pull.close(linger=0)
        self._event_push.close(linger=0)


def forecast_request(request_id: str, preset: str = PRESET) -> ForecastRequest:
    return ForecastRequest(
        request_id=request_id,
        preset=preset,
        steps=1,
        valid_time="2024-06-01T06:00:00",
    )


def build_coordinator(
    context: zmq.Context,
    tmp_path: Path,
    workers: list[StubWorker],
    **config_overrides: int,
) -> ForecastCoordinator:
    return ForecastCoordinator(
        ForecastCoordinatorConfig(
            command_addr=f"ipc://{tmp_path / 'front-commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'front-events.ipc'}",
            workers=tuple(worker.endpoint for worker in workers),
            **{"poll_timeout_ms": 20, "worker_health_timeout_ms": 50, **config_overrides},
        ),
        context=context,
    )


def build_client(coordinator: ForecastCoordinator, context: zmq.Context) -> ForecastClient:
    return ForecastClient(
        ForecastClientConfig(
            command_addr=coordinator.command_addr,
            event_addr=coordinator.event_addr,
            recv_timeout_ms=IO_TIMEOUT_MS,
        ),
        context=context,
    )
