"""Distributed multi workers coordinator for job-level GPU scheduling."""

from __future__ import annotations

import argparse
import signal
from collections import deque
from dataclasses import dataclass
from typing import Deque

import zmq

from flash_aurora.scheduler.protocol import (
    ForecastCommand,
    ForecastEvent,
    ForecastRequest,
    decode_command,
    decode_event,
    encode_command,
    encode_event,
)


@dataclass(frozen=True)
class WorkerEndpoint:
    """Static connection details and advertised worker capacity."""

    worker_id: str
    preset: str
    command_addr: str
    event_addr: str
    device: str | None = None
    capacity: int = 1


@dataclass
class ForecastCoordinatorConfig:
    """Configuration for a front-end scheduler over one or more workers."""

    command_addr: str
    event_addr: str
    workers: tuple[WorkerEndpoint, ...]
    poll_timeout_ms: int = 100
    worker_health_timeout_ms: int = 1000
    sticky_sessions: bool = True


@dataclass
class _WorkerState:
    endpoint: WorkerEndpoint
    command_socket: zmq.Socket
    event_socket: zmq.Socket
    running: set[str]

    @property
    def available_slots(self) -> int:
        return max(0, self.endpoint.capacity - len(self.running))


class ForecastCoordinator:
    """Dispatch forecast jobs to idle workers and forward worker events."""

    def __init__(
        self,
        config: ForecastCoordinatorConfig,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        if not config.workers:
            raise ValueError("coordinator requires at least one worker endpoint")
        self._config = config
        self._owns_context = context is None
        self._context = context or zmq.Context.instance()
        self._running = True
        self._queue: Deque[ForecastCommand] = deque()
        self._request_worker: dict[str, str] = {}
        self._sticky_workers: dict[str, str] = {}

        self._command_socket = self._context.socket(zmq.PULL)
        self._event_socket = self._context.socket(zmq.PUSH)
        self._command_socket.bind(config.command_addr)
        self._event_socket.bind(config.event_addr)
        self._closed = False

        self._workers: dict[str, _WorkerState] = {}
        for endpoint in config.workers:
            if endpoint.capacity < 1:
                raise ValueError(f"worker {endpoint.worker_id!r} capacity must be >= 1")
            command_socket = self._context.socket(zmq.PUSH)
            event_socket = self._context.socket(zmq.PULL)
            command_socket.connect(endpoint.command_addr)
            event_socket.connect(endpoint.event_addr)
            event_socket.setsockopt(zmq.RCVTIMEO, config.worker_health_timeout_ms)
            self._workers[endpoint.worker_id] = _WorkerState(
                endpoint=endpoint,
                command_socket=command_socket,
                event_socket=event_socket,
                running=set(),
            )

    @property
    def command_addr(self) -> str:
        return self._config.command_addr

    @property
    def event_addr(self) -> str:
        return self._config.event_addr

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._running = False
        for worker in self._workers.values():
            try:
                self._send_worker(worker, ForecastCommand(kind="shutdown"))
            except Exception:
                pass
        self._command_socket.close(linger=0)
        self._event_socket.close(linger=0)
        for worker in self._workers.values():
            worker.command_socket.close(linger=0)
            worker.event_socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _emit(self, event: ForecastEvent) -> None:
        self._event_socket.send(encode_event(event))

    def _send_worker(self, worker: _WorkerState, command: ForecastCommand) -> None:
        worker.command_socket.send(encode_command(command))

    def refresh_worker_health(self) -> None:
        """Best-effort worker metadata refresh from health events."""
        for worker in self._workers.values():
            self._send_worker(worker, ForecastCommand(kind="health"))

        for worker in self._workers.values():
            try:
                event = decode_event(worker.event_socket.recv())
            except zmq.Again:
                continue
            if event.kind != "health":
                self._forward_worker_event(worker, event)
                continue
            endpoint = worker.endpoint
            worker.endpoint = WorkerEndpoint(
                worker_id=endpoint.worker_id,
                preset=event.worker_preset or endpoint.preset,
                command_addr=endpoint.command_addr,
                event_addr=endpoint.event_addr,
                device=event.worker_device or endpoint.device,
                capacity=event.worker_capacity or endpoint.capacity,
            )

    def _matching_workers(self, request: ForecastRequest) -> list[_WorkerState]:
        return [
            worker
            for worker in self._workers.values()
            if worker.endpoint.preset == request.preset
        ]

    def _choose_worker(self, request: ForecastRequest) -> _WorkerState | None:
        candidates = self._matching_workers(request)
        if not candidates:
            return None

        if self._config.sticky_sessions and request.sticky_key is not None:
            sticky_id = self._sticky_workers.get(request.sticky_key)
            if sticky_id is not None:
                worker = self._workers.get(sticky_id)
                if (
                    worker is not None
                    and worker.endpoint.preset == request.preset
                    and worker.available_slots > 0
                ):
                    return worker
                return None

        ready = [worker for worker in candidates if worker.available_slots > 0]
        if not ready:
            return None
        return max(
            ready,
            key=lambda worker: (worker.available_slots, worker.endpoint.worker_id),
        )

    def _enqueue_or_fail(self, command: ForecastCommand) -> None:
        request = command.request
        if request is None:
            self._emit(
                ForecastEvent(
                    kind="failed",
                    error="forecast command requires a request payload",
                )
            )
            return
        if not self._matching_workers(request):
            self._emit(
                ForecastEvent(
                    kind="failed",
                    request_id=request.request_id,
                    error=f"no worker registered for preset {request.preset!r}",
                )
            )
            return
        self._queue.append(command)
        self._dispatch_ready()

    def _dispatch_ready(self) -> None:
        deferred: Deque[ForecastCommand] = deque()
        while self._queue:
            command = self._queue.popleft()
            request = command.request
            if request is None:
                continue
            worker = self._choose_worker(request)
            if worker is None:
                deferred.append(command)
                continue
            worker.running.add(request.request_id)
            self._request_worker[request.request_id] = worker.endpoint.worker_id
            if self._config.sticky_sessions and request.sticky_key is not None:
                self._sticky_workers[request.sticky_key] = worker.endpoint.worker_id
            self._send_worker(worker, command)
        self._queue = deferred

    def _forward_worker_event(self, worker: _WorkerState, event: ForecastEvent) -> None:
        self._emit(event)
        request_id = event.request_id
        if request_id is None:
            return
        if event.kind in ("completed", "failed"):
            worker.running.discard(request_id)
            self._request_worker.pop(request_id, None)
            self._dispatch_ready()

    def _handle_command(self, command: ForecastCommand) -> bool:
        if command.kind == "shutdown":
            for worker in self._workers.values():
                self._send_worker(worker, command)
            return False
        if command.kind == "health":
            presets = sorted({worker.endpoint.preset for worker in self._workers.values()})
            self._emit(
                ForecastEvent(
                    kind="health",
                    worker_preset=",".join(presets),
                    worker_id="coordinator",
                    worker_capacity=sum(
                        worker.endpoint.capacity for worker in self._workers.values()
                    ),
                    message="ok",
                )
            )
            return True
        if command.kind == "forecast":
            self._enqueue_or_fail(command)
            return True
        self._emit(ForecastEvent(kind="failed", error=f"unsupported command {command.kind!r}"))
        return True

    def serve_forever(self) -> None:
        self.refresh_worker_health()
        poller = zmq.Poller()
        poller.register(self._command_socket, zmq.POLLIN)
        for worker in self._workers.values():
            poller.register(worker.event_socket, zmq.POLLIN)

        try:
            while self._running:
                events = dict(poller.poll(timeout=self._config.poll_timeout_ms))
                if self._command_socket in events:
                    command = decode_command(self._command_socket.recv())
                    if not self._handle_command(command):
                        break
                for worker in self._workers.values():
                    if worker.event_socket in events:
                        event = decode_event(worker.event_socket.recv())
                        self._forward_worker_event(worker, event)
        finally:
            self.close()


def install_signal_handlers(coordinator: ForecastCoordinator) -> None:
    def _handler(_signum: int, _frame: object) -> None:
        coordinator._running = False
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def parse_worker_endpoint(raw: str) -> WorkerEndpoint:
    """Parse worker_id,preset,command_addr,event_addr[,device[,capacity]]."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in (4, 5, 6):
        raise argparse.ArgumentTypeError(
            "--worker must be worker_id,preset,command_addr,event_addr[,device[,capacity]]"
        )
    worker_id, preset, command_addr, event_addr = parts[:4]
    if not worker_id or not preset or not command_addr or not event_addr:
        raise argparse.ArgumentTypeError("worker id, preset, and addresses must be non-empty")
    device = parts[4] if len(parts) >= 5 and parts[4] else None
    capacity = int(parts[5]) if len(parts) == 6 and parts[5] else 1
    if capacity < 1:
        raise argparse.ArgumentTypeError("worker capacity must be >= 1")
    return WorkerEndpoint(
        worker_id=worker_id,
        preset=preset,
        command_addr=command_addr,
        event_addr=event_addr,
        device=device,
        capacity=capacity,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash-Aurora Distributed multi workers coordinator")
    parser.add_argument(
        "--command-addr",
        default="tcp://127.0.0.1:9855",
        help="ZMQ bind address for incoming client commands",
    )
    parser.add_argument(
        "--event-addr",
        default="tcp://127.0.0.1:9856",
        help="ZMQ bind address for outgoing client events",
    )
    parser.add_argument(
        "--worker",
        action="append",
        required=True,
        type=parse_worker_endpoint,
        help="worker_id,preset,command_addr,event_addr[,device[,capacity]]",
    )
    parser.add_argument("--poll-timeout-ms", type=int, default=100)
    parser.add_argument("--worker-health-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--no-sticky-sessions",
        action="store_true",
        help="Disable sticky_key worker affinity",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ForecastCoordinatorConfig(
        command_addr=args.command_addr,
        event_addr=args.event_addr,
        workers=tuple(args.worker),
        poll_timeout_ms=args.poll_timeout_ms,
        worker_health_timeout_ms=args.worker_health_timeout_ms,
        sticky_sessions=not args.no_sticky_sessions,
    )
    coordinator = ForecastCoordinator(config)
    install_signal_handlers(coordinator)
    worker_desc = ", ".join(
        f"{worker.worker_id}:{worker.preset}:{worker.device or 'device?'}"
        for worker in config.workers
    )
    print(
        f"[coordinator] command={config.command_addr} event={config.event_addr} "
        f"workers=[{worker_desc}]",
        flush=True,
    )
    try:
        coordinator.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.close()


if __name__ == "__main__":
    main()
