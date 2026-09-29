"""Distributed multi workers coordinator for job-level GPU scheduling."""

from __future__ import annotations

import argparse
import signal
import time
from collections import deque
from collections.abc import Callable
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
    # Health probe cadence, and how long a worker that has spoken before may stay
    # silent before its running jobs are failed. Workers never heard from are exempt
    # so a checkpoint preload is not mistaken for a crash.
    worker_probe_interval_ms: int = 2_000
    worker_silence_limit_ms: int = 30_000

    def __post_init__(self) -> None:
        if self.worker_probe_interval_ms < 1:
            raise ValueError("worker_probe_interval_ms must be >= 1")
        if self.worker_silence_limit_ms <= self.worker_probe_interval_ms:
            raise ValueError("worker_silence_limit_ms must exceed worker_probe_interval_ms")


@dataclass
class _WorkerState:
    endpoint: WorkerEndpoint
    command_socket: zmq.Socket
    event_socket: zmq.Socket
    running: set[str]
    dispatched_count: int = 0
    last_heard_s: float | None = None
    unresponsive: bool = False

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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not config.workers:
            raise ValueError("coordinator requires at least one worker endpoint")
        self._config = config
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._clock = clock
        self._next_probe_s = clock()
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
        """Best-effort worker metadata refresh; stops waiting on a worker that stays silent."""
        for worker in self._workers.values():
            self._send_worker(worker, ForecastCommand(kind="health"))
        for worker in self._workers.values():
            self._await_health_reply(worker)

    def _await_health_reply(self, worker: _WorkerState) -> None:
        # Job events can arrive ahead of the reply; route them, keep waiting for health.
        while True:
            try:
                event = decode_event(worker.event_socket.recv())
            except zmq.Again:
                return
            self._handle_worker_event(worker, event)
            if event.kind == "health":
                return

    def _apply_worker_health(self, worker: _WorkerState, event: ForecastEvent) -> None:
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

    def _live_matching_workers(self, request: ForecastRequest) -> list[_WorkerState]:
        return [worker for worker in self._matching_workers(request) if not worker.unresponsive]

    def _choose_worker(self, request: ForecastRequest) -> _WorkerState | None:
        candidates = self._live_matching_workers(request)
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
        # Most free slots first, then the worker that has run the fewest jobs, so idle
        # workers share load instead of the highest worker_id absorbing every burst.
        return min(
            ready,
            key=lambda worker: (
                -worker.available_slots,
                worker.dispatched_count,
                worker.endpoint.worker_id,
            ),
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
        if not self._live_matching_workers(request):
            self._emit_failed_unservable(request)
            return
        self._queue.append(command)
        self._dispatch_ready()

    def _emit_failed_unservable(self, request: ForecastRequest) -> None:
        self._emit(
            ForecastEvent(
                kind="failed",
                request_id=request.request_id,
                error=f"every worker for preset {request.preset!r} is unresponsive",
            )
        )

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
            worker.dispatched_count += 1
            self._request_worker[request.request_id] = worker.endpoint.worker_id
            if self._config.sticky_sessions and request.sticky_key is not None:
                self._sticky_workers[request.sticky_key] = worker.endpoint.worker_id
            self._send_worker(worker, command)
        self._queue = deferred

    def _handle_worker_event(self, worker: _WorkerState, event: ForecastEvent) -> None:
        was_unresponsive = worker.unresponsive
        worker.last_heard_s = self._clock()
        worker.unresponsive = False
        # Health replies answer the coordinator, not a client, so they never leave here.
        if event.kind == "health":
            self._apply_worker_health(worker, event)
        else:
            self._forward_worker_event(worker, event)
        if was_unresponsive:
            self._dispatch_ready()

    def _supervise_workers(self) -> None:
        """Probe workers on a fixed cadence and give up on any that stay silent."""
        now_s = self._clock()
        if now_s >= self._next_probe_s:
            self._next_probe_s = now_s + self._config.worker_probe_interval_ms / 1000.0
            for worker in self._workers.values():
                self._probe_worker(worker)
        for worker in self._workers.values():
            if self._has_gone_silent(worker, now_s):
                self._declare_unresponsive(worker)

    def _probe_worker(self, worker: _WorkerState) -> None:
        # A dead peer's pipe fills to the high-water mark, after which a blocking send
        # would freeze the whole coordinator; the silence limit already covers that worker.
        try:
            worker.command_socket.send(
                encode_command(ForecastCommand(kind="health")),
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _has_gone_silent(self, worker: _WorkerState, now_s: float) -> bool:
        if worker.unresponsive or worker.last_heard_s is None:
            return False
        return now_s - worker.last_heard_s > self._config.worker_silence_limit_ms / 1000.0

    def _declare_unresponsive(self, worker: _WorkerState) -> None:
        worker.unresponsive = True
        for request_id in sorted(worker.running):
            self._request_worker.pop(request_id, None)
            self._emit(self._lost_job_event(worker, request_id))
        worker.running.clear()
        self._sticky_workers = {
            key: worker_id
            for key, worker_id in self._sticky_workers.items()
            if worker_id != worker.endpoint.worker_id
        }
        self._fail_queued_jobs_without_live_worker()

    def _lost_job_event(self, worker: _WorkerState, request_id: str) -> ForecastEvent:
        endpoint = worker.endpoint
        return ForecastEvent(
            kind="failed",
            request_id=request_id,
            worker_id=endpoint.worker_id,
            worker_device=endpoint.device,
            error=(
                f"worker {endpoint.worker_id!r} stopped responding while running this job "
                f"(silent for more than {self._config.worker_silence_limit_ms} ms)"
            ),
        )

    def _fail_queued_jobs_without_live_worker(self) -> None:
        still_servable: Deque[ForecastCommand] = deque()
        for command in self._queue:
            request = command.request
            if request is not None and not self._live_matching_workers(request):
                self._emit_failed_unservable(request)
            else:
                still_servable.append(command)
        self._queue = still_servable

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
        self._next_probe_s = self._clock() + self._config.worker_probe_interval_ms / 1000.0
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
                        self._handle_worker_event(worker, event)
                self._supervise_workers()
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
    parser.add_argument("--worker-probe-interval-ms", type=int, default=2_000)
    parser.add_argument(
        "--worker-silence-limit-ms",
        type=int,
        default=30_000,
        help="Fail a worker's running jobs after it stays silent this long",
    )
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
        worker_probe_interval_ms=args.worker_probe_interval_ms,
        worker_silence_limit_ms=args.worker_silence_limit_ms,
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
