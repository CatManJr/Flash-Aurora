"""Long-lived single-GPU forecast worker (P1)."""

from __future__ import annotations

import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import zmq

from flash_aurora.engine.core.engine import AuroraEngine
from flash_aurora.engine.ingress.download import DataDownloader
from flash_aurora.engine.runtime.vram_preflight import InsufficientVramError
from flash_aurora.scheduler.addresses import resolve_bound_endpoint
from flash_aurora.scheduler.protocol import (
    WORKER_STATUS_BUSY,
    WORKER_STATUS_LISTENING,
    WORKER_STATUS_READY,
    ForecastCommand,
    ForecastEvent,
    ForecastEventKind,
    ForecastRequest,
    decode_command,
    encode_array,
    encode_event,
)

# Sentinel that asks the compute thread to exit after draining queued jobs.
_COMPUTE_STOP = object()
# Shutdown join budget: one rollout step is O(seconds); allow a short multi-step job to finish.
_COMPUTE_JOIN_TIMEOUT_S = 120.0
# A blocking send with no peer, or a full pipe, holds the emit lock and stops
# health replies. The coordinator would then fail a worker that is still computing.
_MAX_QUEUED_EVENTS = 1024
# Inline callers have no poll loop to retry a send that raced the peer handshake.
_INLINE_FLUSH_WAIT_S = 0.05
_INLINE_FLUSH_PAUSE_S = 0.001


@dataclass
class ForecastWorkerConfig:
    preset: str
    asset_root: Path
    command_addr: str
    event_addr: str
    worker_id: str | None = None
    device: str | None = None
    capacity: int = 1
    inference_precision: str | None = None
    export_dir: Path | None = None
    ic_cache: bool | None = None
    forward_warmup_iters: int | None = None
    overlap_ic_load: bool | None = None
    async_export: bool | None = None
    distributed_devices: tuple[str, ...] | None = None
    distributed_max_vram_gib: float | None = None
    distributed_force: bool = False
    poll_timeout_ms: int = 1000
    # When True, call ``engine.load()`` before accepting jobs and emit ``ready``.
    preload: bool = False
    preload_rollout_steps: int = 1
    # Presets this process may load. Empty means only ``preset``. Switching releases
    # GPU memory and builds a new engine; the previous weights are not kept resident.
    presets: tuple[str, ...] = ()


class ForecastWorker:
    """Single-preset worker that processes forecast jobs sequentially."""

    def __init__(
        self,
        config: ForecastWorkerConfig,
        *,
        engine: AuroraEngine | None = None,
        downloader: DataDownloader | None = None,
        context: zmq.Context | None = None,
    ) -> None:
        self._config = config
        if config.capacity < 1:
            raise ValueError("worker capacity must be >= 1")
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._accepted_presets = config.presets or (config.preset,)
        if config.preset not in self._accepted_presets:
            raise ValueError(
                f"initial preset {config.preset!r} is not in {self._accepted_presets}"
            )
        self._loaded_preset = config.preset
        self._engine = engine or self._build_engine()
        self._downloader = downloader or DataDownloader.from_preset(
            config.preset,
            asset_root=config.asset_root,
        )
        self._running = True
        self._model_ready = False
        self._busy = False
        self._state_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._pending_events: deque[ForecastEvent] = deque()
        self._compute_thread: threading.Thread | None = None
        self._detached_compute = False
        self._fatal_error: BaseException | None = None
        self._command_socket = self._context.socket(zmq.PULL)
        self._event_socket = self._context.socket(zmq.PUSH)
        self._command_socket.bind(config.command_addr)
        self._event_socket.bind(config.event_addr)
        self._bound_command_addr = resolve_bound_endpoint(self._command_socket)
        self._bound_event_addr = resolve_bound_endpoint(self._event_socket)
        self._closed = False

    @property
    def preset(self) -> str:
        return self._config.preset

    @property
    def engine(self) -> AuroraEngine:
        return self._engine

    @property
    def command_addr(self) -> str:
        return self._bound_command_addr

    @property
    def event_addr(self) -> str:
        return self._bound_event_addr

    @property
    def model_ready(self) -> bool:
        with self._state_lock:
            return self._model_ready

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy

    @property
    def worker_id(self) -> str:
        if self._config.worker_id is not None:
            return self._config.worker_id
        if self._config.distributed_devices:
            devices = ",".join(self._config.distributed_devices)
            return f"{self._config.preset}@pipeline[{devices}]"
        return f"{self._config.preset}@{self._config.device or 'cuda:0'}"

    @property
    def device(self) -> str:
        """Primary device: the one inputs are placed on."""
        if self._config.device is not None:
            return self._config.device
        if self._config.distributed_devices:
            return self._config.distributed_devices[0]
        engine_config = getattr(self._engine, "config", None)
        engine_device = getattr(engine_config, "device", None)
        return engine_device if isinstance(engine_device, str) else "cuda:0"

    @property
    def capacity(self) -> int:
        return self._config.capacity

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_engine(self, preset: str | None = None) -> AuroraEngine:
        kwargs: dict[str, Any] = {
            "asset_root": self._config.asset_root,
            "allow_hub_download": False,
        }
        if self._config.inference_precision is not None:
            kwargs["inference_precision"] = self._config.inference_precision
        if self._config.export_dir is not None:
            kwargs["export_dir"] = self._config.export_dir
        if self._config.ic_cache is not None:
            kwargs["ic_cache"] = self._config.ic_cache
        if self._config.forward_warmup_iters is not None:
            kwargs["forward_warmup_iters"] = self._config.forward_warmup_iters
        if self._config.overlap_ic_load is not None:
            kwargs["overlap_ic_load"] = self._config.overlap_ic_load
        if self._config.async_export is not None:
            kwargs["async_export"] = self._config.async_export
        if self._config.distributed_devices:
            from flash_aurora.engine.distributed import DistributedConfig

            kwargs["distributed"] = DistributedConfig(
                devices=self._config.distributed_devices,
                max_vram_gib_per_device=self._config.distributed_max_vram_gib,
                force=self._config.distributed_force,
            )
        engine = AuroraEngine.from_preset(preset or self._loaded_preset, **kwargs)
        if self._config.device is not None and not self._config.distributed_devices:
            engine.config.device = self._config.device
        return engine

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._running = False
        self._stop_compute_thread()
        try:
            self._flush_until_inline_deadline()
        except Exception:
            pass
        self._log_memory_high_water()
        try:
            self._engine.close()
        except Exception:
            pass
        self._command_socket.close(linger=0)
        self._event_socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _log_memory_high_water(self) -> None:
        """Print allocator peaks. Reserved, not allocated, is what must fit on the GPU."""
        if not torch.cuda.is_available():
            return
        device_name = self.device if self.device.startswith("cuda") else "cuda"
        try:
            device = torch.device(device_name)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        except Exception:
            return
        if reserved <= 0:
            return
        gib = 1024**3
        print(
            "memory_high_water "
            f"allocated_gib={allocated / gib:.6f} reserved_gib={reserved / gib:.6f}",
            flush=True,
        )

    def _set_model_ready(self, ready: bool) -> None:
        with self._state_lock:
            self._model_ready = ready

    def _set_busy(self, busy: bool) -> None:
        with self._state_lock:
            self._busy = busy

    def _health_message(self) -> str:
        with self._state_lock:
            if self._busy:
                return WORKER_STATUS_BUSY
            if self._model_ready:
                return WORKER_STATUS_READY
            return WORKER_STATUS_LISTENING

    def _emit(self, event: ForecastEvent) -> None:
        """Queue one event and send whatever the pipe will take without blocking."""
        with self._emit_lock:
            if len(self._pending_events) >= _MAX_QUEUED_EVENTS:
                return
            self._pending_events.append(event)
            self._flush_events_locked()
            retry_inline = not self._detached_compute and bool(self._pending_events)
        if retry_inline:
            self._flush_until_inline_deadline()

    def _flush_until_inline_deadline(self) -> None:
        """Give a just-connected peer one short window. The serve loop retries on its own."""
        deadline_s = time.monotonic() + _INLINE_FLUSH_WAIT_S
        while time.monotonic() < deadline_s:
            with self._emit_lock:
                self._flush_events_locked()
                if not self._pending_events:
                    return
            time.sleep(_INLINE_FLUSH_PAUSE_S)

    def _flush_events_locked(self) -> None:
        while self._pending_events:
            try:
                self._event_socket.send(encode_event(self._pending_events[0]), flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            self._pending_events.popleft()

    def _flush_events(self) -> None:
        with self._emit_lock:
            self._flush_events_locked()

    def _job_event(self, kind: ForecastEventKind, **fields: Any) -> ForecastEvent:
        """Build an event attributed to this worker so clients can trace a job to a GPU."""
        return ForecastEvent(
            kind=kind,
            worker_id=self.worker_id,
            worker_device=self.device,
            worker_preset=self._loaded_preset,
            **fields,
        )

    def _emit_ready(self, *, message: str) -> None:
        # NOBLOCK: startup ready must not stall when no PULL peer is connected yet.
        # Late clients recover via health (message ready|listening).
        try:
            with self._emit_lock:
                self._event_socket.send(
                    encode_event(
                        ForecastEvent(
                            kind="ready",
                            worker_preset=self._loaded_preset,
                            worker_id=self.worker_id,
                            worker_device=self.device,
                            worker_capacity=self.capacity,
                            message=message,
                        )
                    ),
                    flags=zmq.NOBLOCK,
                )
        except zmq.Again:
            pass

    def _prepare_engine_for(self, preset: str) -> None:
        """Load ``preset``, releasing the previous engine first when the model changes."""
        if preset == self._loaded_preset:
            self.ensure_loaded()
            return
        self._release_loaded_engine()
        self._engine = self._build_engine(preset)
        self._downloader = DataDownloader.from_preset(
            preset,
            asset_root=self._config.asset_root,
        )
        self._loaded_preset = preset
        self._config.preset = preset
        self._set_model_ready(False)
        self.ensure_loaded()

    def _release_loaded_engine(self) -> None:
        """Drop GPU weights and the caching allocator's reserved blocks."""
        self._set_model_ready(False)
        try:
            self._engine.release_gpu(move_model_to_cpu=True)
        except Exception:
            pass
        try:
            self._engine.close()
        except Exception:
            pass
        self._loaded_preset = None

    def ensure_loaded(self, *, rollout_steps: int | None = None) -> None:
        """Load weights onto the device and mark the worker warm."""
        if self.model_ready:
            return
        steps = rollout_steps if rollout_steps is not None else self._config.preload_rollout_steps
        self._engine.load(rollout_steps=steps)
        self._set_model_ready(True)

    def _validate_request(self, request: ForecastRequest) -> None:
        if request.preset not in self._accepted_presets:
            accepted = ", ".join(self._accepted_presets)
            raise ValueError(
                f"worker accepts {accepted}, not preset {request.preset!r}"
            )
        if request.steps < 1:
            raise ValueError("steps must be >= 1")
        if request.netcdf_path is None and request.valid_time is None:
            raise ValueError("either netcdf_path or valid_time must be provided")

    def _resolve_cache_dir(self, request: ForecastRequest) -> Path | None:
        if request.cache_dir is None:
            return None
        return Path(request.cache_dir).expanduser().resolve()

    def _last_step_array_event(
        self,
        request: ForecastRequest,
        step_index: int,
        prediction,
        member: int | None,
    ) -> ForecastEvent:
        variable = request.preview_var or next(iter(prediction.surf_vars))
        if variable not in prediction.surf_vars:
            available = ", ".join(sorted(prediction.surf_vars))
            raise ValueError(f"surface variable {variable!r} not found; available: {available}")
        array = prediction.surf_vars[variable][0, -1].detach().float().cpu().numpy()
        valid_time = prediction.metadata.time[-1].isoformat()
        return self._job_event(
            "step",
            request_id=request.request_id,
            step=step_index,
            valid_time=valid_time,
            array_name=variable,
            array_data_b64=encode_array(array),
            ensemble_member=member,
        )

    def _rollout_kwargs(self, request: ForecastRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if request.fine_lead_times is not None:
            kwargs["fine_lead_times"] = request.fine_lead_times
        if request.use_noise_accumulation is not None:
            kwargs["use_noise_accumulation"] = request.use_noise_accumulation
        return kwargs

    def _prepare_member_noise(self, request: ForecastRequest, member: int) -> None:
        model = self._engine.model
        if not hasattr(model, "reset_noise"):
            return
        if request.noise_seed is not None:
            torch.manual_seed(int(request.noise_seed) + member)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(request.noise_seed) + member)
        model.reset_noise()

    def _emit_member_rollout(self, request: ForecastRequest, batch, member: int | None) -> None:
        rollout_kwargs = self._rollout_kwargs(request)
        if request.output_mode == "export_paths":
            export_paths = self._engine.rollout_and_export(
                batch,
                request.steps,
                export_dir=request.export_dir,
                async_export=request.async_export,
                **rollout_kwargs,
            )
            for step_index, path in enumerate(export_paths):
                self._emit(
                    self._job_event(
                        "step",
                        request_id=request.request_id,
                        step=step_index,
                        export_path=str(path),
                        ensemble_member=member,
                    )
                )
            return

        stream = self._engine.rollout_stream(batch, request.steps, **rollout_kwargs)
        for step_index, prediction in enumerate(stream):
            if request.output_mode == "last_step_array" and step_index == request.steps - 1:
                self._emit(self._last_step_array_event(request, step_index, prediction, member))
                continue
            valid_time = prediction.metadata.time[-1].isoformat()
            self._emit(
                self._job_event(
                    "step",
                    request_id=request.request_id,
                    step=step_index,
                    valid_time=valid_time,
                    ensemble_member=member,
                )
            )

    def run_forecast(self, request: ForecastRequest) -> None:
        """Run one forecast on the compute plane (``accepted`` already emitted)."""
        self._validate_request(request)
        if request.ensemble_members is not None and request.ensemble_members < 1:
            raise ValueError("ensemble_members must be >= 1 when set")
        self._emit(self._job_event("preparing", request_id=request.request_id))
        self._prepare_engine_for(request.preset)

        if request.netcdf_path is not None:
            batch = self._engine.prepare_from_netcdf(
                request.netcdf_path,
                rollout_steps=request.steps,
                overlap=request.overlap,
            )
        else:
            cache_dir = self._resolve_cache_dir(request)
            ingest = self._downloader.ingest_request(
                request.parsed_valid_time(),
                cache_dir=cache_dir,
                time_index=request.time_index,
                download=request.download,
            )
            batch = self._engine.prepare(
                ingest,
                rollout_steps=request.steps,
                overlap=request.overlap,
            )

        self._emit(self._job_event("running", request_id=request.request_id))

        members = request.ensemble_members
        if members is None or members <= 1:
            self._emit_member_rollout(request, batch, member=None)
        else:
            for member in range(members):
                self._prepare_member_noise(request, member)
                # Fresh IC clone per member; rollout advances a working copy in place.
                member_batch = batch._fmap(lambda tensor: tensor.clone())
                self._emit_member_rollout(request, member_batch, member=member)

        self._emit(self._job_event("completed", request_id=request.request_id))

    def _execute_forecast_job(self, request: ForecastRequest) -> None:
        """Run one job with error handling (compute plane or inline)."""
        self._set_busy(True)
        try:
            self.run_forecast(request)
            self._set_model_ready(True)
        except InsufficientVramError as exc:
            self._emit_failed(request.request_id, str(exc))
            self._fatal_error = exc
            self._running = False
        except Exception as exc:
            try:
                self._engine.release_gpu(move_model_to_cpu=True)
            except Exception:
                pass
            # The weights just moved to the CPU, so health must stop reporting "ready".
            self._set_model_ready(False)
            self._emit_failed(request.request_id, str(exc))
        finally:
            self._set_busy(False)

    def _emit_failed(self, request_id: str | None, error: str) -> None:
        self._emit(self._job_event("failed", request_id=request_id, error=error))

    def _fail_queued_jobs(self, reason: str) -> None:
        """Answer every accepted-but-unstarted job so the coordinator frees its slot."""
        while True:
            try:
                item = self._job_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, ForecastRequest):
                self._emit_failed(item.request_id, reason)

    def _compute_loop(self) -> None:
        while True:
            item = self._job_queue.get()
            if item is _COMPUTE_STOP:
                break
            assert isinstance(item, ForecastRequest)
            self._execute_forecast_job(item)
            if self._fatal_error is not None:
                # The control loop stops on _running=False; nothing will run the queued jobs.
                self._fail_queued_jobs(f"worker stopped after fatal error: {self._fatal_error}")
                break

    def _start_compute_thread(self) -> None:
        if self._compute_thread is not None:
            return
        self._detached_compute = True
        self._compute_thread = threading.Thread(
            target=self._compute_loop,
            name=f"forecast-compute-{self.worker_id}",
            daemon=True,
        )
        self._compute_thread.start()

    def _stop_compute_thread(self) -> None:
        if self._compute_thread is None:
            return
        self._job_queue.put(_COMPUTE_STOP)
        self._compute_thread.join(timeout=_COMPUTE_JOIN_TIMEOUT_S)
        self._compute_thread = None
        self._detached_compute = False

    def handle_command(self, command: ForecastCommand) -> bool:
        """Handle one control-plane command. Returns False when the worker should stop.

        ``health`` / ``shutdown`` run on the calling thread. ``forecast`` is enqueued for
        the compute thread when ``serve_forever`` is active; otherwise it runs inline
        (``serve_once`` / tests without a compute thread).
        """
        if command.kind == "shutdown":
            self._running = False
            if self._detached_compute:
                self._job_queue.put(_COMPUTE_STOP)
            return False
        if command.kind == "health":
            self._emit(
                ForecastEvent(
                    kind="health",
                    worker_preset=self._loaded_preset,
                    worker_id=self.worker_id,
                    worker_device=self.device,
                    worker_capacity=self.capacity,
                    message=self._health_message(),
                )
            )
            return True
        if command.kind == "forecast":
            if command.request is None:
                self._emit_failed(None, "forecast command requires a request payload")
                return True
            request = command.request
            try:
                self._validate_request(request)
                if request.ensemble_members is not None and request.ensemble_members < 1:
                    raise ValueError("ensemble_members must be >= 1 when set")
            except Exception as exc:
                self._emit_failed(request.request_id, str(exc))
                return True
            # Queued / starting acknowledgment from the control plane.
            self._emit(self._job_event("accepted", request_id=request.request_id))
            if self._detached_compute:
                self._job_queue.put(request)
            else:
                self._execute_forecast_job(request)
                if isinstance(self._fatal_error, InsufficientVramError):
                    self.close()
                    raise SystemExit(1) from self._fatal_error
            return True
        self._emit_failed(None, f"unsupported command {command.kind!r}")
        return True

    def serve_forever(self) -> None:
        """Control plane: poll ZMQ. Compute plane: dedicated thread for forecasts."""
        poller = zmq.Poller()
        poller.register(self._command_socket, zmq.POLLIN)

        try:
            if self._config.preload:
                self.ensure_loaded()
                self._emit_ready(message=WORKER_STATUS_READY)
            else:
                # Sockets are bound; GPU load still happens on first forecast.
                self._emit_ready(message=WORKER_STATUS_LISTENING)
            self._start_compute_thread()
            while self._running:
                if self._fatal_error is not None:
                    break
                self._flush_events()
                events = poller.poll(timeout=self._config.poll_timeout_ms)
                if not events:
                    continue
                data = self._command_socket.recv()
                command = decode_command(data)
                if not self.handle_command(command):
                    break
            if isinstance(self._fatal_error, InsufficientVramError):
                raise SystemExit(1) from self._fatal_error
        finally:
            self.close()

    def serve_once(self) -> bool:
        """Process a single command inline (no compute thread). Returns False on shutdown."""
        if not self._command_socket.poll(timeout=self._config.poll_timeout_ms):
            return True
        command = decode_command(self._command_socket.recv())
        return self.handle_command(command)


def install_signal_handlers(worker: ForecastWorker) -> None:
    def _handler(_signum: int, _frame: object) -> None:
        worker._running = False
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


_IPC_SCHEME = "ipc://"
_BIND_POLL_INTERVAL_S = 0.01
# A spawned worker imports torch and builds its engine before it binds.
_DEFAULT_BIND_TIMEOUT_S = 120.0


def wait_for_bind(addr: str, *, timeout_s: float = _DEFAULT_BIND_TIMEOUT_S) -> None:
    """Block until an ``ipc://`` endpoint has its socket file, then return at once.

    Other schemes cannot be probed without connecting, and ZMQ connects retry on
    their own, so they return immediately. Raises TimeoutError if the socket
    file never appears.
    """
    if not addr.startswith(_IPC_SCHEME):
        return
    socket_path = Path(addr[len(_IPC_SCHEME):])
    deadline = time.monotonic() + timeout_s
    while not socket_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{addr} was not bound within {timeout_s:.1f}s")
        time.sleep(_BIND_POLL_INTERVAL_S)
