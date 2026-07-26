"""Long-lived single-GPU forecast worker (P1)."""

from __future__ import annotations

import signal
import time
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
    ForecastCommand,
    ForecastEvent,
    ForecastRequest,
    decode_command,
    encode_array,
    encode_event,
)


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
        self._context = context or zmq.Context.instance()
        self._engine = engine or self._build_engine()
        self._downloader = downloader or DataDownloader.from_preset(
            config.preset,
            asset_root=config.asset_root,
        )
        self._running = True
        self._model_ready = False
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
        return self._model_ready

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
        if self._config.device is not None:
            return self._config.device
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

    def _build_engine(self) -> AuroraEngine:
        kwargs: dict[str, Any] = {
            "asset_root": self._config.asset_root,
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
        engine = AuroraEngine.from_preset(self._config.preset, **kwargs)
        if self._config.device is not None and not self._config.distributed_devices:
            engine.config.device = self._config.device
        return engine

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._running = False
        try:
            self._engine.close()
        except Exception:
            pass
        self._command_socket.close(linger=0)
        self._event_socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _emit(self, event: ForecastEvent) -> None:
        self._event_socket.send(encode_event(event))

    def _emit_ready(self, *, message: str) -> None:
        # NOBLOCK: startup ready must not stall when no PULL peer is connected yet.
        # Late clients recover via health (message ready|listening).
        try:
            self._event_socket.send(
                encode_event(
                    ForecastEvent(
                        kind="ready",
                        worker_preset=self._config.preset,
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

    def ensure_loaded(self, *, rollout_steps: int | None = None) -> None:
        """Load weights onto the device and mark the worker warm."""
        if self._model_ready:
            return
        steps = rollout_steps if rollout_steps is not None else self._config.preload_rollout_steps
        self._engine.load(rollout_steps=steps)
        self._model_ready = True

    def _validate_request(self, request: ForecastRequest) -> None:
        if request.preset != self._config.preset:
            raise ValueError(
                f"worker preset {self._config.preset!r} does not match "
                f"request preset {request.preset!r}"
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
    ) -> ForecastEvent:
        variable = request.preview_var or next(iter(prediction.surf_vars))
        if variable not in prediction.surf_vars:
            available = ", ".join(sorted(prediction.surf_vars))
            raise ValueError(f"surface variable {variable!r} not found; available: {available}")
        array = prediction.surf_vars[variable][0, -1].detach().float().cpu().numpy()
        valid_time = prediction.metadata.time[-1].isoformat()
        return ForecastEvent(
            kind="step",
            request_id=request.request_id,
            step=step_index,
            valid_time=valid_time,
            array_name=variable,
            array_data_b64=encode_array(array),
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
                    ForecastEvent(
                        kind="step",
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
                event = self._last_step_array_event(request, step_index, prediction)
                self._emit(
                    ForecastEvent(
                        kind=event.kind,
                        request_id=event.request_id,
                        step=event.step,
                        valid_time=event.valid_time,
                        array_name=event.array_name,
                        array_data_b64=event.array_data_b64,
                        ensemble_member=member,
                    )
                )
                continue
            valid_time = prediction.metadata.time[-1].isoformat()
            self._emit(
                ForecastEvent(
                    kind="step",
                    request_id=request.request_id,
                    step=step_index,
                    valid_time=valid_time,
                    ensemble_member=member,
                )
            )

    def run_forecast(self, request: ForecastRequest) -> None:
        self._validate_request(request)
        if request.ensemble_members is not None and request.ensemble_members < 1:
            raise ValueError("ensemble_members must be >= 1 when set")
        self._emit(ForecastEvent(kind="accepted", request_id=request.request_id))
        self._emit(ForecastEvent(kind="preparing", request_id=request.request_id))

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

        self._emit(ForecastEvent(kind="running", request_id=request.request_id))

        members = request.ensemble_members
        if members is None or members <= 1:
            self._emit_member_rollout(request, batch, member=None)
        else:
            for member in range(members):
                self._prepare_member_noise(request, member)
                # Fresh IC clone per member; rollout advances a working copy in place.
                member_batch = batch._fmap(lambda tensor: tensor.clone())
                self._emit_member_rollout(request, member_batch, member=member)

        self._emit(ForecastEvent(kind="completed", request_id=request.request_id))

    def handle_command(self, command: ForecastCommand) -> bool:
        """Handle one command. Returns False when the worker should stop."""
        if command.kind == "shutdown":
            return False
        if command.kind == "health":
            self._emit(
                ForecastEvent(
                    kind="health",
                    worker_preset=self._config.preset,
                    worker_id=self.worker_id,
                    worker_device=self.device,
                    worker_capacity=self.capacity,
                    message="ready" if self._model_ready else "listening",
                )
            )
            return True
        if command.kind == "forecast":
            if command.request is None:
                self._emit(
                    ForecastEvent(
                        kind="failed",
                        error="forecast command requires a request payload",
                    )
                )
                return True
            try:
                self.run_forecast(command.request)
                # prepare()/rollout paths load the model; treat success as warm.
                self._model_ready = True
            except InsufficientVramError as exc:
                request_id = command.request.request_id
                self._emit(
                    ForecastEvent(
                        kind="failed",
                        request_id=request_id,
                        error=str(exc),
                    )
                )
                self.close()
                raise SystemExit(1) from exc
            except Exception as exc:
                request_id = command.request.request_id
                try:
                    self._engine.release_gpu(move_model_to_cpu=True)
                except Exception:
                    pass
                self._emit(
                    ForecastEvent(
                        kind="failed",
                        request_id=request_id,
                        error=str(exc),
                    )
                )
            return True
        self._emit(ForecastEvent(kind="failed", error=f"unsupported command {command.kind!r}"))
        return True

    def serve_forever(self) -> None:
        poller = zmq.Poller()
        poller.register(self._command_socket, zmq.POLLIN)

        try:
            if self._config.preload:
                self.ensure_loaded()
                self._emit_ready(message="ready")
            else:
                # Sockets are bound; GPU load still happens on first forecast.
                self._emit_ready(message="listening")
            while self._running:
                events = poller.poll(timeout=self._config.poll_timeout_ms)
                if not events:
                    continue
                data = self._command_socket.recv()
                command = decode_command(data)
                if not self.handle_command(command):
                    break
        finally:
            self.close()

    def serve_once(self) -> bool:
        """Process a single command (for tests). Returns False on shutdown."""
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


def wait_for_bind(addr: str, *, timeout_s: float = 5.0) -> None:
    """Allow ZMQ bind/connect ordering in tests."""
    del addr
    time.sleep(0.05)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(0.01)
