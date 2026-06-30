from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Generator, Iterable

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.checkpoint import CheckpointLoader
from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.hooks import RolloutObserver
from flash_aurora.engine.core.hub import HF_MIRROR_ENDPOINT
from flash_aurora.engine.core.asset_root import normalize_asset_root
from flash_aurora.engine.core.paths import AssetStore, normalize_asset_path, normalize_user_path
from flash_aurora.engine.core.prepare import LoadTiming, overlap_ic_and_load, serial_ic_then_load
from flash_aurora.engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from flash_aurora.engine.core.rollout_session import RolloutSession
from flash_aurora.engine.egress.export import PipelineRolloutExporter, RolloutExporter
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.adapters import IngestRequest
from flash_aurora.engine.ingress.validator import BatchValidator
from flash_aurora.engine.runtime.graph_pool import GraphPool
from flash_aurora.engine.runtime.gpu_guard import (
    GpuGuardRegistry,
    GpuGuardTicket,
    gpu_guard_enabled,
    resolve_guard_dir,
    try_local_cuda_cleanup,
)
from flash_aurora.engine.distributed import (
    DistributedConfig,
    apply_pipeline_parallel,
    is_pipeline_parallel,
    plan_parallelism,
)
from flash_aurora.engine.distributed.pipeline import distributed_status, restore_pipeline_parallel


class AuroraEngine:
    def __init__(
        self,
        config: EngineConfig,
        presets: PresetRegistry | None = None,
    ) -> None:
        self.config = config
        if self.config.user_cwd is None:
            self.config.user_cwd = Path.cwd()
        self._presets = presets or DEFAULT_PRESETS
        self._model: Aurora | None = None
        self._loader = CheckpointLoader(config)
        self._validator = BatchValidator(config.variant)
        self._graph_pool = GraphPool()
        self._exporter = RolloutExporter(self._resolved_export_dir())
        self._gpu_ticket: GpuGuardTicket | None = None
        self._forward_warmed = False
        self._closed = False
        self._parallel_plan = None

    def __enter__(self) -> AuroraEngine:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release GPU resources and close exporter state."""
        if self._closed:
            return
        self._closed = True
        try:
            self._exporter.close()
        finally:
            try:
                self.release_gpu(move_model_to_cpu=False)
            finally:
                self._model = None
                self._graph_pool.clear()
                self._forward_warmed = False

    def _resolved_export_dir(self) -> Path:
        if self.config.export_dir is not None:
            return normalize_user_path(
                self.config.export_dir,
                user_cwd=self.config.user_cwd,
            )
        return self.asset_dir / "output"

    def set_export_dir(self, export_dir: Path | str) -> Path:
        """Update the rollout export directory and refresh the exporter."""
        resolved = normalize_user_path(export_dir, user_cwd=self.config.user_cwd)
        self.config.export_dir = resolved
        self._exporter = RolloutExporter(resolved)
        return resolved

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        asset_root: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        allow_hub_download: bool | None = None,
        hf_endpoint: str | None = None,
        hf_mirror: bool = False,
        hf_revision: str | None = None,
        hf_token: str | None = None,
        export_dir: Path | str | None = None,
        inference_precision: str | None = None,
        overlap_ic_load: bool | None = None,
        async_export: bool | None = None,
        export_pool_size: int | None = None,
        export_max_inflight: int | None = None,
        export_use_egress_stream: bool | None = None,
        ic_cache: bool | None = None,
        forward_warmup_iters: int | None = None,
        distributed: "DistributedConfig | None" = None,
        presets: PresetRegistry | None = None,
    ) -> AuroraEngine:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        user_cwd = Path.cwd()
        config.user_cwd = user_cwd
        config.asset_root = normalize_asset_root(asset_root, user_cwd=user_cwd)
        if checkpoint_path is not None:
            config.checkpoint_path = normalize_user_path(checkpoint_path, user_cwd=user_cwd)
        if export_dir is not None:
            config.export_dir = normalize_user_path(export_dir, user_cwd=user_cwd)
        if allow_hub_download is not None:
            config.allow_hub_download = allow_hub_download
        if hf_mirror:
            config.hf_endpoint = HF_MIRROR_ENDPOINT
        elif hf_endpoint is not None:
            config.hf_endpoint = hf_endpoint
        if hf_revision is not None:
            config.hf_revision = hf_revision
        if hf_token is not None:
            config.hf_token = hf_token
        if inference_precision is not None:
            config.inference_precision = inference_precision
        if overlap_ic_load is not None:
            config.overlap_ic_load = overlap_ic_load
        if async_export is not None:
            config.async_export = async_export
        if export_pool_size is not None:
            config.export_pool_size = export_pool_size
        if export_max_inflight is not None:
            config.export_max_inflight = export_max_inflight
        if export_use_egress_stream is not None:
            config.export_use_egress_stream = export_use_egress_stream
        if ic_cache is not None:
            config.ic_cache = ic_cache
        if forward_warmup_iters is not None:
            config.forward_warmup_iters = forward_warmup_iters
        if distributed is not None:
            config.distributed = distributed
            config.cuda_graph = False
            config.gpu_guard = False
        engine = cls(config, presets=registry)
        engine.config.preset_name = name
        return engine

    @property
    def asset_dir(self) -> Path:
        store = AssetStore(root=self.config.asset_root)
        return store.resolve_root(self.config.asset_root, self.config.user_cwd)

    @property
    def fetched_dir(self) -> Path:
        """Deprecated alias for :attr:`asset_dir`."""
        return self.asset_dir

    def _allowed_roots(self) -> tuple[Path, ...]:
        return AssetStore(root=self.config.asset_root).allowed_roots(
            self.config.asset_root,
            self.config.user_cwd,
        )

    def _builder(self) -> InitialConditionBuilder:
        return InitialConditionBuilder(self.config)

    @property
    def model(self) -> Aurora:
        if self._model is None:
            raise RuntimeError("Call load() before using the model.")
        return self._model

    def _device_index(self) -> int:
        device = self.config.device
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        return 0

    def acquire_gpu(self, *, rollout_steps: int | None = None) -> GpuGuardTicket | None:
        """Reserve GPU memory across processes (share small jobs, queue large ones)."""
        if self.config.distributed is not None:
            return None
        if self._gpu_ticket is not None:
            return self._gpu_ticket
        if not self.config.gpu_guard or not gpu_guard_enabled():
            return None
        steps = self.config.gpu_rollout_steps if rollout_steps is None else rollout_steps
        preset = self.config.preset_name or self.config.variant.name
        registry = GpuGuardRegistry(resolve_guard_dir(self.config.asset_root))
        self._gpu_ticket = registry.acquire(
            device_index=self._device_index(),
            preset=preset,
            variant=self.config.variant,
            rollout_steps=steps,
            inference_precision=self.config.inference_precision,
            timeout=self.config.gpu_guard_timeout,
        )
        return self._gpu_ticket

    def release_gpu(self, *, move_model_to_cpu: bool = True) -> None:
        """Release the cross-process GPU lease and optionally free local CUDA cache."""
        if self._gpu_ticket is not None:
            self._gpu_ticket.release()
            self._gpu_ticket = None
        if move_model_to_cpu and self._model is not None:
            if is_pipeline_parallel(self._model):
                restore_pipeline_parallel(self._model)
            self._model.cpu()
            self._parallel_plan = None
        distributed = self.config.distributed
        if distributed is not None:
            for device_name in distributed.devices:
                if device_name.startswith("cuda:"):
                    try_local_cuda_cleanup(device_index=int(device_name.split(":", 1)[1]))
        else:
            try_local_cuda_cleanup(device_index=self._device_index())

    def distributed_status(self) -> dict[str, object]:
        """Return pipeline-parallel placement metadata when multi-GPU inference is active."""
        if self._model is None:
            return {"enabled": False, "loaded": False}
        if is_pipeline_parallel(self._model):
            status: dict[str, object] = distributed_status(self._model)
        else:
            status = {"enabled": False}
        status["loaded"] = True
        if self.config.distributed is not None:
            status["strategy"] = "pipeline"
        if self._parallel_plan is not None:
            status["plan"] = {
                "devices": self._parallel_plan.devices,
                "estimated_peak_gib": self._parallel_plan.estimated_peak_gib,
                "estimated_per_device_gib": self._parallel_plan.estimated_per_device_gib,
            }
            from flash_aurora.engine.distributed.plan import estimate_device_busy_fraction

            status["estimated_busy_fraction"] = dict(
                estimate_device_busy_fraction(self._parallel_plan)
            )
        return status

    def gpu_guard_status(self):
        """Return active leases and queue entries for this engine's device."""
        registry = GpuGuardRegistry(resolve_guard_dir(self.config.asset_root))
        return registry.status(device_index=self._device_index())

    def load(self, *, rollout_steps: int | None = None) -> Aurora:
        try:
            self.acquire_gpu(rollout_steps=rollout_steps)
            self._model = self._load_model_to_device(rollout_steps=rollout_steps)[0]
            return self._model
        except Exception:
            self.release_gpu()
            raise

    def _load_model_to_device(
        self,
        *,
        rollout_steps: int | None = None,
    ) -> tuple[Aurora, LoadTiming]:
        t0 = time.perf_counter()
        model = self._loader.build_model()
        build_model_ms = (time.perf_counter() - t0) * 1000.0

        # Checkpoint stays on CPU until weights are loaded; H2D happens once below.
        t0 = time.perf_counter()
        self._loader.load(model)
        load_ckpt_ms = (time.perf_counter() - t0) * 1000.0

        distributed = self.config.distributed
        if distributed is not None:
            steps = rollout_steps if rollout_steps is not None else distributed.rollout_steps
            dist_config = replace(distributed, rollout_steps=steps)
            plan = plan_parallelism(
                self.config.variant,
                dist_config,
                inference_precision=self.config.inference_precision,
            )
            self._parallel_plan = plan
            self.config.device = plan.input_device
            apply_pipeline_parallel(model, plan)
            model_h2d_ms = 0.0
        else:
            device = torch.device(self.config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")

            t0 = time.perf_counter()
            if device.type == "cuda":
                model.to(device, non_blocking=True)
                torch.cuda.synchronize(device)
            else:
                model.to(device)
            model_h2d_ms = (time.perf_counter() - t0) * 1000.0

        timing = LoadTiming(
            build_model_ms=build_model_ms,
            load_ckpt_ms=load_ckpt_ms,
            model_h2d_ms=model_h2d_ms,
        )
        return model, timing

    def _resolve_overlap(self, overlap: bool | None) -> bool:
        if overlap is not None:
            return overlap
        return self.config.overlap_ic_load

    def _resolve_async_export(self, async_export: bool | None) -> bool:
        if async_export is not None:
            return async_export
        distributed = self.config.distributed
        if distributed is not None and distributed.overlap_rollout:
            return True
        return self.config.async_export

    def _use_distributed_rollout(self) -> bool:
        distributed = self.config.distributed
        if distributed is None or not distributed.overlap_rollout:
            return False
        from flash_aurora.engine.distributed.pipeline import is_pipeline_parallel

        return is_pipeline_parallel(self._model)

    def _distributed_rollout_stream(
        self,
        batch: Batch,
        steps: int,
        *,
        observers: Iterable[RolloutObserver] | None = None,
        on_step_export: Callable[[int, Batch], None] | None = None,
    ) -> Generator[Batch, None, None]:
        from flash_aurora.engine.distributed.rollout_pipeline import distributed_rollout

        distributed = self.config.distributed
        assert distributed is not None
        overlap_export = on_step_export is not None and distributed.overlap_rollout

        with torch.inference_mode():
            for step_index, pred in enumerate(
                distributed_rollout(
                    self._model,
                    batch,
                    steps,
                    overlap_export=overlap_export,
                    on_step_export=on_step_export,
                )
            ):
                for observer in observers or ():
                    observer.on_step(step_index, pred)
                yield pred

    def _pipeline_exporter(self, export_dir: Path) -> PipelineRolloutExporter:
        return PipelineRolloutExporter.async_netcdf(
            export_dir,
            pool_size=self.config.export_pool_size,
            max_inflight=self.config.export_max_inflight,
            use_egress_stream=self.config.export_use_egress_stream,
        )

    def prepare(
        self,
        request: IngestRequest,
        *,
        rollout_steps: int | None = None,
        overlap: bool | None = None,
    ) -> Batch:
        """Build initial conditions and load the model.

        When overlap is enabled (``EngineConfig.overlap_ic_load``, default True),
        IC construction runs on a background thread while the model loads.
        """
        self.acquire_gpu(rollout_steps=rollout_steps)
        build_ic = lambda: self._builder().from_source(request)
        load = lambda: self._load_model_to_device(rollout_steps=rollout_steps)
        use_overlap = self._resolve_overlap(overlap)

        try:
            if use_overlap:
                batch, model, _timing = overlap_ic_and_load(build_ic, load)
            else:
                batch, model, _timing = serial_ic_then_load(build_ic, load)
            self._model = model
            self._maybe_warmup(batch)
            return batch
        except Exception:
            self.release_gpu()
            raise

    def prepare_from_netcdf(
        self,
        path: Path | str,
        *,
        rollout_steps: int | None = None,
        overlap: bool | None = None,
    ) -> Batch:
        """Load IC from a NetCDF path and initialize the model."""
        resolved = Path(path)
        self.acquire_gpu(rollout_steps=rollout_steps)
        build_ic = lambda: self._builder().from_netcdf_path(resolved)
        load = lambda: self._load_model_to_device(rollout_steps=rollout_steps)
        use_overlap = self._resolve_overlap(overlap)

        try:
            if use_overlap:
                batch, model, _timing = overlap_ic_and_load(build_ic, load)
            else:
                batch, model, _timing = serial_ic_then_load(build_ic, load)
            self._model = model
            self._maybe_warmup(batch)
            return batch
        except Exception:
            self.release_gpu()
            raise

    def _maybe_warmup(self, batch: Batch) -> None:
        if self._forward_warmed or self.config.forward_warmup_iters <= 0:
            return
        self._graph_pool.warmup(self.model, batch, self.config)
        self._forward_warmed = True

    def warmup(self, batch: Batch, *, forward_iters: int | None = None) -> None:
        """Run forward warmup (CuTe JIT) and optional CUDA graph capture.

        Idempotent unless ``forward_iters`` is passed explicitly.
        """
        if forward_iters is not None and forward_iters <= 0:
            return
        if forward_iters is not None or not self._forward_warmed:
            self._graph_pool.warmup(
                self.model,
                batch,
                self.config,
                forward_iters=forward_iters,
            )
            self._forward_warmed = True

    def predict(self, batch: Batch) -> Batch:
        self.validate(batch)
        self._maybe_warmup(batch)
        with torch.inference_mode():
            return self.model.forward(batch)

    def run_from_netcdf(self, path: Path | str, steps: int = 1) -> list[Batch]:
        batch = self._builder().from_netcdf_path(Path(path))
        if steps == 1:
            return [self.predict(batch)]
        return list(self.rollout_stream(batch, steps))

    def run_from_adapter(self, request: IngestRequest, steps: int = 1) -> list[Batch]:
        batch = self._builder().from_source(request)
        if steps == 1:
            return [self.predict(batch)]
        return list(self.rollout_stream(batch, steps))

    def validate(self, batch: Batch) -> None:
        self._validator.validate(batch)

    def rollout_stream(
        self,
        batch: Batch,
        steps: int,
        observers: Iterable[RolloutObserver] | None = None,
    ) -> Generator[Batch, None, None]:
        self.acquire_gpu(rollout_steps=steps)
        if self._gpu_ticket is not None:
            self._gpu_ticket.heartbeat()
        self.validate(batch)
        self._maybe_warmup(batch)
        try:
            if self._use_distributed_rollout():
                yield from self._distributed_rollout_stream(batch, steps, observers=observers)
            else:
                session = RolloutSession(self.model, observers)
                yield from session.run(batch, steps)
        except Exception:
            self.release_gpu()
            raise

    def rollout_and_export(
        self,
        batch: Batch,
        steps: int,
        *,
        export_dir: Path | str | None = None,
        async_export: bool | None = None,
    ) -> Generator[Path, None, None]:
        if export_dir is not None:
            self.set_export_dir(export_dir)

        resolved_dir = self._resolved_export_dir()
        self._exporter = RolloutExporter(resolved_dir)
        if self._use_distributed_rollout() and self._resolve_async_export(async_export):
            with self._pipeline_exporter(resolved_dir) as exporter:
                exported: dict[int, Path] = {}

                def on_export(step_index: int, cpu_batch: Batch) -> None:
                    exported[step_index] = exporter.write_owned_step(step_index, cpu_batch)

                for step_index, _prediction in enumerate(
                    self._distributed_rollout_stream(batch, steps, on_step_export=on_export)
                ):
                    prior = step_index - 1
                    if prior >= 0 and prior in exported:
                        yield exported[prior]
                last = steps - 1
                if last in exported:
                    yield exported[last]
            return

        if self._resolve_async_export(async_export):
            with self._pipeline_exporter(resolved_dir) as exporter:
                for step_index, prediction in enumerate(self.rollout_stream(batch, steps)):
                    yield exporter.write_step(step_index, prediction)
            return

        for step_index, prediction in enumerate(self.rollout_stream(batch, steps)):
            yield self._exporter.write_step(step_index, prediction)
