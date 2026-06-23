"""Engine lifecycle timings excluding data download."""

from __future__ import annotations

import dataclasses
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.checkpoint import CheckpointLoader
from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.egress.export import RolloutExporter
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.download import DataDownloader
from flash_aurora.engine.ingress.download.layout import cache_subdir

from _stage_timing import StageTiming, time_forward_stages


@dataclass(frozen=True)
class EngineCycleTiming:
    preset: str
    rollout_steps: int
    inference_precision: str
    ingest_ms: float
    build_ic_ms: float
    build_model_ms: float
    load_ckpt_ms: float
    model_h2d_ms: float
    batch_prep_ms: float
    forward_per_step_ms: float
    rollout_overhead_ms: float
    rollout_total_ms: float
    export_per_step_ms: float | None
    forward_stages: StageTiming | None
    bottleneck: str

    @property
    def load_total_ms(self) -> float:
        return self.build_model_ms + self.load_ckpt_ms + self.model_h2d_ms

    @property
    def cpu_total_ms(self) -> float:
        return self.ingest_ms + self.build_ic_ms + self.load_total_ms

    @property
    def gpu_compute_ms(self) -> float:
        return self.batch_prep_ms + self.rollout_total_ms

    @property
    def engine_total_ms(self) -> float:
        export = (self.export_per_step_ms or 0.0) * self.rollout_steps
        return (
            self.ingest_ms
            + self.build_ic_ms
            + self.load_total_ms
            + self.batch_prep_ms
            + self.rollout_total_ms
            + export
        )

    def stage_rows(self) -> list[tuple[str, float, float]]:
        """Return (label, ms, pct_of_total) sorted by ms descending."""
        export_total = (self.export_per_step_ms or 0.0) * self.rollout_steps
        rows = [
            ("build_ic", self.build_ic_ms),
            ("load_ckpt", self.load_ckpt_ms),
            ("model_h2d", self.model_h2d_ms),
            ("build_model", self.build_model_ms),
            ("batch_prep_h2d", self.batch_prep_ms),
            ("rollout_forward", self.forward_per_step_ms * self.rollout_steps),
            ("rollout_overhead", self.rollout_overhead_ms),
            ("export", export_total),
            ("ingest_request", self.ingest_ms),
        ]
        total = self.engine_total_ms or 1.0
        return sorted(
            ((name, ms, 100.0 * ms / total) for name, ms in rows if ms > 0.0),
            key=lambda row: row[1],
            reverse=True,
        )


def _prepare_rollout_batch(model: Aurora, batch: Batch) -> Batch:
    batch = model.batch_transform_hook(batch)
    param = next(model.parameters())
    batch = batch.type(param.dtype)
    batch = batch.crop(model.patch_size)
    return batch.to(param.device)


def _time_export_step(exporter: RolloutExporter, step_index: int, batch: Batch) -> float:
    t0 = time.perf_counter()
    exporter.write_step(step_index, batch)
    return (time.perf_counter() - t0) * 1000.0


def _time_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Return batch_prep_ms, forward_per_step_ms, overhead_ms, rollout_total_ms."""
    t0 = time.perf_counter()
    batch = _prepare_rollout_batch(model, batch)
    batch_prep_ms = (time.perf_counter() - t0) * 1000.0

    step_ms: list[float] = []
    overhead_ms = 0.0

    with torch.inference_mode():
        loop_t0 = time.perf_counter()
        for _ in range(steps):
            if device.type == "cuda":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                pred = model.forward(batch)
                e1.record()
                torch.cuda.synchronize(device)
                step_ms.append(e0.elapsed_time(e1))
            else:
                fwd_t0 = time.perf_counter()
                pred = model.forward(batch)
                step_ms.append((time.perf_counter() - fwd_t0) * 1000.0)

            oh_t0 = time.perf_counter()
            batch = dataclasses.replace(
                pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.surf_vars.items()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.atmos_vars.items()
                },
            )
            overhead_ms += (time.perf_counter() - oh_t0) * 1000.0

        rollout_total_ms = (time.perf_counter() - loop_t0) * 1000.0

    forward_per_step = sum(step_ms) / len(step_ms) if step_ms else 0.0
    return batch_prep_ms, forward_per_step, overhead_ms, rollout_total_ms


def measure_engine_cycle(
    preset_name: str,
    config: EngineConfig,
    *,
    valid_time: datetime,
    time_index: int,
    rollout_steps: int,
    device: torch.device,
    forward_stage_warmup: int = 1,
    forward_stage_repeat: int = 3,
    include_export: bool = True,
    export_dir: Path | None = None,
    ic_loader: Callable[[], Batch] | None = None,
) -> EngineCycleTiming:
    """Profile engine stages with cached ingress only (no download)."""
    ingest_ms = 0.0
    build_ic_ms = 0.0

    if ic_loader is not None:
        t0 = time.perf_counter()
        batch = ic_loader()
        build_ic_ms = (time.perf_counter() - t0) * 1000.0
    else:
        cache = config.asset_root.expanduser().resolve() / cache_subdir(config.source)
        downloader = DataDownloader(config)
        missing = downloader.missing(valid_time, cache_dir=cache)
        if missing:
            raise FileNotFoundError(
                f"Incomplete ingress cache for preset {preset_name!r}: missing {missing}"
            )

        t0 = time.perf_counter()
        request = downloader.ingest_request(
            valid_time,
            cache_dir=cache,
            time_index=time_index,
            download=False,
        )
        ingest_ms = (time.perf_counter() - t0) * 1000.0

        builder = InitialConditionBuilder(config)
        t0 = time.perf_counter()
        batch = builder.from_source(request)
        build_ic_ms = (time.perf_counter() - t0) * 1000.0

    assert batch is not None

    loader = CheckpointLoader(config)
    t0 = time.perf_counter()
    model = loader.build_model()
    build_model_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    loader.load(model)
    load_ckpt_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_h2d_ms = (time.perf_counter() - t0) * 1000.0

    batch_prep_ms, forward_per_step_ms, rollout_overhead_ms, rollout_total_ms = _time_rollout(
        model,
        batch,
        rollout_steps,
        device=device,
    )

    export_per_step_ms: float | None = None
    if include_export:
        out_dir = export_dir or (config.asset_root / ".flash-aurora" / "bench_export")
        exporter = RolloutExporter(out_dir)
        with torch.inference_mode():
            gpu_batch = _prepare_rollout_batch(model, batch)
            pred = model.forward(gpu_batch)
        export_samples: list[float] = []
        for _ in range(min(2, rollout_steps)):
            export_samples.append(_time_export_step(exporter, 0, pred))
        export_per_step_ms = sum(export_samples) / len(export_samples)

    forward_stages: StageTiming | None = None
    if device.type == "cuda":
        with torch.inference_mode():
            gpu_batch = _prepare_rollout_batch(model, batch)
            forward_stages, _ = time_forward_stages(
                model,
                gpu_batch,
                warmup=forward_stage_warmup,
                repeat=forward_stage_repeat,
                device=device,
            )

    rows = EngineCycleTiming(
        preset=preset_name,
        rollout_steps=rollout_steps,
        inference_precision=config.inference_precision or "default",
        ingest_ms=ingest_ms,
        build_ic_ms=build_ic_ms,
        build_model_ms=build_model_ms,
        load_ckpt_ms=load_ckpt_ms,
        model_h2d_ms=model_h2d_ms,
        batch_prep_ms=batch_prep_ms,
        forward_per_step_ms=forward_per_step_ms,
        rollout_overhead_ms=rollout_overhead_ms,
        rollout_total_ms=rollout_total_ms,
        export_per_step_ms=export_per_step_ms,
        forward_stages=forward_stages,
        bottleneck="",
    ).stage_rows()
    bottleneck = rows[0][0] if rows else "unknown"

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return EngineCycleTiming(
        preset=preset_name,
        rollout_steps=rollout_steps,
        inference_precision=config.inference_precision or "default",
        ingest_ms=ingest_ms,
        build_ic_ms=build_ic_ms,
        build_model_ms=build_model_ms,
        load_ckpt_ms=load_ckpt_ms,
        model_h2d_ms=model_h2d_ms,
        batch_prep_ms=batch_prep_ms,
        forward_per_step_ms=forward_per_step_ms,
        rollout_overhead_ms=rollout_overhead_ms,
        rollout_total_ms=rollout_total_ms,
        export_per_step_ms=export_per_step_ms,
        forward_stages=forward_stages,
        bottleneck=bottleneck,
    )


def purge_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
