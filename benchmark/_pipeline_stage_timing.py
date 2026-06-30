"""CUDA-event stage timings for pipeline-parallel Aurora forward."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from flash_aurora.engine.distributed.batch_utils import batch_to_device


@dataclass(frozen=True)
class PipelineStageTiming:
    prepare_ms: float
    encoder_ms: float
    enc_to_bb_ms: float
    backbone_ms: float
    bb_to_dec_ms: float
    decoder_ms: float
    post_ms: float
    total_ms: float

    @property
    def compute_ms(self) -> float:
        return self.encoder_ms + self.backbone_ms + self.decoder_ms + self.post_ms

    @property
    def transfer_ms(self) -> float:
        return self.enc_to_bb_ms + self.bb_to_dec_ms


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    allocated_mib: float
    reserved_mib: float
    active_mib: float


@dataclass(frozen=True)
class PipelineLoadProfile:
    timing: PipelineStageTiming
    peak_allocated_mib: dict[str, float]
    peak_reserved_mib: dict[str, float]
    after_load_allocated_mib: dict[str, float]
    param_bytes: dict[str, int]
    plan_estimated_per_device_gib: tuple[float, ...]
    plan_estimated_peak_gib: float


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _snapshot_devices(device_names: tuple[str, ...]) -> dict[str, DeviceMemorySnapshot]:
    out: dict[str, DeviceMemorySnapshot] = {}
    for name in device_names:
        idx = torch.device(name).index or 0
        out[name] = DeviceMemorySnapshot(
            allocated_mib=torch.cuda.memory_allocated(idx) / 1e6,
            reserved_mib=torch.cuda.memory_reserved(idx) / 1e6,
            active_mib=torch.cuda.memory_stats(idx).get("active_bytes.all.current", 0) / 1e6,
        )
    return out


def _same_device(left: torch.device, right: torch.device) -> bool:
    return left.type == right.type and (left.index or 0) == (right.index or 0)


def _param_bytes_on_device(module: torch.nn.Module, device: torch.device) -> int:
    total = 0
    for param in module.parameters():
        if _same_device(param.device, device):
            total += param.numel() * param.element_size()
    for buf in module.buffers():
        if _same_device(buf.device, device):
            total += buf.numel() * buf.element_size()
    return total


def _run_pipeline_forward_once(
    model: Any,
    batch: Any,
) -> tuple[PipelineStageTiming, Any]:
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing
    from flash_aurora.engine.distributed.pipeline import parallel_plan

    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("model is not pipeline-parallel")

    enc_dev = torch.device(plan.encoder_device)
    bb_dev = torch.device(plan.backbone_device)
    dec_dev = torch.device(plan.decoder_device)

    import time

    t_prepare0 = time.perf_counter()
    batch = batch_to_device(batch, enc_dev)
    batch, transformed_batch, patch_res = model._prepare_encoder_batch(batch)
    _sync(enc_dev)
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    e_enc0 = torch.cuda.Event(enable_timing=True)
    e_enc1 = torch.cuda.Event(enable_timing=True)
    e_bb0 = torch.cuda.Event(enable_timing=True)
    e_bb1 = torch.cuda.Event(enable_timing=True)
    e_dec0 = torch.cuda.Event(enable_timing=True)
    e_dec1 = torch.cuda.Event(enable_timing=True)
    e_post1 = torch.cuda.Event(enable_timing=True)

    with _device_context(enc_dev):
        e_enc0.record()
        with torch.inference_mode():
            x = run_with_encoder_decoder_routing(
                model.encoder,
                transformed_batch,
                autocast_bf16=model.autocast_encoder_decoder,
                use_tensor_core=model.encoder_decoder_use_tensor_core,
                lead_time=model.timestep,
            )
        e_enc1.record()
        _sync(enc_dev)

    t_xfer0 = time.perf_counter()
    if bb_dev != enc_dev:
        x = x.to(bb_dev, non_blocking=True)
        _sync(bb_dev)
    enc_to_bb_ms = (time.perf_counter() - t_xfer0) * 1000.0

    with _device_context(bb_dev):
        e_bb0.record()
        with torch.inference_mode():
            x = model._run_backbone(
                x,
                lead_time=model.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        e_bb1.record()
        _sync(bb_dev)

    t_xfer1 = time.perf_counter()
    if plan.decoder_spatial_parallel:
        west_dev = torch.device(plan.decoder_spatial_devices[0])
        east_dev = torch.device(plan.decoder_spatial_devices[1])
        if east_dev != bb_dev:
            _sync(bb_dev)
    else:
        if dec_dev != bb_dev:
            x = x.to(dec_dev, non_blocking=True)
        if dec_dev != enc_dev:
            batch = batch_to_device(batch, dec_dev)
        if dec_dev != bb_dev:
            _sync(dec_dev)
    bb_to_dec_ms = (time.perf_counter() - t_xfer1) * 1000.0

    if plan.decoder_spatial_parallel:
        from flash_aurora.engine.distributed.decoder_spatial import forward_decoder_spatial_parallel

        west_dev = torch.device(plan.decoder_spatial_devices[0])
        east_dev = torch.device(plan.decoder_spatial_devices[1])
        with _device_context(east_dev):
            e_dec0.record()
        with torch.inference_mode():
            pred = forward_decoder_spatial_parallel(
                model,
                x,
                batch,
                patch_res=patch_res,
                lead_time=model.timestep,
                spatial_devices=(west_dev, east_dev),
                autocast_bf16=model.autocast_encoder_decoder,
                use_tensor_core=model.encoder_decoder_use_tensor_core,
            )
        with _device_context(east_dev):
            e_dec1.record()
        with torch.inference_mode():
            pred = model._finish_prediction(batch, pred)
        with _device_context(east_dev):
            e_post1.record()
        _sync(west_dev)
        _sync(east_dev)
    else:
        with _device_context(dec_dev):
            e_dec0.record()
            with torch.inference_mode():
                pred = run_with_encoder_decoder_routing(
                    model.decoder,
                    x,
                    batch,
                    autocast_bf16=model.autocast_encoder_decoder,
                    use_tensor_core=model.encoder_decoder_use_tensor_core,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                )
            e_dec1.record()
            with torch.inference_mode():
                pred = model._finish_prediction(batch, pred)
            e_post1.record()
            _sync(dec_dev)

    timing = PipelineStageTiming(
        prepare_ms=prepare_ms,
        encoder_ms=e_enc0.elapsed_time(e_enc1),
        enc_to_bb_ms=enc_to_bb_ms,
        backbone_ms=e_bb0.elapsed_time(e_bb1),
        bb_to_dec_ms=bb_to_dec_ms,
        decoder_ms=e_dec0.elapsed_time(e_dec1),
        post_ms=e_dec1.elapsed_time(e_post1),
        total_ms=prepare_ms
        + e_enc0.elapsed_time(e_enc1)
        + enc_to_bb_ms
        + e_bb0.elapsed_time(e_bb1)
        + bb_to_dec_ms
        + e_dec0.elapsed_time(e_dec1)
        + e_dec1.elapsed_time(e_post1),
    )
    pred = batch_to_device(pred, enc_dev)
    _sync(enc_dev)
    return timing, pred


def time_pipeline_forward_stages(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device_names: tuple[str, ...],
) -> tuple[PipelineStageTiming, Any]:
    device_indices = tuple(
        dict.fromkeys(torch.device(name).index or 0 for name in device_names)
    )
    with torch.inference_mode():
        for _ in range(warmup):
            _run_pipeline_forward_once(model, batch)
        for idx in device_indices:
            torch.cuda.synchronize(idx)
        if device_indices:
            torch.cuda.empty_cache()

        sums = [0.0] * 8
        pred = None
        for i in range(repeat):
            timing, pred = _run_pipeline_forward_once(model, batch)
            sums[0] += timing.prepare_ms
            sums[1] += timing.encoder_ms
            sums[2] += timing.enc_to_bb_ms
            sums[3] += timing.backbone_ms
            sums[4] += timing.bb_to_dec_ms
            sums[5] += timing.decoder_ms
            sums[6] += timing.post_ms
            sums[7] += timing.total_ms
            if i + 1 < repeat:
                del pred
                pred = None
                torch.cuda.empty_cache()

        n = float(repeat)
        avg = PipelineStageTiming(
            prepare_ms=sums[0] / n,
            encoder_ms=sums[1] / n,
            backbone_ms=sums[3] / n,
            enc_to_bb_ms=sums[2] / n,
            bb_to_dec_ms=sums[4] / n,
            decoder_ms=sums[5] / n,
            post_ms=sums[6] / n,
            total_ms=sums[7] / n,
        )
    return avg, pred


def reset_peak_stats(device_names: tuple[str, ...]) -> None:
    for name in device_names:
        idx = torch.device(name).index or 0
        torch.cuda.reset_peak_memory_stats(idx)


def peak_mib_per_device(device_names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
    alloc: dict[str, float] = {}
    reserved: dict[str, float] = {}
    for name in device_names:
        idx = torch.device(name).index or 0
        alloc[name] = torch.cuda.max_memory_allocated(idx) / 1e6
        reserved[name] = torch.cuda.max_memory_reserved(idx) / 1e6
    return alloc, reserved


def build_load_profile(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
) -> PipelineLoadProfile:
    from flash_aurora.engine.distributed.pipeline import parallel_plan

    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("model is not pipeline-parallel")

    device_names = plan.devices
    enc_dev = torch.device(plan.encoder_device)
    bb_dev = torch.device(plan.backbone_device)
    dec_dev = torch.device(plan.decoder_device)

    after_load = {
        name: snap.allocated_mib for name, snap in _snapshot_devices(device_names).items()
    }
    enc_bytes = _param_bytes_on_device(model.encoder, enc_dev)
    bb_bytes = _param_bytes_on_device(model.backbone, bb_dev)
    dec_bytes = _param_bytes_on_device(model.decoder, dec_dev)
    if plan.decoder_spatial_parallel:
        from flash_aurora.engine.distributed.decoder_spatial import decoder_spatial_replica

        west_dev = torch.device(plan.decoder_spatial_devices[0])
        replica = decoder_spatial_replica(model)
        replica_bytes = _param_bytes_on_device(replica, west_dev) if replica is not None else 0
        param_bytes = {
            plan.encoder_device: enc_bytes + replica_bytes,
            plan.backbone_device: bb_bytes + dec_bytes,
        }
    else:
        param_bytes = {}
        if enc_dev == dec_dev:
            param_bytes[plan.encoder_device] = enc_bytes + dec_bytes
        else:
            param_bytes[plan.encoder_device] = enc_bytes
            param_bytes[plan.decoder_device] = dec_bytes
        param_bytes[plan.backbone_device] = bb_bytes

    reset_peak_stats(device_names)
    timing, _ = time_pipeline_forward_stages(
        model,
        batch,
        warmup=warmup,
        repeat=repeat,
        device_names=device_names,
    )
    peak_alloc, peak_reserved = peak_mib_per_device(device_names)

    return PipelineLoadProfile(
        timing=timing,
        peak_allocated_mib=peak_alloc,
        peak_reserved_mib=peak_reserved,
        after_load_allocated_mib=after_load,
        param_bytes=param_bytes,
        plan_estimated_per_device_gib=plan.estimated_per_device_gib,
        plan_estimated_peak_gib=plan.estimated_peak_gib,
    )
