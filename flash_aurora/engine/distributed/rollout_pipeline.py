"""Staged pipeline forward for distributed inference."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import torch
from flash_aurora.aurora.batch import Batch
from flash_aurora.aurora.rollout import advance_rollout_batch, prepare_rollout_batch

from flash_aurora.engine.distributed.batch_utils import batch_to_device
from flash_aurora.engine.runtime.cuda_memory import (
    release_batch_gpu_storage,
    release_tensor_storage,
    trim_cuda_cache,
)
from flash_aurora.engine.distributed.pipeline import parallel_plan

if TYPE_CHECKING:
    from flash_aurora.aurora.model.aurora import Aurora


def run_encoder_stage(model: Aurora, batch: Batch) -> tuple[Batch, torch.Tensor, tuple[int, int, int]]:
    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("model is not pipeline-parallel")

    enc_dev = torch.device(plan.encoder_device)
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

    batch = batch_to_device(batch, enc_dev)
    batch, transformed_batch, patch_res = model._prepare_encoder_batch(batch)
    with torch.inference_mode():
        with _device_context(enc_dev):
            x = run_with_encoder_decoder_routing(
                model.encoder,
                transformed_batch,
                autocast_bf16=model.autocast_encoder_decoder,
                use_tensor_core=model.encoder_decoder_use_tensor_core,
                lead_time=model.timestep,
            )
    return batch, x, patch_res


def run_backbone_stage(
    model: Aurora,
    x: torch.Tensor,
    batch: Batch,
    patch_res: tuple[int, int, int],
) -> torch.Tensor:
    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("model is not pipeline-parallel")

    enc_dev = torch.device(plan.encoder_device)
    bb_dev = torch.device(plan.backbone_device)
    if bb_dev != enc_dev:
        x = x.to(bb_dev, non_blocking=True)
    with torch.inference_mode():
        with _device_context(bb_dev):
            out = model._run_backbone(
                x,
                lead_time=model.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
    if bb_dev != enc_dev and x.is_cuda:
        release_tensor_storage(x)
    return out


def run_decoder_stage(
    model: Aurora,
    x: torch.Tensor,
    batch: Batch,
    patch_res: tuple[int, int, int],
) -> Batch:
    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("model is not pipeline-parallel")

    enc_dev = torch.device(plan.encoder_device)
    bb_dev = torch.device(plan.backbone_device)
    dec_dev = torch.device(plan.decoder_device)

    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

    with torch.inference_mode():
        if plan.decoder_spatial_parallel:
            from flash_aurora.engine.distributed.decoder_spatial import forward_decoder_spatial_parallel

            west_dev = torch.device(plan.decoder_spatial_devices[0])
            east_dev = torch.device(plan.decoder_spatial_devices[1])
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
        else:
            if dec_dev != bb_dev:
                x = x.to(dec_dev, non_blocking=True)
            if dec_dev != enc_dev:
                batch = batch_to_device(batch, dec_dev)
            with _device_context(dec_dev):
                pred = run_with_encoder_decoder_routing(
                    model.decoder,
                    x,
                    batch,
                    autocast_bf16=model.autocast_encoder_decoder,
                    use_tensor_core=model.encoder_decoder_use_tensor_core,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                )

    pred = model._finish_prediction(batch, pred)
    result = batch_to_device(pred, enc_dev)
    if bb_dev != enc_dev and x.is_cuda:
        release_tensor_storage(x)
        trim_cuda_cache(bb_dev)
    return result


def _device_context(device: torch.device):
    import contextlib

    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def distributed_rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    *,
    on_step_export: Callable[[int, Batch], None] | None = None,
) -> Generator[Batch, None, None]:
    """Roll out through encoder, backbone, and decoder pipeline stages."""
    plan = parallel_plan(model)
    if plan is None:
        raise RuntimeError("distributed_rollout requires a pipeline-parallel model")

    enc_dev = torch.device(plan.encoder_device)
    batch = prepare_rollout_batch(model, batch)

    for step in range(steps):
        step_batch, x_enc, patch_res = run_encoder_stage(model, batch)
        x_bb = run_backbone_stage(model, x_enc, step_batch, patch_res)
        if x_enc.is_cuda:
            release_tensor_storage(x_enc)

        pred = run_decoder_stage(model, x_bb, step_batch, patch_res)
        if x_bb.is_cuda:
            release_tensor_storage(x_bb)
        yield pred

        if on_step_export is not None:
            on_step_export(step, pred)

        if step + 1 < steps:
            batch = advance_rollout_batch(batch, pred)

    trim_cuda_cache(enc_dev)
