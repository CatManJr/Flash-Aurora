from __future__ import annotations

import contextlib
import types
from typing import Any

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.distributed.batch_utils import batch_to_device
from flash_aurora.engine.distributed.config import ParallelPlan

_DISTRIBUTED_PLAN_ATTR = "_flash_aurora_parallel_plan"
_ORIGINAL_FORWARD_ATTR = "_flash_aurora_original_forward"


def is_pipeline_parallel(model: Aurora) -> bool:
    return getattr(model, _DISTRIBUTED_PLAN_ATTR, None) is not None


def parallel_plan(model: Aurora) -> ParallelPlan | None:
    return getattr(model, _DISTRIBUTED_PLAN_ATTR, None)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def pipeline_forward(model: Aurora, batch: Batch) -> Batch:
    """Run Aurora forward with encoder / backbone / decoder on separate devices."""
    plan: ParallelPlan = getattr(model, _DISTRIBUTED_PLAN_ATTR)
    enc_dev = torch.device(plan.encoder_device)
    bb_dev = torch.device(plan.backbone_device)
    dec_dev = torch.device(plan.decoder_device)

    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing
    from flash_aurora.aurora.model.nvtx import nvtx_range

    batch = batch_to_device(batch, enc_dev)
    batch, transformed_batch, patch_res = model._prepare_encoder_batch(batch)

    with _device_context(enc_dev):
        with nvtx_range("aurora::encoder"):
            with torch.inference_mode():
                x = run_with_encoder_decoder_routing(
                    model.encoder,
                    transformed_batch,
                    autocast_bf16=model.autocast_encoder_decoder,
                    use_tensor_core=model.encoder_decoder_use_tensor_core,
                    lead_time=model.timestep,
                )

    if bb_dev != enc_dev:
        x = x.to(bb_dev, non_blocking=True)
        _maybe_sync(bb_dev)

    with _device_context(bb_dev):
        with nvtx_range("aurora::backbone"):
            with torch.inference_mode():
                x = model._run_backbone(
                    x,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                    rollout_step=batch.metadata.rollout_step,
                )

    if dec_dev != bb_dev:
        x = x.to(dec_dev, non_blocking=True)
    if dec_dev != enc_dev:
        batch = batch_to_device(batch, dec_dev)
    if dec_dev != bb_dev:
        _maybe_sync(dec_dev)

    with _device_context(dec_dev):
        with nvtx_range("aurora::decoder"):
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

    pred = model._finish_prediction(batch, pred)
    return batch_to_device(pred, enc_dev)


def apply_pipeline_parallel(model: Aurora, plan: ParallelPlan) -> Aurora:
    """Place Aurora submodules on pipeline devices and patch ``forward``."""
    if is_pipeline_parallel(model):
        existing: ParallelPlan = getattr(model, _DISTRIBUTED_PLAN_ATTR)
        if existing == plan:
            return model
        restore_pipeline_parallel(model)

    if hasattr(model, "clear_inference_cuda_graph"):
        model.clear_inference_cuda_graph()

    model.encoder.to(plan.encoder_device)
    model.backbone.to(plan.backbone_device)
    model.decoder.to(plan.decoder_device)

    setattr(model, _DISTRIBUTED_PLAN_ATTR, plan)
    setattr(model, _ORIGINAL_FORWARD_ATTR, model.forward)
    model.forward = types.MethodType(pipeline_forward, model)  # type: ignore[method-assign]
    return model


def restore_pipeline_parallel(model: Aurora) -> None:
    """Undo pipeline patching and move all submodules to the encoder device."""
    original = getattr(model, _ORIGINAL_FORWARD_ATTR, None)
    if original is not None:
        model.forward = original  # type: ignore[method-assign]
        delattr(model, _ORIGINAL_FORWARD_ATTR)

    plan: ParallelPlan | None = getattr(model, _DISTRIBUTED_PLAN_ATTR, None)
    if plan is None:
        return

    target = plan.encoder_device
    model.to(target)
    delattr(model, _DISTRIBUTED_PLAN_ATTR)


def distributed_status(model: Aurora) -> dict[str, Any]:
    plan = parallel_plan(model)
    if plan is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "devices": plan.devices,
        "encoder_device": plan.encoder_device,
        "backbone_device": plan.backbone_device,
        "decoder_device": plan.decoder_device,
        "estimated_peak_gib": plan.estimated_peak_gib,
        "estimated_per_device_gib": plan.estimated_per_device_gib,
    }
