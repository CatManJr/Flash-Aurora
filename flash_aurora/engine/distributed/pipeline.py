from __future__ import annotations

import contextlib
import types
from typing import Any

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.distributed.config import ParallelPlan

_DISTRIBUTED_PLAN_ATTR = "_flash_aurora_parallel_plan"
_ORIGINAL_FORWARD_ATTR = "_flash_aurora_original_forward"


def is_pipeline_parallel(model: Aurora) -> bool:
    return getattr(model, _DISTRIBUTED_PLAN_ATTR, None) is not None


def parallel_plan(model: Aurora) -> ParallelPlan | None:
    return getattr(model, _DISTRIBUTED_PLAN_ATTR, None)


def pipeline_forward(model: Aurora, batch: Batch) -> Batch:
    """Run Aurora forward with encoder / backbone / decoder on separate devices."""
    from flash_aurora.engine.distributed.rollout_pipeline import (
        run_backbone_stage,
        run_decoder_stage,
        run_encoder_stage,
    )

    with torch.inference_mode():
        step_batch, x, patch_res = run_encoder_stage(model, batch)
        x = run_backbone_stage(model, x, step_batch, patch_res)
        return run_decoder_stage(model, x, step_batch, patch_res)


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
    from flash_aurora.engine.distributed.decoder_spatial import apply_decoder_spatial_placement

    apply_decoder_spatial_placement(
        model,
        decoder_device=plan.decoder_device,
        decoder_spatial_parallel=plan.decoder_spatial_parallel,
        decoder_spatial_devices=plan.decoder_spatial_devices,
    )

    setattr(model, _DISTRIBUTED_PLAN_ATTR, plan)
    setattr(model, _ORIGINAL_FORWARD_ATTR, model.forward)
    model.forward = types.MethodType(pipeline_forward, model)  # type: ignore[method-assign]
    return model


def restore_pipeline_parallel(model: Aurora) -> None:
    """Undo pipeline patching and move all submodules to the encoder device."""
    from flash_aurora.engine.distributed.decoder_spatial import clear_decoder_spatial_replica

    original = getattr(model, _ORIGINAL_FORWARD_ATTR, None)
    if original is not None:
        model.forward = original  # type: ignore[method-assign]
        delattr(model, _ORIGINAL_FORWARD_ATTR)

    plan: ParallelPlan | None = getattr(model, _DISTRIBUTED_PLAN_ATTR, None)
    if plan is None:
        return

    clear_decoder_spatial_replica(model)
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
        "decoder_spatial_parallel": plan.decoder_spatial_parallel,
        "decoder_spatial_devices": plan.decoder_spatial_devices,
        "estimated_peak_gib": plan.estimated_peak_gib,
        "estimated_per_device_gib": plan.estimated_per_device_gib,
    }
