"""Shared batch prep and forward warmup for rollout (matches benchmark semantics)."""

from __future__ import annotations

import dataclasses

import torch
from flash_aurora.models.aurora import Batch
from flash_aurora.models.aurora.model.aurora import Aurora


def prepare_rollout_batch(model: Aurora, batch: Batch) -> Batch:
    batch = model.batch_transform_hook(batch)
    param = next(model.parameters())
    batch = batch.type(param.dtype)
    batch = batch.crop(model.patch_size)
    return batch.to(param.device)


def advance_rollout_batch(batch: Batch, pred: Batch) -> Batch:
    return dataclasses.replace(
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


def _forward_for_warmup(model: Aurora, batch: Batch) -> Batch:
    """One forward for warmup; Aurora 1.5 requires an explicit ``lead_times`` tensor."""
    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout

    if not model_uses_v1p5_rollout(model):
        return model.forward(batch)

    example = next(iter(batch.surf_vars.values()))
    lead_hours = model.timestep.total_seconds() / 3600.0
    lead_times = torch.full(
        (example.shape[0],),
        lead_hours,
        device=example.device,
        dtype=example.dtype,
    )
    return model.forward(batch, lead_times=lead_times)


def warmup_forwards(
    model: Aurora,
    batch: Batch,
    *,
    iters: int,
    device: torch.device,
) -> Batch:
    """Run ``iters`` untimed forwards to JIT-compile CuTe kernels on custom-precision paths."""
    if iters <= 0:
        return batch

    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout

    uses_v1p5 = model_uses_v1p5_rollout(model)
    with torch.inference_mode():
        for _ in range(iters):
            pred = _forward_for_warmup(model, batch)
            # V1p5 predictions include output-only vars absent from the IC; keep the same
            # prepared batch for JIT warmup instead of sliding the history window.
            if not uses_v1p5:
                batch = advance_rollout_batch(batch, pred)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    return batch
