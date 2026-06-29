"""Shared batch prep and forward warmup for rollout (matches benchmark semantics)."""

from __future__ import annotations

import dataclasses

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora


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
    with torch.inference_mode():
        for _ in range(iters):
            pred = model.forward(batch)
            batch = advance_rollout_batch(batch, pred)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    return batch
