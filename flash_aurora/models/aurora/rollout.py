"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator

import torch

from flash_aurora.aurora.batch import Batch
from flash_aurora.aurora.model.aurora import Aurora

__all__ = ["advance_rollout_batch", "prepare_rollout_batch", "rollout"]


def prepare_rollout_batch(model: Aurora, batch: Batch) -> Batch:
    """Apply rollout hooks and move ``batch`` to the model device."""
    batch = model.batch_transform_hook(batch)
    param = next(model.parameters())
    return batch.type(param.dtype).crop(model.patch_size).to(param.device)


def advance_rollout_batch(batch: Batch, pred: Batch) -> Batch:
    """Slide history window forward using the latest prediction."""
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


def rollout(model: Aurora, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.Batch`: The prediction after every step.
    """
    batch = prepare_rollout_batch(model, batch)

    for _ in range(steps):
        pred = model.forward(batch)
        yield pred
        batch = advance_rollout_batch(batch, pred)
