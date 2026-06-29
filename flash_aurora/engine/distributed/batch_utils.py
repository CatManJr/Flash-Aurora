from __future__ import annotations

import dataclasses

import torch
from flash_aurora.aurora import Batch


def batch_to_device(batch: Batch, device: torch.device | str) -> Batch:
    """Move all batch tensors to ``device`` (metadata stays on CPU)."""
    dev = torch.device(device)
    return dataclasses.replace(
        batch,
        surf_vars={name: tensor.to(dev, non_blocking=True) for name, tensor in batch.surf_vars.items()},
        atmos_vars={name: tensor.to(dev, non_blocking=True) for name, tensor in batch.atmos_vars.items()},
        static_vars={name: tensor.to(dev, non_blocking=True) for name, tensor in batch.static_vars.items()},
    )
