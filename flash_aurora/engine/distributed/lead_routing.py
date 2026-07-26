"""Lead-time helpers for pipeline-parallel Aurora (legacy timedelta vs v1p5 tensor)."""

from __future__ import annotations

from typing import Any

import torch
from flash_aurora.models.aurora.batch import Batch


def uses_variable_lead_times(model: Any) -> bool:
    return bool(getattr(model, "variable_lead_time", False))


def resolve_lead_times_tensor(
    model: Any,
    batch: Batch,
    lead_times: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return a ``(B,)`` hours tensor for Aurora 1.5; ``None`` for the legacy family."""
    if not uses_variable_lead_times(model):
        return None

    example = next(iter(batch.surf_vars.values()))
    batch_size = int(example.shape[0])
    device = example.device
    dtype = example.dtype

    if lead_times is None:
        lead_hours = model.timestep.total_seconds() / 3600.0
        return torch.full((batch_size,), lead_hours, device=device, dtype=dtype)

    resolved = lead_times.to(device=device, dtype=dtype)
    if resolved.ndim == 0:
        return resolved.expand(batch_size)
    if int(resolved.shape[0]) != batch_size:
        raise ValueError(
            f"`lead_times` batch size {resolved.shape[0]} does not match batch {batch_size}"
        )
    return resolved


def encoder_decoder_lead_kwargs(
    model: Any,
    lead_times: torch.Tensor | None,
) -> dict[str, Any]:
    if uses_variable_lead_times(model):
        if lead_times is None:
            raise ValueError("Aurora 1.5 pipeline stages require a lead_times tensor")
        return {"lead_times": lead_times}
    return {"lead_time": model.timestep}


def backbone_lead_kwargs(
    model: Any,
    lead_times: torch.Tensor | None,
) -> dict[str, Any]:
    return encoder_decoder_lead_kwargs(model, lead_times)
