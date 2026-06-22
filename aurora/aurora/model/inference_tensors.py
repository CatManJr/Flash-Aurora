"""Copyright (c) Catman Jr. Licensed under the MIT license.

Cached constant GPU tensors for fixed-shape inference (incl. CUDA graph replay).

Python scalars / tuples (pressure levels, timestamps, etc.) must not be converted with
``torch.tensor(..., device=cuda)`` inside a captured graph — that performs an implicit
CPU→GPU copy.  Populate the cache during eager warmup; replay then reads existing tensors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import torch

# (values, device, dtype) -> tensor
_CONSTANT_CACHE: dict[tuple, torch.Tensor] = {}


def cached_constant_tensor(
    values: Sequence[float | int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialise ``values`` once on ``device`` and reuse the same tensor."""
    dev = torch.device(device)
    key = (tuple(float(v) for v in values), dev.type, dev.index, dtype)
    cached = _CONSTANT_CACHE.get(key)
    if cached is not None and cached.device == dev and cached.dtype == dtype:
        return cached
    tensor = torch.tensor(list(values), device=dev, dtype=dtype)
    _CONSTANT_CACHE[key] = tensor
    return tensor


def cached_absolute_hours_tensor(
    times: tuple[datetime, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Absolute timestamps as hours since epoch, cached on ``device``."""
    hours = tuple(t.timestamp() / 3600.0 for t in times)
    return cached_constant_tensor(hours, device=device, dtype=dtype)


def clear_constant_tensor_cache() -> None:
    """Drop cached tensors (tests only)."""
    _CONSTANT_CACHE.clear()
