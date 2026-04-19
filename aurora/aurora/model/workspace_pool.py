"""Copyright (c) Catman Jr. Licensed under the MIT license.

Reusable scratch buffers for fixed-shape inference (Stage D3): avoids repeated
``empty`` allocations on hot paths when shapes are stable across forwards.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["InferenceWorkspacePool"]


def _normalize_device(device: torch.device) -> torch.device:
    """Make ``cuda`` and ``cuda:0`` comparable (PyTorch treats them as unequal devices)."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return d


class InferenceWorkspacePool:
    """Keyed buffers reused when ``(key, shape, device, dtype)`` match."""

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        key: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a tensor of ``shape`` on ``device`` with ``dtype``, reusing storage when possible."""
        device = _normalize_device(device)
        t: Optional[torch.Tensor] = self._buffers.get(key)
        if (
            t is None
            or t.shape != shape
            or t.dtype != dtype
            or t.device != device
        ):
            self._buffers[key] = torch.empty(shape, device=device, dtype=dtype)
            t = self._buffers[key]
        return t

    def clear(self) -> None:
        """Drop all buffers (e.g. between tests or after device migration)."""
        self._buffers.clear()
