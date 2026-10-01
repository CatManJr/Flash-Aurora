"""Backbone noise of the Aurora 1.5 ensemble under a longitude split (deviation D3).

``Swin3DTransformerBackbone`` draws ``torch.randn(B, L, D_t)`` over all ``C * H * W``
tokens and adds ``noise_mlp`` of it to the conditioning. Independent per-shard draws
would produce a different ensemble member from the single-device one, so member-by-
member comparison would be impossible.

Every rank of a DTP instance therefore draws the global field with the same RNG
state as the single-device backbone and keeps its longitude columns. One member
seed then yields the same member on any mesh, which is what lets uncertainty-
parallel members be scored one by one against single-device members. The cost is
one transient global draw per step per rank; a counter-based generator keyed on the
global token index would avoid it, but PyTorch's generators expose no sub-block of a
``randn`` stream.

Noise accumulation (``set_noise_accumulation``) keeps the single-device FIFO
semantics on the local slices; the accumulated sum is elementwise, so slicing
commutes with it.
"""

from __future__ import annotations

import warnings

import torch


def slice_token_columns(
    tokens: torch.Tensor, resolution: tuple[int, int, int], columns: slice
) -> torch.Tensor:
    """Tokens ``(B, C * H * W, E)`` restricted to longitude ``columns``."""
    levels, height, width = resolution
    batch, _, channels = tokens.shape
    grid = tokens.view(batch, levels, height, width, channels)
    return grid[:, :, :, columns].reshape(batch, -1, channels)


class ShardedNoiseField:
    """Per-token noise sliced from the single-device draw, with optional FIFO accumulation."""

    def __init__(self) -> None:
        self._cache: list[torch.Tensor] = []
        self._cache_size = 0
        self._accumulate = False

    def reset(self) -> None:
        self._cache.clear()

    def set_accumulation(self, n: int) -> None:
        self._cache.clear()
        self._cache_size = max(n, 0)
        self._accumulate = self._cache_size > 0

    def draw(
        self,
        *,
        batch_size: int,
        resolution: tuple[int, int, int],
        embed_dim: int,
        columns: slice,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Effective noise of this step on this rank's columns, ``(B, C * H * W_r, E)``."""
        levels, height, width = resolution
        global_shape = (batch_size, levels * height * width, embed_dim)

        def draw_local() -> torch.Tensor:
            noise = torch.randn(global_shape, device=device, dtype=dtype)
            return slice_token_columns(noise, resolution, columns)

        noise = draw_local()
        if not self._accumulate:
            return noise
        if self._cache and self._cache[0].shape != noise.shape:
            warnings.warn(
                f"Noise shape changed from {self._cache[0].shape} to {noise.shape}; clearing noise cache.",
                stacklevel=2,
            )
            self._cache.clear()
        if len(self._cache) >= self._cache_size:
            self._cache.pop(0)
        self._cache.append(noise)
        while len(self._cache) < self._cache_size:
            self._cache.append(draw_local())
        return torch.stack(self._cache).sum(dim=0) / (self._cache_size**0.5)
