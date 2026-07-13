"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Triton GELU (tanh approx) for legacy ``use_triton_gelu`` Swin3D MLP path only.
Decoder/backbone MLP fusion is planned under :mod:`aurora.ops.cute.mlp_ffn` (CuTe / cuDNN).

References:
- flash-attn ``flash_attn/ops/triton/k_activations.py`` ``gelu_approx`` (Tri Dao; BSD-3 via xFormers).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_TRITON_DTYPE = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}


def gelu_forward_triton(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation (legacy ``use_triton_gelu`` path only)."""
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("gelu_forward_triton requires CUDA float32/bfloat16 input.")
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_tanh_kernel[grid](x, out, n, BLOCK=BLOCK, DTYPE=_TRITON_DTYPE[x.dtype])
    return out


# Tanh-approx GELU - same formulation as flash_attn k_activations.py::gelu_approx
@triton.jit
def _gelu_tanh_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offs, y.to(DTYPE), mask=mask)
