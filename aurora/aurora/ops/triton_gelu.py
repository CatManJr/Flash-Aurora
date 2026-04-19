"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Triton GELU forward for Swin3D MLP inference path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def gelu_forward_triton(x: torch.Tensor) -> torch.Tensor:
    """Compute GELU(x) with tanh approximation on CUDA float32."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("gelu_forward_triton requires CUDA float32 input.")
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offs, y, mask=mask)
