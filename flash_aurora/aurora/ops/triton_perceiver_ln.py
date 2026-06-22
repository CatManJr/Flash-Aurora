"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Triton fused LayerNorm (affine) + optional residual add over the last dimension.

Matches ``nn.LayerNorm(D)`` with ``elementwise_affine=True`` for tensors ``(B, L, D)``.
Fuses ``layer_norm(x) + residual`` for :class:`~aurora.model.perceiver.PerceiverResampler``.

References:
- flash-attn ``flash_attn/ops/triton/layer_norm.py`` - row-wise Triton norm patterns (Tri Dao).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_D = 4096
_BLOCK_D = 128


def layernorm_affine_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """LayerNorm only (same math as ``nn.LayerNorm`` on the last dimension)."""
    return _dispatch(x, None, weight, bias, float(eps), add_residual=False)


def layernorm_affine_add_residual_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``layer_norm(x) + residual``, shapes ``x`` and ``residual`` ``(B, L, D)``."""
    if residual.shape != x.shape:
        raise ValueError("residual must match x shape.")
    return _dispatch(x, residual, weight, bias, float(eps), add_residual=True)


def _dispatch(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    add_residual: bool,
) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError("triton_perceiver_ln ops require CUDA tensors.")
    if x.dim() != 3:
        raise ValueError(f"Expected x (B, L, D), got shape {tuple(x.shape)}.")
    B, L, D = x.shape
    if D > _MAX_D:
        raise ValueError(f"D must be <= {_MAX_D}, got {D}.")
    if weight.shape != (D,) or bias.shape != (D,):
        raise ValueError("weight and bias must be shape (D,).")

    out = torch.empty_like(x)
    grid = (B * L,)
    # Dummy pointer when not adding residual (never loaded).
    res_ptr = residual if add_residual else x

    kind = _dtype_kind(x.dtype)
    _ln_affine_residual_kernel[grid](
        x,
        out,
        weight,
        bias,
        res_ptr,
        L=L,
        D=D,
        BLOCK_D=_BLOCK_D,
        EPS=eps,
        ADD_RESIDUAL=add_residual,
        DTYPE_KIND=kind,
    )
    return out


def _dtype_kind(dt: torch.dtype) -> int:
    if dt == torch.bfloat16:
        return 0
    if dt == torch.float16:
        return 1
    if dt == torch.float32:
        return 2
    raise ValueError(f"Unsupported dtype {dt}; use float16, bfloat16, or float32.")


@triton.jit
def _ln_affine_residual_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    b_ptr,
    residual_ptr,
    L,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    DTYPE_KIND: tl.constexpr,
):
    row = tl.program_id(0)
    inv_d = 1.0 / D

    sum_x = tl.full((), 0.0, dtype=tl.float32)
    sum_x2 = tl.full((), 0.0, dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        xv = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(xv)
        sum_x2 += tl.sum(xv * xv)

    mean = sum_x * inv_d
    var = sum_x2 * inv_d - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        xv = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        yn = (xv - mean) * inv_std
        y = yn * wv + bv
        if ADD_RESIDUAL:
            rv = tl.load(residual_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            y = y + rv
        if DTYPE_KIND == 0:
            tl.store(out_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)
        elif DTYPE_KIND == 1:
            tl.store(out_ptr + row * D + offs, y.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + row * D + offs, y, mask=mask)
