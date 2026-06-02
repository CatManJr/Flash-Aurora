"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Fused LayerNorm (no affine) + FiLM modulation for :class:`AdaptiveLayerNorm` inference.
Optional fused residual add: ``out = residual + film(ln(x))`` in one kernel.

The ``output_fp32`` path loads BF16 (or FP32) activations, computes in FP32, and
stores FP32 — avoiding a separate ``tensor.to(float32)`` before the norm boundary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_D = 2048
_BLOCK_D = 128


_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def adaptive_layernorm_film_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_bias: float,
    eps: float = 1e-5,
    *,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Match ``AdaptiveLayerNorm.forward`` when ``ln`` has ``elementwise_affine=False``.

    Args:
        x: ``(B, L, D)`` float32 or bfloat16 CUDA.
        scale, shift: ``(B, 1, D)`` CUDA (typically same dtype as ``x``).
        scale_bias: Added to ``scale`` before multiply (same as module).
        eps: LayerNorm epsilon.
        output_fp32: If ``True``, write FP32 outputs (LN math still in FP32).

    Returns:
        ``(B, L, D)`` tensor.
    """
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("adaptive_layernorm_film_forward requires CUDA float32/bfloat16 x.")
    B, L, D = x.shape
    if D > _MAX_D:
        raise ValueError(f"adaptive_layernorm_film_forward supports D <= {_MAX_D} for now.")
    assert scale.shape == (B, 1, D) and shift.shape == (B, 1, D)
    out_dtype = torch.float32 if output_fp32 else x.dtype
    out = torch.empty(B, L, D, device=x.device, dtype=out_dtype)
    grid = (B * L,)
    _adaln_film_blocked_kernel[grid](
        x,
        out,
        shift,
        scale,
        x,
        L=L,
        D=D,
        BLOCK_D=_BLOCK_D,
        scale_bias=float(scale_bias),
        eps=float(eps),
        WITH_RESIDUAL=False,
        X_DTYPE=_TRITON_DTYPE[x.dtype],
        RES_DTYPE=_TRITON_DTYPE[x.dtype],
        MOD_DTYPE=_TRITON_DTYPE[scale.dtype],
        STORE_FP32=output_fp32,
    )
    return out


def adaptive_layernorm_film_add_residual_forward(
    residual: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_bias: float,
    eps: float = 1e-5,
    *,
    output_fp32: bool = False,
) -> torch.Tensor:
    """``residual + film(ln(x))`` with the same math as :func:`adaptive_layernorm_film_forward`.

    When ``output_fp32=True``, ``residual`` must be FP32 and ``x`` may be BF16/FP32; the
    result is FP32 (norm boundary for ``bf16_mixed`` backbone without a pre-kernel copy).

    Args:
        residual: ``(B, L, D)`` float32 or bfloat16 CUDA (must not alias ``x``).
        x: ``(B, L, D)`` CUDA, input to LayerNorm + FiLM.
        scale, shift: ``(B, 1, D)`` CUDA.
        scale_bias: Added to ``scale`` before multiply.
        eps: LayerNorm epsilon.
        output_fp32: Store FP32 outputs; allows ``residual`` (FP32) and ``x`` (BF16) to differ.

    Returns:
        ``(B, L, D)`` tensor.
    """
    if residual.device.type != "cuda" or x.device.type != "cuda":
        raise ValueError("adaptive_layernorm_film_add_residual_forward requires CUDA tensors.")
    if residual.dtype not in _SUPPORTED_DTYPES or x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("adaptive_layernorm_film_add_residual_forward requires float32/bfloat16.")
    if not output_fp32 and residual.dtype != x.dtype:
        raise ValueError("residual and x must have the same dtype unless output_fp32=True.")
    if output_fp32 and residual.dtype != torch.float32:
        raise ValueError("output_fp32=True requires residual.dtype == float32.")
    if residual.shape != x.shape:
        raise ValueError("residual and x must have the same shape.")
    B, L, D = x.shape
    if D > _MAX_D:
        raise ValueError(
            f"adaptive_layernorm_film_add_residual_forward supports D <= {_MAX_D}."
        )
    assert scale.shape == (B, 1, D) and shift.shape == (B, 1, D)
    out_dtype = torch.float32 if output_fp32 else x.dtype
    out = torch.empty(B, L, D, device=x.device, dtype=out_dtype)
    grid = (B * L,)
    _adaln_film_blocked_kernel[grid](
        x,
        out,
        shift,
        scale,
        residual,
        L=L,
        D=D,
        BLOCK_D=_BLOCK_D,
        scale_bias=float(scale_bias),
        eps=float(eps),
        WITH_RESIDUAL=True,
        X_DTYPE=_TRITON_DTYPE[x.dtype],
        RES_DTYPE=_TRITON_DTYPE[residual.dtype],
        MOD_DTYPE=_TRITON_DTYPE[scale.dtype],
        STORE_FP32=output_fp32,
    )
    return out


@triton.jit
def _adaln_film_blocked_kernel(
    x_ptr,
    out_ptr,
    shift_ptr,
    scale_ptr,
    residual_ptr,
    L,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    scale_bias: tl.constexpr,
    eps: tl.constexpr,
    WITH_RESIDUAL: tl.constexpr,
    X_DTYPE: tl.constexpr,
    RES_DTYPE: tl.constexpr,
    MOD_DTYPE: tl.constexpr,
    STORE_FP32: tl.constexpr,
):
    """Per (B,L) row: LN + FiLM; optionally add residual from ``residual_ptr``.

    Loads ``x`` / residual / modulation in ``X_DTYPE`` / ``RES_DTYPE`` / ``MOD_DTYPE``;
    all reductions and arithmetic are in FP32. Stores FP32 when ``STORE_FP32``, else
    rounds to ``X_DTYPE``.
    """
    row = tl.program_id(0)
    b = row // L
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
    inv_std = 1.0 / tl.sqrt(var + eps)

    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        xv = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + b * D + offs, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + b * D + offs, mask=mask, other=0.0).to(tl.float32)
        yn = (xv - mean) * inv_std
        y = yn * (scale_bias + sc) + sh
        if WITH_RESIDUAL:
            rv = tl.load(residual_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            y = y + rv
        if STORE_FP32:
            tl.store(out_ptr + row * D + offs, y, mask=mask)
        else:
            tl.store(out_ptr + row * D + offs, y.to(X_DTYPE), mask=mask)
