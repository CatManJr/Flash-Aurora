"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Fused MLP FFN for inference: ``fc1 → GELU (exact erf) → fc2``.

Shared by :class:`~aurora.model.perceiver.MLP` and :class:`~aurora.model.swin3d.MLP`.
Phase A uses cuBLAS ``F.linear`` plus :func:`~aurora.ops.triton_gelu.gelu_forward_triton_exact`.

References:
- flash-attn ``flash_attn/ops/fused_dense.py``, ``flash_attn/ops/triton/linear.py`` (Tri Dao)
  — two-layer MLP / linear+activation fusion layout (not linked at runtime).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)


def fused_mlp_ffn_enabled() -> bool:
    return os.environ.get("AURORA_FUSED_MLP_FFN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def mlp_ffn_forward(
    x: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor | None,
    weight2: torch.Tensor,
    bias2: torch.Tensor | None,
) -> torch.Tensor:
    """``fc1 → GELU (exact erf) → fc2`` for inference (dropout applied outside)."""
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("mlp_ffn_forward requires CUDA float32/bfloat16.")
    from aurora.ops.triton_gelu import gelu_forward_triton_exact

    lead = x.shape[:-1]
    in_dim = x.shape[-1]
    x2 = x.reshape(-1, in_dim)
    hidden = F.linear(x2, weight1, bias1)
    hidden = gelu_forward_triton_exact(hidden)
    out = F.linear(hidden, weight2, bias2)
    return out.reshape(*lead, out.shape[-1])


def mlp_ffn_forward_eager(
    x: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor | None,
    weight2: torch.Tensor,
    bias2: torch.Tensor | None,
) -> torch.Tensor:
    """Reference eager path (same numerics as ``nn.Sequential(Linear, GELU, Linear)``)."""
    lead = x.shape[:-1]
    in_dim = x.shape[-1]
    x2 = x.reshape(-1, in_dim)
    h = F.gelu(F.linear(x2, weight1, bias1), approximate="none")
    out = F.linear(h, weight2, bias2)
    return out.reshape(*lead, out.shape[-1])
