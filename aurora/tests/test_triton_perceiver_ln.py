"""Copyright (c) Catman Jr. Licensed under the MIT license."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_affine_matches_module(dtype: torch.dtype) -> None:
    pytest.importorskip("triton")
    from aurora.ops.triton_perceiver_ln import layernorm_affine_forward

    B, L, D = 2, 31, 256
    x = torch.randn(B, L, D, device="cuda", dtype=dtype)
    ln = nn.LayerNorm(D).to(device="cuda", dtype=dtype)
    ref = ln(x)
    got = layernorm_affine_forward(x, ln.weight, ln.bias, float(ln.eps))
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(ref, got, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_affine_add_residual_matches_torch(dtype: torch.dtype) -> None:
    pytest.importorskip("triton")
    from aurora.ops.triton_perceiver_ln import layernorm_affine_add_residual_forward

    B, L, D = 2, 31, 256
    x = torch.randn(B, L, D, device="cuda", dtype=dtype)
    residual = torch.randn(B, L, D, device="cuda", dtype=dtype)
    ln = nn.LayerNorm(D).to(device="cuda", dtype=dtype)
    ref = ln(x) + residual
    got = layernorm_affine_add_residual_forward(x, residual, ln.weight, ln.bias, float(ln.eps))
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(ref, got, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_perceiver_resampler_fused_matches_eager_bf16() -> None:
    pytest.importorskip("triton")
    from aurora.model.perceiver import PerceiverResampler

    B, Lq, Lk, D = 2, 8, 64, 128
    eager = PerceiverResampler(
        D, D, depth=2, head_dim=32, num_heads=4, use_flash_attn=False, use_triton_ln_residual_fusion=False
    )
    fused = PerceiverResampler(
        D, D, depth=2, head_dim=32, num_heads=4, use_flash_attn=False, use_triton_ln_residual_fusion=True
    )
    fused.load_state_dict(eager.state_dict())
    # ``nn.Module`` has no ``.bf16()``; cast parameters/buffers explicitly.
    eager = eager.to(device="cuda", dtype=torch.bfloat16).eval()
    fused = fused.to(device="cuda", dtype=torch.bfloat16).eval()
    latents = torch.randn(B, Lq, D, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(B, Lk, D, device="cuda", dtype=torch.bfloat16)
    o_e = eager(latents, context)
    o_f = fused(latents, context)
    # Resampler stacks LN + residual + GELU MLP: bf16 end-to-end needs looser tol than LN-only kernels.
    torch.testing.assert_close(o_e, o_f, rtol=1e-1, atol=2e-1)
