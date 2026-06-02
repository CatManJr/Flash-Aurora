"""Pytest coverage for aurora.ops.cute window attention.

Test matrix
-----------
TF32_ACC_FP32  – stable FP32-logits path (TF32 matmul allowed) vs strict-FP32 SDPA.
BF16_MIXED     – stable path vs FP32 SDPA reference.
window_attn_dispatch – CUDA stable path or SDPA (strict / use_cute=False / CPU).

Shape coverage
--------------
N = 144  aurora-pretrain-small (window 2×6×12)
N = 288  2×6×24 or 4×6×12  (double spatial resolution)
N = 576  2×12×24            (quadruple spatial resolution)
Dh = 64  uniform across all Aurora encoder heads

CuTeDSL GEMM tests: set ``AURORA_CUTE_WINDOW_ATTN=1`` (optional); most tests need
only CUDA (``requires_cuda``).
"""
from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from aurora.ops.cute.window_attn_fwd import (
    _CUTE_AVAILABLE,
    _attention_window_stable,
    _expand_bias_for_sdpa,
    _choose_tile_n,
    _choose_tile_n_tf32,
    _tf32_hybrid_smem_bytes,
    _get_smem_budget_bytes,
    WinAttnPrecision,
    window_attn_dispatch,
    window_attn_fwd_cute,
    window_attn_fwd_cute_qkvpacked,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
requires_cute = pytest.mark.skipif(
    not torch.cuda.is_available() or not _CUTE_AVAILABLE,
    reason="CUDA + CuTeDSL (cutlass + quack) required",
)
requires_cute_env = pytest.mark.skipif(
    not torch.cuda.is_available() or not _CUTE_AVAILABLE,
    reason="CUDA + CuTeDSL + AURORA_CUTE_WINDOW_ATTN=1",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qkv(
    Bwin: int,
    H: int,
    N: int,
    Dh: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 42,
    *,
    activation_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (q, k, v); default scale is small, use larger for regression tests."""
    g = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(dtype=dtype, device=device, generator=g)
    q = torch.randn(Bwin, H, N, Dh, **kwargs) * activation_scale
    k = torch.randn(Bwin, H, N, Dh, **kwargs) * activation_scale
    v = torch.randn(Bwin, H, N, Dh, **kwargs) * activation_scale
    return q, k, v


def _make_bias(nW: int, N: int, device: str, seed: int = 7) -> torch.Tensor:
    """(nW, N, N) float32 bias simulating a shifted-window mask (-100 like Swin)."""
    bias = torch.zeros(nW, N, N, dtype=torch.float32, device=device)
    bias[:, N // 2 :, : N // 2] = -100.0
    return bias


def _fp32_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: float,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Strict-FP32 or TF32 SDPA reference (inputs cast to float32)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    Bwin, H, N, _ = qf.shape
    attn_mask = _expand_bias_for_sdpa(bias, Bwin, H, N) if bias is not None else None

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=attn_mask, scale=scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    return out


# ---------------------------------------------------------------------------
# Aurora-relevant parametrised shapes
# ---------------------------------------------------------------------------

# (Bwin, H, N, Dh, nW)
#
# N=144 : aurora-pretrain-small (window_size 2×6×12)
# N=288 : 2×6×24 or 4×6×12  — single-pass for BF16 on 99 KB SMEM
# N=576 : 2×12×24            — 2-pass streaming for BF16 on 99 KB SMEM
AURORA_SHAPES = [
    (16,  8, 144, 64, 4),   # encoder stage H=8,  N=144 (small model default)
    ( 8, 16, 144, 64, 4),   # encoder stage H=16, N=144
    ( 4, 32, 144, 64, 4),   # encoder stage H=32, N=144
    ( 8,  8, 288, 64, 4),   # 2× spatial res, H=8  — tests larger-N single-pass (BF16)
    ( 4, 16, 288, 64, 4),   # 2× spatial res, H=16
    ( 2, 32, 576, 64, 4),   # 4× spatial res, H=32 — tests streaming (2 KV passes)
]

EXTRA_SHAPES = [
    ( 4,  8,  64, 64, 2),   # smaller N (sub-tile)
    ( 6,  4,  96, 64, 3),   # non-power-of-2 N
    ( 4,  8, 256, 64, 2),   # N=256 — 2×8×16 window
    ( 2,  8, 400, 64, 2),   # N=400 — 2×10×20 window (streaming for TF32, single-pass BF16)
]


# ===========================================================================
# TF32_ACC_FP32 path  (CuTeDSL TF32 kernel)
# ===========================================================================

@requires_cute
@pytest.mark.parametrize("has_bias", [False, True])
def test_tf32_acc_fp32_close_to_strict_reference(has_bias: bool) -> None:
    """TF32_ACC_FP32 output must be numerically close to strict-FP32 reference.

    TF32 truncates the mantissa to 10 bits during multiply, so results differ
    from exact FP32 by up to ~1e-3 relative error.
    """
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    with torch.no_grad():
        ref = _fp32_sdpa_reference(q, k, v, bias, scale, allow_tf32=False)
        out = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.TF32_ACC_FP32
        )

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@requires_cute
@pytest.mark.parametrize("Bwin,H,N,Dh,nW", AURORA_SHAPES)
def test_tf32_aurora_shapes(Bwin: int, H: int, N: int, Dh: int, nW: int) -> None:
    """TF32 CuTe kernel works for all Aurora stage shapes."""
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias = _make_bias(nW, N, "cuda")
    scale = 1.0 / math.sqrt(Dh)
    with torch.no_grad():
        ref = _fp32_sdpa_reference(q, k, v, bias, scale, allow_tf32=False)
        out = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.TF32_ACC_FP32
        )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@requires_cute
def test_tf32_output_shape_dtype() -> None:
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    with torch.no_grad():
        out = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.TF32_ACC_FP32)
    assert out.shape == (Bwin, H, N, Dh)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Regression: Swin -100 mask + large activations (production-scale logits)
# ---------------------------------------------------------------------------

_LARGE_ACT_SCALES = (1.0, 4.0, 5.0)

# vs torch SDPA (TF32 matmul allowed) — tight at small logits, looser at production scale
_TF32_VS_SDPA_MAX_ABS = {1.0: 0.05, 4.0: 0.16, 5.0: 0.26}
# vs explicit stable softmax — guards mask regression (was O(10) before fix)
_TF32_VS_STABLE_MAX_ABS = {1.0: 0.05, 4.0: 0.16, 5.0: 0.26}
_BF16_VS_STABLE_MAX_ABS = {1.0: 0.02, 4.0: 0.8, 5.0: 1.7}


@requires_cute
@pytest.mark.parametrize("activation_scale", _LARGE_ACT_SCALES)
def test_tf32_swin_mask_large_activations(activation_scale: float) -> None:
    """CuTe TF32 + Swin -100 mask: no O(10) drift vs SDPA/stable at large logits."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(
        Bwin, H, N, Dh, torch.float32, "cuda", activation_scale=activation_scale
    )
    bias = _make_bias(nW, N, "cuda")
    scale = 1.0 / math.sqrt(Dh)
    tol_sdpa = _TF32_VS_SDPA_MAX_ABS[activation_scale]
    tol_stable = _TF32_VS_STABLE_MAX_ABS[activation_scale]

    with torch.no_grad():
        ref = _fp32_sdpa_reference(q, k, v, bias, scale, allow_tf32=True)
        stable = _attention_window_stable(q, k, v, bias, scale, strict_fp32=False)
        out = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.TF32_ACC_FP32
        )

    max_sdpa = (out - ref).abs().max().item()
    max_stable = (out - stable).abs().max().item()
    assert max_sdpa < tol_sdpa, (
        f"vs SDPA activation_scale={activation_scale} max_err={max_sdpa} tol={tol_sdpa}"
    )
    assert max_stable < tol_stable, (
        f"vs stable activation_scale={activation_scale} max_err={max_stable} tol={tol_stable}"
    )


@requires_cute
@pytest.mark.parametrize("activation_scale", _LARGE_ACT_SCALES)
def test_tf32_qkvpacked_swin_mask_large_activations(activation_scale: float) -> None:
    """Production qkvpacked path must match SDPA under large masked logits."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(
        Bwin, H, N, Dh, torch.float32, "cuda", activation_scale=activation_scale
    )
    bias = _make_bias(nW, N, "cuda")
    scale = 1.0 / math.sqrt(Dh)

    qkv = torch.empty(Bwin, N, 3 * H * Dh, device="cuda", dtype=torch.float32)
    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    qkv_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    qkv_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    qkv_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))

    with torch.no_grad():
        ref = _fp32_sdpa_reference(q, k, v, bias, scale, allow_tf32=True)
        out = window_attn_fwd_cute_qkvpacked(
            qkv, H, bias=bias, scale_qk=scale, output_layout="bnc"
        )

    tol = _TF32_VS_SDPA_MAX_ABS[activation_scale]
    max_err = (out - ref.permute(0, 2, 1, 3).reshape(Bwin, N, H * Dh)).abs().max().item()
    assert max_err < tol, (
        f"qkvpacked vs SDPA activation_scale={activation_scale} max_err={max_err} tol={tol}"
    )


@requires_cute
@pytest.mark.parametrize("activation_scale", _LARGE_ACT_SCALES)
def test_bf16_swin_mask_large_activations(activation_scale: float) -> None:
    """BF16 CuTe + mask must track FP32 stable (catches mask regression, allows BF16 MMA slack)."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q_f, k_f, v_f = _make_qkv(
        Bwin, H, N, Dh, torch.float32, "cuda", activation_scale=activation_scale
    )
    bias = _make_bias(nW, N, "cuda")
    scale = 1.0 / math.sqrt(Dh)
    q, k, v = q_f.bfloat16(), k_f.bfloat16(), v_f.bfloat16()
    tol = _BF16_VS_STABLE_MAX_ABS[activation_scale]

    with torch.no_grad():
        stable = _attention_window_stable(q_f, k_f, v_f, bias, scale, strict_fp32=False)
        out = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.BF16_MIXED
        )

    max_err = (out.float() - stable).abs().max().item()
    assert max_err < tol, (
        f"vs stable activation_scale={activation_scale} max_err={max_err} tol={tol}"
    )


@requires_cute
def test_tf32_is_deterministic() -> None:
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    with torch.no_grad():
        out1 = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.TF32_ACC_FP32)
        out2 = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.TF32_ACC_FP32)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


# ===========================================================================
# BF16_MIXED path  (CuTeDSL kernel)
# ===========================================================================

@requires_cute
@pytest.mark.parametrize("Bwin,H,N,Dh,nW", AURORA_SHAPES + EXTRA_SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_bf16_mixed_close_to_fp32_reference(
    Bwin: int, H: int, N: int, Dh: int, nW: int, has_bias: bool
) -> None:
    """BF16_MIXED output must be numerically close to the FP32 ground truth.

    BF16 has ~0.4 % relative precision; we allow 2 % rtol / 2e-2 atol to
    account for accumulated rounding across the attention softmax.
    """
    q_f, k_f, v_f = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    q = q_f.bfloat16()
    k = k_f.bfloat16()
    v = v_f.bfloat16()

    with torch.no_grad():
        ref = _fp32_sdpa_reference(q_f, k_f, v_f, bias, scale, allow_tf32=False)
        out = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.BF16_MIXED
        )

    torch.testing.assert_close(
        out.float(), ref,
        rtol=2e-2, atol=2e-2,
        msg=f"shape=({Bwin},{H},{N},{Dh}) has_bias={has_bias}",
    )


@requires_cute
def test_bf16_mixed_output_dtype_and_shape() -> None:
    Bwin, H, N, Dh = 4, 8, 144, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.bfloat16, "cuda")
    with torch.no_grad():
        out = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
    assert out.dtype == torch.bfloat16
    assert out.shape == (Bwin, H, N, Dh)


@requires_cute
def test_bf16_mixed_is_deterministic() -> None:
    Bwin, H, N, Dh = 4, 8, 144, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.bfloat16, "cuda")
    with torch.no_grad():
        out1 = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
        out2 = window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


@requires_cute
@pytest.mark.parametrize("has_bias", [False, True])
def test_bf16_qkvpacked_matches_regular_cute(has_bias: bool) -> None:
    """Packed qkv path must match the existing q/k/v CuTe path."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.bfloat16, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    qkv = torch.empty(Bwin, N, 3 * H * Dh, device="cuda", dtype=torch.bfloat16)
    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    qkv_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    qkv_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    qkv_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))

    with torch.no_grad():
        regular = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.BF16_MIXED
        )
        packed = window_attn_fwd_cute_qkvpacked(
            qkv, H, bias=bias, scale_qk=scale
        )

    torch.testing.assert_close(packed, regular, rtol=0, atol=0)


@requires_cute
@pytest.mark.parametrize("has_bias", [False, True])
def test_bf16_qkvpacked_bnc_output_matches_regular_cute(has_bias: bool) -> None:
    """Packed qkv can write directly to the projection-friendly (Bwin,N,C) layout."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.bfloat16, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    qkv = torch.empty(Bwin, N, 3 * H * Dh, device="cuda", dtype=torch.bfloat16)
    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    qkv_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    qkv_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    qkv_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))

    with torch.no_grad():
        regular = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.BF16_MIXED
        )
        packed_bnc = window_attn_fwd_cute_qkvpacked(
            qkv, H, bias=bias, scale_qk=scale, output_layout="bnc"
        )

    expected = regular.permute(0, 2, 1, 3).reshape(Bwin, N, H * Dh)
    assert packed_bnc.shape == (Bwin, N, H * Dh)
    assert packed_bnc.is_contiguous()
    torch.testing.assert_close(packed_bnc, expected, rtol=0, atol=0)


@requires_cute
@pytest.mark.parametrize("has_bias", [False, True])
def test_tf32_qkvpacked_matches_regular_cute(has_bias: bool) -> None:
    """Packed qkv TF32 path must match the existing q/k/v CuTe path."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    qkv = torch.empty(Bwin, N, 3 * H * Dh, device="cuda", dtype=torch.float32)
    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    qkv_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    qkv_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    qkv_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))

    with torch.no_grad():
        regular = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.TF32_ACC_FP32
        )
        packed = window_attn_fwd_cute_qkvpacked(
            qkv, H, bias=bias, scale_qk=scale
        )

    torch.testing.assert_close(packed, regular, rtol=0, atol=0)


@requires_cute
@pytest.mark.parametrize("has_bias", [False, True])
def test_tf32_qkvpacked_bnc_output_matches_regular_cute(has_bias: bool) -> None:
    """Packed qkv TF32 can write directly to the projection-friendly (Bwin,N,C) layout."""
    Bwin, H, N, Dh, nW = 6, 8, 144, 64, 3
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias = _make_bias(nW, N, "cuda") if has_bias else None
    scale = 1.0 / math.sqrt(Dh)

    qkv = torch.empty(Bwin, N, 3 * H * Dh, device="cuda", dtype=torch.float32)
    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    qkv_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    qkv_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    qkv_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))

    with torch.no_grad():
        regular = window_attn_fwd_cute(
            q, k, v, bias, scale_qk=scale, precision=WinAttnPrecision.TF32_ACC_FP32
        )
        packed_bnc = window_attn_fwd_cute_qkvpacked(
            qkv, H, bias=bias, scale_qk=scale, output_layout="bnc"
        )

    expected = regular.permute(0, 2, 1, 3).reshape(Bwin, N, H * Dh)
    assert packed_bnc.shape == (Bwin, N, H * Dh)
    assert packed_bnc.is_contiguous()
    torch.testing.assert_close(packed_bnc, expected, rtol=0, atol=0)


# ===========================================================================
# window_attn_dispatch — routing tests
# ===========================================================================

@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dispatch_output_shape_and_dtype(dtype: torch.dtype) -> None:
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, dtype, "cuda")
    with torch.no_grad():
        out = window_attn_dispatch(q, k, v)
    assert out.shape == (Bwin, H, N, Dh)
    assert out.dtype == dtype


@requires_cuda
def test_dispatch_fp32_tf32_vs_strict_numerically_close() -> None:
    """TF32 CuTe kernel and strict-FP32 SDPA produce results within tolerance.

    fp32_precision="tf32"   → CuTeDSL TF32 kernel (or SDPA if CuTe unavailable)
    fp32_precision="strict" → torch SDPA with TF32 disabled (Aurora's native path)
    """
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    with torch.no_grad():
        tf32_out   = window_attn_dispatch(q, k, v, fp32_precision="tf32")
        strict_out = window_attn_dispatch(q, k, v, fp32_precision="strict")
    # TF32 truncates mantissa to 10 bits; differences should be small but non-zero
    torch.testing.assert_close(tf32_out, strict_out, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_dispatch_strict_fp32_matches_sdpa() -> None:
    """fp32_precision='strict' must exactly match torch SDPA (TF32 disabled)."""
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    scale = 1.0 / math.sqrt(Dh)
    with torch.no_grad():
        strict_out = window_attn_dispatch(
            q, k, v, scale_qk=scale, fp32_precision="strict"
        )
        ref = _fp32_sdpa_reference(q, k, v, None, scale, allow_tf32=False)
    torch.testing.assert_close(strict_out, ref, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_dispatch_bias_broadcast_correctness() -> None:
    """(nW, N, N) bias must produce same result as manually expanded (Bwin,1,N,N).

    Uses the strict-FP32 SDPA path for bit-exact comparison.
    """
    Bwin, H, N, Dh, nW = 8, 4, 64, 64, 4
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    bias_nW = _make_bias(nW, N, "cuda")           # (nW, N, N)

    win_ids = torch.arange(Bwin, device="cuda") % nW
    bias_full = bias_nW[win_ids].unsqueeze(1)     # (Bwin, 1, N, N)
    scale = 1.0 / math.sqrt(Dh)

    with torch.no_grad():
        out = window_attn_dispatch(
            q, k, v, bias=bias_nW, scale_qk=scale, fp32_precision="strict"
        )
        ref = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias_full, scale=scale
        )

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_dispatch_no_cute_fallback_to_sdpa() -> None:
    """use_cute=False must give the same result as plain SDPA."""
    Bwin, H, N, Dh = 4, 8, 64, 64
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cuda")
    scale = 1.0 / math.sqrt(Dh)
    with torch.no_grad():
        no_cute = window_attn_dispatch(q, k, v, scale_qk=scale, use_cute=False)
        sdpa    = F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.testing.assert_close(no_cute, sdpa, rtol=1e-6, atol=1e-6)


# ===========================================================================
# Bias expand helper
# ===========================================================================

@requires_cuda
def test_expand_bias_for_sdpa_Bwin_multiple_of_nW() -> None:
    """When Bwin % nW == 0, expand uses a zero-copy repeat."""
    nW, N, Bwin, H = 4, 64, 8, 4
    bias = _make_bias(nW, N, "cuda")
    expanded = _expand_bias_for_sdpa(bias, Bwin, H, N)
    assert expanded.shape == (Bwin, 1, N, N)
    for b in range(Bwin):
        torch.testing.assert_close(expanded[b, 0], bias[b % nW])


@requires_cuda
def test_expand_bias_for_sdpa_general_Bwin() -> None:
    """General (Bwin not multiple of nW) path uses index-gather."""
    nW, N, Bwin, H = 3, 64, 7, 4   # 7 % 3 != 0
    bias = _make_bias(nW, N, "cuda")
    expanded = _expand_bias_for_sdpa(bias, Bwin, H, N)
    assert expanded.shape == (Bwin, 1, N, N)
    for b in range(Bwin):
        torch.testing.assert_close(expanded[b, 0], bias[b % nW])


# ===========================================================================
# Error / guard tests (CPU-compatible)
# ===========================================================================

def test_tf32_dtype_guard() -> None:
    """TF32_ACC_FP32 must reject non-float32 tensors."""
    q, k, v = (torch.randn(2, 4, 16, 16) for _ in range(3))
    with pytest.raises(AssertionError, match="float32"):
        window_attn_fwd_cute(
            q.bfloat16(), k.bfloat16(), v.bfloat16(),
            precision=WinAttnPrecision.TF32_ACC_FP32,
        )


def test_bf16_mixed_dtype_guard() -> None:
    """BF16_MIXED must reject non-bfloat16 tensors."""
    q, k, v = (torch.randn(2, 4, 16, 16) for _ in range(3))
    with pytest.raises(AssertionError, match="bfloat16"):
        window_attn_fwd_cute(
            q, k, v,
            precision=WinAttnPrecision.BF16_MIXED,
        )


def test_dispatch_cpu_fallback_returns_correct_shape() -> None:
    """CPU tensors must fall through to plain SDPA without error."""
    Bwin, H, N, Dh = 2, 4, 16, 16
    q, k, v = _make_qkv(Bwin, H, N, Dh, torch.float32, "cpu")
    out = window_attn_dispatch(q, k, v)
    assert out.shape == (Bwin, H, N, Dh)


# ===========================================================================
# Tile-selection unit tests (CPU-compatible, no CUDA required)
# ===========================================================================

class TestChooseTileN:
    """Validate SMEM-budget-aware tile selection logic.

    We pass an explicit ``smem_budget_bytes`` to avoid dependency on the
    host GPU, making these tests runnable without CUDA.
    """

    # ----- BF16 _choose_tile_n -----

    def test_bf16_n144_single_pass_on_48kb(self) -> None:
        """N=144 single-pass on 48 KB (mask is uint8 in gmem, not SMEM)."""
        tile_n = _choose_tile_n(144, head_dim=64, tile_m=64, smem_budget_bytes=48 * 1024)
        assert tile_n == 144
        smem = 64 * 64 * 2 + 2 * tile_n * 64 * 2
        assert smem <= 48 * 1024

    def test_bf16_n288_single_pass_99kb(self) -> None:
        """N=288 must select tile_n=288 (single-pass) on a 99 KB device."""
        tile_n = _choose_tile_n(288, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
        assert tile_n == 288
        smem = 64 * 64 * 2 + 2 * tile_n * 64 * 2
        assert smem <= 99 * 1024

    def test_bf16_n288_streaming_on_48kb(self) -> None:
        """N=288 must stream (tile_n < 288) on a 48 KB device."""
        tile_n = _choose_tile_n(288, head_dim=64, tile_m=64, smem_budget_bytes=48 * 1024)
        assert tile_n < 288
        smem = 64 * 64 * 2 + 4 * tile_n * 64 * 2
        assert smem <= 48 * 1024 // 2

    def test_bf16_n576_streaming_99kb(self) -> None:
        """N=576 must stream even on 99 KB (total would be 152 KB)."""
        tile_n = _choose_tile_n(576, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
        assert tile_n < 576
        smem = 64 * 64 * 2 + 4 * tile_n * 64 * 2
        assert smem <= 99 * 1024 // 2

    def test_bf16_tile_n_aligned_to_8(self) -> None:
        """tile_n must be a multiple of 8 (MMA-N dimension)."""
        for N in (100, 150, 200, 300, 500):
            tile_n = _choose_tile_n(N, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
            assert tile_n % 8 == 0, f"tile_n={tile_n} not aligned for N={N}"

    def test_bf16_tile_n_never_exceeds_seq_len(self) -> None:
        """tile_n must be <= seq_len (no over-allocation into undefined K/V rows)."""
        budget = 99 * 1024
        for N in (8, 32, 64, 144, 288, 576):
            tile_n = _choose_tile_n(N, head_dim=64, tile_m=64, smem_budget_bytes=budget)
            assert tile_n <= N, f"tile_n={tile_n} > N={N}"
            stages = 1 if tile_n >= N else 2
            kv_factor = 2 if stages == 1 else 4
            smem = 64 * 64 * 2 + kv_factor * tile_n * 64 * 2
            cap = budget if stages == 1 else budget // 2
            assert smem <= cap, f"SMEM {smem} > cap {cap} for N={N}"

    # ----- TF32 _choose_tile_n_tf32 -----

    def test_tf32_n144_single_pass_99kb(self) -> None:
        """N=144 must select tile_n=144 (single-pass) on 99 KB device."""
        tile_n = _choose_tile_n_tf32(144, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
        # sQ=16KB, sK+sV=144×64×(4+2)B=54KB → 70KB < 99KB
        assert tile_n == 144
        assert _tf32_hybrid_smem_bytes(tile_n, 64, num_stages=1) <= 99 * 1024

    def test_tf32_n288_streams_99kb_hybrid(self) -> None:
        """N=288 must stream on 99 KB; 2-stage hybrid SMEM fits half budget."""
        tile_n = _choose_tile_n_tf32(288, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
        assert tile_n < 288
        assert tile_n == 32
        assert _tf32_hybrid_smem_bytes(tile_n, 64, num_stages=2) <= 99 * 1024 // 2

    def test_tf32_n144_streaming_on_48kb(self) -> None:
        """N=144 must stream on a 48 KB device (70 KB single-pass > 48 KB budget)."""
        tile_n = _choose_tile_n_tf32(144, head_dim=64, tile_m=64, smem_budget_bytes=48 * 1024)
        assert tile_n < 144
        assert _tf32_hybrid_smem_bytes(tile_n, 64, num_stages=2) <= 48 * 1024

    def test_tf32_tile_n_aligned_to_8(self) -> None:
        for N in (100, 144, 200, 288):
            tile_n = _choose_tile_n_tf32(N, head_dim=64, tile_m=64, smem_budget_bytes=99 * 1024)
            assert tile_n % 8 == 0

    def test_tf32_tile_n_never_exceeds_seq_len(self) -> None:
        """tile_n must be <= seq_len (no over-allocation into undefined K/V rows)."""
        budget = 99 * 1024
        for N in (8, 32, 64, 144, 288):
            tile_n = _choose_tile_n_tf32(N, head_dim=64, tile_m=64, smem_budget_bytes=budget)
            assert tile_n <= N, f"tile_n={tile_n} > N={N}"
            stages = 1 if tile_n >= N else 2
            limit = budget if stages == 1 else budget // 2
            smem = _tf32_hybrid_smem_bytes(tile_n, 64, num_stages=stages)
            assert smem <= limit, f"SMEM {smem} > limit {limit} for N={N}"

    # ----- _get_smem_budget_bytes -----

    def test_smem_budget_is_positive(self) -> None:
        budget = _get_smem_budget_bytes()
        assert budget > 0
        assert budget % 1024 == 0  # should be an integral number of KB

    @requires_cuda
    def test_smem_budget_ge_48kb_on_cuda(self) -> None:
        """Any CUDA device should have at least 48 KB available."""
        budget = _get_smem_budget_bytes()
        assert budget >= 48 * 1024
