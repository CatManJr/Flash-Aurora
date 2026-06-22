"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Window attention forward via CuTeDSL kernels (SM80+ MMA; SM120 TMA stream when available).

Kernels: ``_kernel_bf16.py`` / ``_kernel_fp32.py``. Tile sizes: ``_smem_utils.py``.

References:
- flash-attn ``flash_attn/cute/interface.py`` — dispatch / compile-cache patterns (Tri Dao).
"""
import math
import os
from enum import Enum
from typing import Optional

import torch

from ._smem_utils import (  # noqa: F401
    _get_smem_budget_bytes,
    _choose_tile_n,
    _choose_tile_n_tf32,
    _tf32_hybrid_smem_bytes,
)
from ._window_softmax import swin_attn_mask_u8

try:
    from ._kernel_bf16 import (
        _CUTE_AVAILABLE,
        _get_or_compile_bf16,
        _get_or_compile_bf16_stream,
        _get_or_compile_bf16_qkvpacked,
    )
    from ._kernel_fp32 import (
        _get_or_compile_tf32,
        _get_or_compile_tf32_qkvpacked,
    )
    from cutlass import Float32

except ImportError:
    _CUTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Precision mode enum
# ---------------------------------------------------------------------------

class WinAttnPrecision(Enum):
    """Precision mode for :func:`window_attn_fwd_cute`.

    ``BF16_MIXED``
        BF16 I/O, FP32 accumulators — SM80 ``mma.sync.m16n8k16.bf16.bf16.f32``.
    ``TF32_ACC_FP32``
        FP32 I/O, TF32 matmul — SM80 ``mma.sync.m16n8k8.tf32.tf32.f32``.
    """

    BF16_MIXED    = "bf16_mixed"
    TF32_ACC_FP32 = "tf32_acc_fp32"


# ---------------------------------------------------------------------------
# Torch-side helpers (SDPA reference for tests / benchmarks)
# ---------------------------------------------------------------------------

def _require_cute_available() -> None:
    if not _CUTE_AVAILABLE:
        raise RuntimeError(
            "Aurora CuTe window attention requires CuTeDSL (nvidia-cutlass-dsl + quack). "
            "Use the original Aurora model path with use_cute_window_attn=False instead."
        )


def _expand_bias_for_sdpa(
    bias: torch.Tensor,
    Bwin: int,
    H: int,
    N: int,
) -> torch.Tensor:
    """Expand (nW, N, N) bias to (Bwin, 1, N, N) for torch SDPA.

    Uses ``expand`` (a zero-copy view) followed by a contiguous copy only
    when necessary so that SDPA can index the attention bias directly.
    The broadcast over H is done by the leading dim of size 1.
    """
    nW = bias.shape[0]
    if Bwin % nW == 0:
        bias_expanded = bias.unsqueeze(1)   # (nW, 1, N, N)
        reps = Bwin // nW
        if reps > 1:
            bias_expanded = bias_expanded.repeat(reps, 1, 1, 1)
    else:
        win_ids = torch.arange(Bwin, device=bias.device) % nW
        bias_expanded = bias[win_ids].unsqueeze(1)   # (Bwin, 1, N, N)
    return bias_expanded


# ---------------------------------------------------------------------------
# CuTeDSL entry point
# ---------------------------------------------------------------------------

# Single-pass BF16 (tile_n >= N, e.g. production N=144) routing.
# Default OFF: the 128-thread cp.async kernel beats the TMA Stream kernel here.
# With a single KV tile there is nothing for the dedicated DMA warp to prefetch,
# so its warp-specialization (idle warp + extra mbarrier syncs) is pure overhead.
# Empirically (sm_120a, ERA5 enc): nomask ties (~0.73ms), masked Stream is ~8%
# slower (1.03ms vs 0.95ms cp.async) even after register/mask tuning. The TMA
# pipeline only pays off for multi-pass (tile_n < N), which always uses Stream.
# Set AURORA_BF16_STREAM_SINGLE_PASS=1 to force Stream for A/B comparison.
_ENV_BF16_STREAM_SINGLE = "AURORA_BF16_STREAM_SINGLE_PASS"
_CUTE_KERNEL_VERSION = "cute"


def _bf16_stream_single_pass_enabled() -> bool:
    return os.environ.get(_ENV_BF16_STREAM_SINGLE, "0") != "0"


def _best_tile_m(is_bf16: bool, has_bias: bool) -> int:
    """Optimal tile_m for single-pass N=144 on sm_120a (benchmark/_sweep_tile_m.py).

    BF16  → 64 always (tile_m=128 is ~2.2x slower when masked).
    TF32  → 64 when masked (Swin SW-MSA blocks); 128 when unmasked (W-MSA blocks).
    """
    if is_bf16:
        return 64
    return 64 if has_bias else 128


def _bf16_tile_n(seq_len: int, head_dim: int, tile_m: int, tile_n: Optional[int]) -> int:
    if tile_n is not None:
        return tile_n
    return _choose_tile_n(seq_len, head_dim=head_dim, tile_m=tile_m)


def _require_cuda_for_cute(q: torch.Tensor) -> None:
    """CuTe kernels require CUDA; arch is picked by CuTe DSL from the current device."""
    if not q.is_cuda:
        raise RuntimeError("Aurora CuTe window attention requires CUDA tensors.")


def window_attn_fwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    scale_qk: Optional[float] = None,
    precision: WinAttnPrecision = WinAttnPrecision.BF16_MIXED,
    tile_m: int = 64,
    tile_n: Optional[int] = None,
) -> torch.Tensor:
    """Scaled dot-product attention on windowed ``q,k,v``; optional per-window ``bias`` added to logits.

    Parameters
    ----------
    q, k, v:
        Shape ``(Bwin, H, N, Dh)`` where ``Bwin = B * nW``.
        Dtype must match ``precision``:
        * ``BF16_MIXED``    → ``torch.bfloat16``
        * ``TF32_ACC_FP32`` → ``torch.float32``
    bias:
        Optional shifted-window attention mask, shape ``(nW, N, N)``,
        dtype ``torch.float32``.  ``None`` means no mask.
    scale_qk:
        Softmax scaling factor.  Defaults to ``1 / sqrt(Dh)``.
    precision:
        See :class:`WinAttnPrecision`.
    tile_m, tile_n:
        GEMM tile sizes for the CuTeDSL kernel.
    """
    _require_cute_available()
    if precision == WinAttnPrecision.TF32_ACC_FP32:
        assert q.dtype == torch.float32, "TF32_ACC_FP32 requires float32 tensors"
    else:
        assert q.dtype == torch.bfloat16, "BF16_MIXED requires bfloat16 tensors"

    Bwin, H, N, Dh = q.shape
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(Dh)

    _require_cuda_for_cute(q)

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    mask_u8 = swin_attn_mask_u8(bias) if bias is not None else None
    if mask_u8 is not None and not mask_u8.is_contiguous():
        mask_u8 = mask_u8.contiguous()

    # empty (not zeros): the kernel epilogue writes every output row in [0, N) for
    # every (window, head); M-tile rows beyond seqlen map outside the output's N
    # dim and are never read.  zeros_like would memset the full output each call
    # (~0.18 ms for Bwin=1800), erasing the kernel's edge over SDPA.
    out = torch.empty_like(q)
    scale_log2 = Float32(math.log2(math.e) * scale_qk)
    has_bias = bias is not None

    v_run = v
    if precision == WinAttnPrecision.TF32_ACC_FP32:
        _tile_n = tile_n if tile_n is not None else _choose_tile_n_tf32(N, head_dim=Dh, tile_m=tile_m)
        # PV smem uses BF16 + 8x8x16b transpose (ldmatrix.trans is 16-bit only).
        # Single-pass (+ aligned head_dim): pass V as FP32 and let the kernel convert
        # to BF16 during the gmem→smem load, fusing away this full-tensor cast.
        # Multi-pass: keep the host cast (cp.async V prefetch path stays intact).
        if _tile_n >= N and Dh % 16 == 0:
            v_run = v
        else:
            v_run = v.to(torch.bfloat16)
        fn = _get_or_compile_tf32(
            head_dim=Dh, seq_len=N, has_bias=has_bias,
            tile_m=tile_m, tile_n=_tile_n,
            q=q, k=k, v=v_run, o=out, bias_or_none=mask_u8,
        )
    else:
        _tile_n = _bf16_tile_n(N, head_dim=Dh, tile_m=tile_m, tile_n=tile_n)
        # Multi-pass (tile_n < N) always uses the TMA Stream kernel. Single-pass
        # (tile_n >= N, e.g. production N=144) uses the 128-thread cp.async kernel
        # by default — it beats Stream when there is only one KV tile (nothing for
        # the DMA warp to prefetch). Set AURORA_BF16_STREAM_SINGLE_PASS=1 to force
        # Stream on SM120 + Dh=64 for A/B comparison.
        use_stream = _tile_n < N or (
            _bf16_stream_single_pass_enabled() and Dh == 64
        )
        if use_stream:
            fn = _get_or_compile_bf16_stream(
                head_dim=Dh, seq_len=N, has_bias=has_bias,
                tile_m=tile_m, tile_n=_tile_n,
                q=q, k=k, v=v, o=out, bias_or_none=mask_u8,
            )
        else:
            fn = _get_or_compile_bf16(
                head_dim=Dh, seq_len=N, has_bias=has_bias,
                tile_m=tile_m, tile_n=_tile_n,
                q=q, k=k, v=v, o=out, bias_or_none=mask_u8,
            )

    fn(q, k, v_run, out, mask_u8, scale_log2)
    return out


def window_attn_fwd_cute_qkvpacked(
    qkv: torch.Tensor,
    num_heads: int,
    bias: Optional[torch.Tensor] = None,
    *,
    scale_qk: Optional[float] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    output_layout: str = "bhnd",
) -> torch.Tensor:
    """CuTe attention reading Q/K/V directly from packed ``qkv``.

    ``qkv`` is the contiguous output of ``Linear(dim, 3 * dim)`` with shape
    ``(Bwin, N, 3 * num_heads * head_dim)``.  This creates zero-copy strided
    views shaped ``(Bwin, H, N, Dh)`` and uses a separate CuTe compile cache for
    those strides, avoiding the three explicit ``q/k/v.contiguous()`` copies in
    the Swin3D inference path.

    Supports ``torch.bfloat16`` (BF16 mixed) and ``torch.float32`` (TF32 MMA).

    ``output_layout="bnc"`` returns a contiguous ``(Bwin, N, H * Dh)`` tensor by
    passing a strided ``(Bwin, H, N, Dh)`` view into the unchanged CuTe kernel.
    """
    if qkv.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"qkv-packed CuTe path supports bfloat16 and float32; got {qkv.dtype}"
        )
    if qkv.ndim != 3:
        raise ValueError(f"qkv must have shape (Bwin, N, 3*C); got {tuple(qkv.shape)}")
    if not qkv.is_contiguous():
        raise ValueError("qkv-packed CuTe path requires contiguous qkv linear output")
    if qkv.shape[-1] % (3 * num_heads) != 0:
        raise ValueError(
            f"last dim {qkv.shape[-1]} must be divisible by 3*num_heads={3 * num_heads}"
        )
    if output_layout not in {"bhnd", "bnc"}:
        raise ValueError(f"output_layout must be 'bhnd' or 'bnc', got {output_layout!r}")

    _require_cute_available()

    Bwin, N, three_c = qkv.shape
    Dh = three_c // (3 * num_heads)
    H = num_heads
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(Dh)

    _require_cuda_for_cute(qkv)

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    mask_u8 = swin_attn_mask_u8(bias) if bias is not None else None
    if mask_u8 is not None and not mask_u8.is_contiguous():
        mask_u8 = mask_u8.contiguous()

    qkv_view = qkv.view(Bwin, N, 3, H, Dh)
    q = qkv_view[:, :, 0].permute(0, 2, 1, 3)
    k = qkv_view[:, :, 1].permute(0, 2, 1, 3)
    v = qkv_view[:, :, 2].permute(0, 2, 1, 3)
    if output_layout == "bnc":
        out = torch.empty((Bwin, N, H * Dh), device=qkv.device, dtype=qkv.dtype)
        out_kernel = out.view(Bwin, N, H, Dh).permute(0, 2, 1, 3)
    else:
        out = torch.empty((Bwin, H, N, Dh), device=qkv.device, dtype=qkv.dtype)
        out_kernel = out
    scale_log2 = Float32(math.log2(math.e) * scale_qk)
    has_bias = bias is not None
    is_bf16 = qkv.dtype == torch.bfloat16
    if tile_m is None:
        tile_m = _best_tile_m(is_bf16, has_bias)

    if is_bf16:
        _tile_n = _bf16_tile_n(N, head_dim=Dh, tile_m=tile_m, tile_n=tile_n)
        if _tile_n >= N:
            fn = _get_or_compile_bf16_qkvpacked(
                head_dim=Dh, seq_len=N, has_bias=has_bias,
                tile_m=tile_m, tile_n=_tile_n,
                q=q, k=k, v=v, o=out_kernel, bias_or_none=mask_u8,
                output_layout=output_layout,
            )
            v_run = v
        else:
            return window_attn_fwd_cute(
                q.contiguous(), k.contiguous(), v.contiguous(), bias=bias,
                scale_qk=scale_qk, precision=WinAttnPrecision.BF16_MIXED,
                tile_m=tile_m, tile_n=_tile_n,
            )
    else:
        _tile_n = tile_n if tile_n is not None else _choose_tile_n_tf32(N, head_dim=Dh, tile_m=tile_m)
        if _tile_n < N:
            return window_attn_fwd_cute(
                q.contiguous(), k.contiguous(), v.contiguous(), bias=bias,
                scale_qk=scale_qk, precision=WinAttnPrecision.TF32_ACC_FP32,
                tile_m=tile_m, tile_n=_tile_n,
            )
        if Dh % 16 == 0:
            v_compile, v_run = v, v
        else:
            v_compile = v.to(torch.bfloat16)
            v_run = v_compile
        fn = _get_or_compile_tf32_qkvpacked(
            head_dim=Dh, seq_len=N, has_bias=has_bias,
            tile_m=tile_m, tile_n=_tile_n,
            q=q, k=k, v=v_compile, o=out_kernel, bias_or_none=mask_u8,
            output_layout=output_layout,
        )

    fn(q, k, v_run, out_kernel, mask_u8, scale_log2)
    return out


# ---------------------------------------------------------------------------
# Auto-routing dispatch (preferred call-site API)
# ---------------------------------------------------------------------------

def window_attn_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    scale_qk: Optional[float] = None,
    tile_m: int = 64,
    tile_n: Optional[int] = None,
) -> torch.Tensor:
    """Run window attention via CuTeDSL (CUDA; arch from current GPU or ``CUTE_DSL_ARCH``).

    Thin wrapper around :func:`window_attn_fwd_cute` that picks precision from
    ``q.dtype`` (``bfloat16`` -> ``BF16_MIXED``, ``float32`` -> ``TF32_ACC_FP32``).
    """
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(q.shape[-1])

    precision = (
        WinAttnPrecision.BF16_MIXED
        if q.dtype == torch.bfloat16
        else WinAttnPrecision.TF32_ACC_FP32
    )
    has_bias = bias is not None
    is_bf16 = q.dtype == torch.bfloat16
    return window_attn_fwd_cute(
        q,
        k,
        v,
        bias,
        scale_qk=scale_qk,
        precision=precision,
        tile_m=_best_tile_m(is_bf16, has_bias) if tile_m == 64 else tile_m,
        tile_n=tile_n,
    )
