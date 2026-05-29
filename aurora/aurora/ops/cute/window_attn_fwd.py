"""Window attention forward: correctness-first stable softmax, optional CuTe, SDPA fallback.

Default :func:`window_attn_fwd_cute` follows the same numerics as a typical fused
window kernel: ``QK^T * scale + bias`` in FP32, row-wise ``max``, ``exp``,
normalize, then ``@ V``.  Set ``AURORA_CUTE_WINDOW_ATTN=1`` to use the
experimental v2 CuTeDSL kernels (``_kernel_bf16_v2.py``, ``_kernel_fp32_v2.py``).
SMEM tile sizes: ``_smem_utils.py``.
"""
import contextlib
import math
import os
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from ._smem_utils import (  # noqa: F401
    _get_smem_budget_bytes,
    _choose_tile_n,
    _choose_tile_n_tf32,
    _tf32_hybrid_smem_bytes,
)

try:
    from ._kernel_bf16_v2 import (
        WindowAttnFwdBf16V2,
        _get_or_compile_bf16_v2,
        _CUTE_AVAILABLE,
    )
    # v1: simpler 128-thread cp.async BF16 kernel.  Used for the single-pass case
    # (the dominant N=144 production shape), where v2's dedicated DMA warp has no
    # prefetch overlap to exploit and only costs MMA throughput.
    from ._kernel_bf16 import _get_or_compile_bf16
    from ._kernel_fp32_v2 import (
        WindowAttnFwdTF32V2,
        _get_or_compile_tf32_v2,
    )
    from cutlass import Float32

except ImportError:
    _CUTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Precision mode enum
# ---------------------------------------------------------------------------

class WinAttnPrecision(Enum):
    """Precision mode for :func:`window_attn_fwd_cute`.

    By default :func:`window_attn_fwd_cute` uses an explicit FP32 softmax path
    (BF16 or FP32 I/O).  With ``AURORA_CUTE_WINDOW_ATTN=1`` and CuTeDSL available:

    ``BF16_MIXED``
        BF16 I/O, FP32 accumulators — SM80 ``mma.sync.m16n8k16.bf16.bf16.f32``.
    ``TF32_ACC_FP32``
        FP32 I/O, TF32 matmul — SM80 ``mma.sync.m16n8k8.tf32.tf32.f32``.

    For strict FP32 (no TF32 in torch matmuls) use
    ``fp32_precision="strict"`` with :func:`window_attn_dispatch`.
    """

    BF16_MIXED    = "bf16_mixed"
    TF32_ACC_FP32 = "tf32_acc_fp32"


# ---------------------------------------------------------------------------
# Torch-side helpers (SDPA fallback path)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _tf32_disabled():
    """Temporarily disable TF32 for torch matmuls (strict FP32 path)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


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


def _attention_window_stable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_qk: float,
    *,
    strict_fp32: bool = False,
) -> torch.Tensor:
    """Scaled softmax attention with explicit stable softmax (FP32 logits).

    Same recipe as a merged sliding-window reference: logits, row max subtract,
    ``exp``, L1 normalize rows, weighted sum of ``V``.  Output dtype matches ``q``.
    """
    dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    Bwin, H, N, Dh = q.shape
    ctx = _tf32_disabled() if strict_fp32 else contextlib.nullcontext()
    with ctx:
        logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale_qk
    if bias is not None:
        logits = logits + _expand_bias_for_sdpa(bias, Bwin, H, N)
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits = logits - logits_max
    weights = torch.exp(logits)
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom
    with ctx:
        out = torch.matmul(weights, vf)
    return out.to(dtype)


# ---------------------------------------------------------------------------
# CuTeDSL entry point
# ---------------------------------------------------------------------------

_ENV_CUTE_WINDOW = "AURORA_CUTE_WINDOW_ATTN"
_CUTE_KERNEL_VERSION = "v2"


def _require_sm120_v2(q: torch.Tensor) -> None:
    """Fail loudly when the v2 CuTe path is used outside its target architecture."""
    if not q.is_cuda:
        raise RuntimeError("Aurora CuTe window attention v2 requires CUDA tensors.")

    major, minor = torch.cuda.get_device_capability(q.device)
    if major * 10 + minor != 120:
        raise RuntimeError(
            "Aurora CuTe window attention v2 currently targets sm_120 only; "
            f"got compute capability {major}.{minor}."
        )

    cute_arch = os.environ.get("CUTE_DSL_ARCH")
    if cute_arch is not None and cute_arch not in {"sm_120", "sm_120a"}:
        raise RuntimeError(
            "Aurora CuTe window attention v2 expects CUTE_DSL_ARCH=sm_120 "
            f"or sm_120a when set; got {cute_arch!r}."
        )


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
        GEMM tile sizes for the CuTeDSL path (set ``AURORA_CUTE_WINDOW_ATTN=0`` to opt out).
    """
    if precision == WinAttnPrecision.TF32_ACC_FP32:
        assert q.dtype == torch.float32, "TF32_ACC_FP32 requires float32 tensors"
    else:
        assert q.dtype == torch.bfloat16, "BF16_MIXED requires bfloat16 tensors"

    Bwin, H, N, Dh = q.shape
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(Dh)

    # Use the v2 CuTe GEMM kernel by default; set AURORA_CUTE_WINDOW_ATTN=0 to
    # explicitly disable it. Missing CuTe support should fail loudly on this path.
    use_cute_kernel = os.environ.get(_ENV_CUTE_WINDOW, "1") != "0"
    if use_cute_kernel and not _CUTE_AVAILABLE:
        raise RuntimeError(
            "Aurora CuTe window attention v2 is enabled but CuTeDSL is unavailable. "
            f"Set {_ENV_CUTE_WINDOW}=0 to explicitly use the torch fallback."
        )
    if not use_cute_kernel:
        return _attention_window_stable(
            q, k, v, bias, scale_qk, strict_fp32=False,
        )
    _require_sm120_v2(q)

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

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
        fn = _get_or_compile_tf32_v2(
            head_dim=Dh, seq_len=N, has_bias=has_bias,
            tile_m=tile_m, tile_n=_tile_n,
            q=q, k=k, v=v_run, o=out, bias_or_none=bias,
        )
    else:
        _tile_n = tile_n if tile_n is not None else _choose_tile_n(N, head_dim=Dh, tile_m=tile_m)
        # Adaptive kernel selection by KV-pass count:
        #   single-pass (tile_n >= N): v1 — simpler 128-thread cp.async; a dedicated
        #     DMA warp gives no prefetch overlap here and only wastes MMA threads.
        #   multi-pass  (tile_n <  N): v2 — 160-thread heterogeneous TMA pipeline, where
        #     the DMA warp overlaps K/V prefetch with compute across KV tiles.
        if _tile_n >= N:
            fn = _get_or_compile_bf16(
                head_dim=Dh, seq_len=N, has_bias=has_bias,
                tile_m=tile_m, tile_n=_tile_n,
                q=q, k=k, v=v, o=out, bias_or_none=bias,
            )
        else:
            fn = _get_or_compile_bf16_v2(
                head_dim=Dh, seq_len=N, has_bias=has_bias,
                tile_m=tile_m, tile_n=_tile_n,
                q=q, k=k, v=v, o=out, bias_or_none=bias,
            )

    fn(q, k, v_run, out, bias, scale_log2)
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
    use_cute: bool = True,
    fp32_precision: str = "tf32",
    tile_m: int = 64,
    tile_n: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch window attention to the best available backend.

    Routing table
    -------------
    ``use_cute=True`` + CUDA + BF16                        →  :func:`window_attn_fwd_cute` (stable)
    ``use_cute=True`` + CUDA + FP32 + ``fp32_precision="tf32"``
                                                           →  :func:`window_attn_fwd_cute` (stable; TF32 allowed in torch matmuls)
    ``fp32_precision="strict"`` or ``use_cute=False``      →  torch SDPA

    CuTeDSL GEMM kernels run only when ``AURORA_CUTE_WINDOW_ATTN=1`` inside
    :func:`window_attn_fwd_cute`.

    Parameters
    ----------
    q, k, v:
        Shape ``(Bwin, H, N, Dh)``.  Dtype must be ``float32`` or
        ``bfloat16``.
    bias:
        Optional ``(nW, N, N)`` float32 shifted-window mask.
    scale_qk:
        Attention scale.  Defaults to ``1 / sqrt(Dh)``.
    use_cute:
        If ``True`` (default), use the stable explicit-softmax path on CUDA.
        If ``False``, use torch SDPA.
    fp32_precision:
        For float32 inputs only.
        ``"tf32"``    Stable path with TF32 allowed in matmuls (default).
        ``"strict"``  torch SDPA with TF32 disabled.
    tile_m, tile_n:
        Forwarded to CuTeDSL only when ``AURORA_CUTE_WINDOW_ATTN=1``.
    """
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(q.shape[-1])

    Bwin, H, N, _ = q.shape

    use_cute_kernel = (
        use_cute
        and q.is_cuda
        and q.dtype in (torch.float32, torch.bfloat16)
        and not (q.dtype == torch.float32 and fp32_precision == "strict")
    )

    if use_cute_kernel:
        precision = (
            WinAttnPrecision.BF16_MIXED
            if q.dtype == torch.bfloat16
            else WinAttnPrecision.TF32_ACC_FP32
        )
        return window_attn_fwd_cute(
            q, k, v, bias,
            scale_qk=scale_qk,
            precision=precision,
            tile_m= 128 if q.dtype == torch.bfloat16 else 64, # manully tuned on Pro 6000 Blackwell GPU
            tile_n=tile_n,
        )

    # Fallback: Aurora's native SDPA path (strict FP32 / CPU / use_cute=False).
    attn_mask = _expand_bias_for_sdpa(bias, Bwin, H, N) if bias is not None else None
    use_tf32  = not (q.dtype == torch.float32 and fp32_precision == "strict")
    ctx = contextlib.nullcontext() if use_tf32 else _tf32_disabled()
    with ctx:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale_qk)
