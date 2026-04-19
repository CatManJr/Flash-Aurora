"""3D Flash Window Attention kernels.

Modified from:
@misc{zhang2025flashwindowattentionspeedup,
      title={Flash Window Attention: speedup the attention computation for Swin Transformer},
      author={Zhendong Zhang},
      year={2025},
      eprint={2501.06480},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.06480},
      https://github.com/zzd1992/FlashWindowAttention
}
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_FLASH_WINDOW_ATTN_AUTOTUNE_CONFIGS = [
    triton.Config({"block_m": 16, "block_n": 16}, num_warps=4, num_stages=2),
    triton.Config({"block_m": 16, "block_n": 32}, num_warps=4, num_stages=2),
    triton.Config({"block_m": 32, "block_n": 16}, num_warps=4, num_stages=2),
    triton.Config({"block_m": 32, "block_n": 32}, num_warps=4, num_stages=2),
    triton.Config({"block_m": 32, "block_n": 32}, num_warps=8, num_stages=2),
    triton.Config({"block_m": 64, "block_n": 16}, num_warps=8, num_stages=2),
    triton.Config({"block_m": 64, "block_n": 32}, num_warps=8, num_stages=2),
]

_FLASH_WINDOW_ATTN_FULL_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_FLASH_WINDOW_ATTN_FULL_AUTOTUNE_CONFIGS, key=["seq", "head_dim", "has_bias"])
@triton.jit
def _flash_window_attn_3d_fwd_full_kernel(
    Q,
    K,
    V,
    bias,
    O,
    scale_qk: tl.constexpr,
    has_bias: tl.constexpr,
    batch: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    seq: tl.constexpr,
    seq_pad: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    stride_head = seq * head_dim
    stride_batch = stride_head * head
    offset = batch_id * stride_batch + head_id * stride_head

    if has_bias:
        bias_ptr = tl.make_block_ptr(
            base=bias + (batch_id * head + head_id) * seq * seq,
            shape=(seq, seq),
            strides=(seq, 1),
            offsets=(0, 0),
            block_shape=(seq_pad, seq_pad),
            order=(1, 0),
        )
        bias_data = tl.load(bias_ptr, boundary_check=(0, 1), padding_option="zero")

    mask = tl.arange(0, seq_pad) < seq
    attn = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)

    q_ptr = tl.make_block_ptr(
        base=Q + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    k_ptr = tl.make_block_ptr(
        base=K + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    for _ in range(head_chunk):
        q_data = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
        attn = tl.dot(q_data, k_data.trans(1, 0), acc=attn, input_precision="ieee")
        q_ptr = tl.advance(q_ptr, (0, chunk_dim))
        k_ptr = tl.advance(k_ptr, (0, chunk_dim))

    attn *= scale_qk
    if has_bias:
        attn += bias_data
    attn += tl.where(mask[None, :], 0, -float("inf"))
    attn -= tl.max(attn, axis=1, keep_dims=True)
    attn = tl.math.exp(attn)
    attn /= tl.sum(attn, axis=1, keep_dims=True)
    attn = attn.to(Q.dtype.element_ty)

    v_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    index = offset + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    o_ptr = O + index
    for _ in range(head_chunk):
        v_data = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
        o_data = tl.dot(attn, v_data, input_precision="ieee")
        tl.store(o_ptr, o_data.to(Q.dtype.element_ty), mask=mask[:, None])
        v_ptr = tl.advance(v_ptr, (0, chunk_dim))
        o_ptr += chunk_dim


@triton.autotune(configs=_FLASH_WINDOW_ATTN_AUTOTUNE_CONFIGS, key=["seq", "head_dim", "has_bias"])
@triton.jit
def _flash_window_attn_3d_fwd_kernel(
    Q,
    K,
    V,
    bias,
    O,
    scale_qk: tl.constexpr,
    has_bias: tl.constexpr,
    batch: tl.constexpr,
    head: tl.constexpr,
    head_dim: tl.constexpr,
    seq: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    pid2 = tl.program_id(2)
    num_d_blocks = tl.cdiv(head_dim, block_d)
    m_block_id = pid2 // num_d_blocks
    d_block_id = pid2 % num_d_blocks

    stride_head = seq * head_dim
    stride_batch = stride_head * head
    offset = batch_id * stride_batch + head_id * stride_head
    offs_m = m_block_id * block_m + tl.arange(0, block_m)
    offs_d = d_block_id * block_d + tl.arange(0, block_d)
    mask_m = offs_m < seq
    mask_d = offs_d < head_dim

    m_i = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((block_m,), dtype=tl.float32)
    acc = tl.zeros((block_m, block_d), dtype=tl.float32)

    n_start = 0
    while n_start < seq:
        offs_n = n_start + tl.arange(0, block_n)
        mask_n = offs_n < seq

        qk = tl.zeros((block_m, block_n), dtype=tl.float32)
        d_start = 0
        while d_start < head_dim:
            offs_hd = d_start + tl.arange(0, block_d)
            mask_hd = offs_hd < head_dim

            q_ptrs = Q + offset + offs_m[:, None] * head_dim + offs_hd[None, :]
            k_ptrs = K + offset + offs_n[:, None] * head_dim + offs_hd[None, :]
            q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_hd[None, :], other=0.0)
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_hd[None, :], other=0.0)
            qk = tl.dot(q, k.trans(1, 0), acc=qk, input_precision="ieee")
            d_start += block_d

        qk *= scale_qk
        if has_bias:
            b_ptrs = bias + (batch_id * head + head_id) * seq * seq + offs_m[:, None] * seq + offs_n[None, :]
            b = tl.load(b_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            qk += b

        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        p = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_new)
        beta = tl.math.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij

        old_scale = tl.where(l_i > 0, (alpha * l_i) / l_new, 0.0)
        new_scale = beta / l_new
        v_ptrs = V + offset + offs_n[:, None] * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc = acc * old_scale[:, None] + tl.dot(p, v, input_precision="ieee") * new_scale[:, None]

        m_i = m_new
        l_i = l_new
        n_start += block_n

    o_ptrs = O + offset + offs_m[:, None] * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


HEAD_CHUNK_DIM: Final[int] = 16
MAX_SEQ_PAD: Final[int] = 256
MAX_FULL_TILE_SEQ_PAD: Final[int] = 128


def _ceil_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def _full_tile_smem_bytes(seq_pad: int, *, has_bias: bool) -> int:
    # fp32 attention matrix plus optional bias tile.
    bytes_attn = seq_pad * seq_pad * 4
    bytes_bias = seq_pad * seq_pad * 4 if has_bias else 0
    return bytes_attn + bytes_bias


def _device_smem_limit_bytes(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    optin = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    default = int(getattr(props, "shared_memory_per_block", 0) or 0)
    return optin if optin > 0 else default


def _normalize_bias(
    bias: torch.Tensor | None,
    *,
    batch: int,
    head: int,
    seq: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if bias is None:
        return None
    if bias.shape == (seq, seq):
        out = bias.unsqueeze(0).unsqueeze(0).expand(batch, head, seq, seq)
    elif bias.shape == (head, seq, seq):
        out = bias.unsqueeze(0).expand(batch, head, seq, seq)
    elif bias.shape == (1, head, seq, seq):
        out = bias.expand(batch, head, seq, seq)
    elif bias.shape == (1, 1, seq, seq):
        out = bias.expand(batch, head, seq, seq)
    elif bias.shape == (batch, 1, seq, seq):
        out = bias.expand(batch, head, seq, seq)
    elif bias.shape == (batch, head, seq, seq):
        out = bias
    else:
        raise ValueError(
            "Unsupported bias shape. Expected (N,N), (H,N,N), (1,H,N,N), (B,1,N,N), or (B,H,N,N); "
            f"got {tuple(bias.shape)}."
        )
    return out.to(device=device, dtype=dtype).contiguous()


def torch_window_attention_3d_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scale_qk: float | None = None,
) -> torch.Tensor:
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias,
        dropout_p=0.0,
        scale=float(scale_qk),
    )


def flash_window_attn_3d_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scale_qk: float | None = None,
    allow_torch_fallback: bool = False,
) -> torch.Tensor:
    """Compute flash window attention over `(Bwin, H, N, Dh)`.

    Stage-E milestone scope: inference-only forward path.
    """
    if allow_torch_fallback:
        raise ValueError(
            "allow_torch_fallback=True is not supported: flash_window_attn_forward is Triton-only."
        )
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise NotImplementedError("flash_window_attn_forward is inference-only in Stage E milestone.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 tensors shaped (Bwin, H, N, Dh).")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have identical shapes; got {q.shape}, {k.shape}, {v.shape}.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flash_window_attn_forward requires CUDA tensors.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share the same dtype.")

    batch, head, seq, head_dim = q.shape
    scale_qk = float(1.0 / math.sqrt(head_dim)) if scale_qk is None else float(scale_qk)
    # Fast path for the common no-mask/no-bias case: avoid any bias tensor normalization/allocation.
    if bias is None:
        bias_n = None
    else:
        bias_n = _normalize_bias(
            bias,
            batch=batch,
            head=head,
            seq=seq,
            device=q.device,
            dtype=q.dtype,
        )

    if head_dim % HEAD_CHUNK_DIM != 0:
        raise ValueError(
            f"Unsupported head_dim={head_dim}; must be divisible by {HEAD_CHUNK_DIM}."
        )
    seq_pad = _ceil_pow2(seq)
    if seq_pad > MAX_SEQ_PAD:
        raise ValueError(
            f"Unsupported sequence length N={seq} (padded={seq_pad}); expected padded N <= {MAX_SEQ_PAD}."
        )

    out = torch.empty_like(q)
    full_smem_ok = _full_tile_smem_bytes(seq_pad, has_bias=bias_n is not None) <= _device_smem_limit_bytes(q.device)
    use_full_tile = seq_pad <= MAX_FULL_TILE_SEQ_PAD and full_smem_ok

    if use_full_tile:
        _flash_window_attn_3d_fwd_full_kernel[(batch, head)](
            q,
            k,
            v,
            bias_n,
            out,
            scale_qk,
            bias_n is not None,
            batch,
            head,
            head_dim,
            head_dim // HEAD_CHUNK_DIM,
            HEAD_CHUNK_DIM,
            seq,
            seq_pad,
        )
    else:
        def grid(meta):
            return (batch, head, triton.cdiv(seq, meta["block_m"]) * triton.cdiv(head_dim, HEAD_CHUNK_DIM))

        _flash_window_attn_3d_fwd_kernel[grid](
            q,
            k,
            v,
            bias_n,
            out,
            scale_qk,
            bias_n is not None,
            batch,
            head,
            head_dim,
            seq,
            block_d=HEAD_CHUNK_DIM,
        )
    return out


def flash_window_attn_3d_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_qk: float | None = None,
) -> torch.Tensor:
    """Compatibility alias for external call sites."""
    return flash_window_attn_3d_forward(q, k, v, bias, scale_qk=scale_qk)