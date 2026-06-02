# Copyright (c) 2025, Tri Dao.  (online softmax structure — adapted from flash-attn softmax.py.)
# SPDX-License-Identifier: MIT
"""Row-wise online softmax over attention logits (CuTeDSL).

Fixed ``arch=80`` reduction style: avoids SM100 packed paths that mismatch SM120 + SM80 MMA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8

from quack import layout_utils

# Swin shifted-window masks use -100; treat as hard mask (SDPA-equivalent) not a soft offset.
WINDOW_ATTN_MASK_THRESHOLD = -50.0
# Host packs ``(bias < threshold)`` into uint8 before the CuTe path (1 = masked, 0 = allowed).
WINDOW_ATTN_MASK_U8_DTYPE = torch.uint8
# Large finite sentinel (not -inf) so multi-pass online softmax avoids NaN from -inf - finite.
WINDOW_ATTN_MASKED_LOGIT = -1.0e4


@cute.jit
def _fmax_reduce_row(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    res = cute.make_rmem_tensor(x.shape, Float32)
    res.store(x)
    v0, v1, v2, v3 = res[0], res[1], res[2], res[3]
    for i in cutlass.range_constexpr(4, cute.size(res), 4):
        v0 = cute.arch.fmax(v0, res[i + 0])
        v1 = cute.arch.fmax(v1, res[i + 1])
        v2 = cute.arch.fmax(v2, res[i + 2])
        v3 = cute.arch.fmax(v3, res[i + 3])
    v0 = cute.arch.fmax(v0, v1)
    v2 = cute.arch.fmax(v2, v3)
    v0 = cute.arch.fmax(v0, v2)
    if cutlass.const_expr(init_val is not None):
        v0 = cute.arch.fmax(v0, init_val)
    return v0


@cute.jit
def _fadd_reduce_row(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    if cutlass.const_expr(init_val is None):
        return x.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@cute.jit
def apply_partial_kv_mask(
    acc_S: cute.Tensor,
    tScS: cute.Tensor,
    n_start: Int32,
    seqlen: Int32,
) -> None:
    """Zero-out logits for key indices past ``seqlen`` (partial last KV tile)."""
    neg_inf = -Float32.inf
    for i in cutlass.range(cute.size(acc_S), unroll_full=True):
        n_idx = n_start + tScS[i][1]
        if n_idx >= seqlen:
            acc_S[i] = neg_inf


@cute.jit
def apply_swin_bias_mask(
    acc_S: cute.Tensor,
    tScS: cute.Tensor,
    mBias_w: cute.Tensor,
    m_start: Int32,
    n_start: Int32,
    seqlen: Int32,
) -> None:
    """Apply Swin shifted-window mask: disallowed pairs → large negative, allowed keep matmul."""
    neg_inf = -Float32.inf
    masked_logit = Float32(WINDOW_ATTN_MASKED_LOGIT)
    threshold = Float32(WINDOW_ATTN_MASK_THRESHOLD)

    for i in cutlass.range(cute.size(acc_S), unroll_full=True):
        n_idx = n_start + tScS[i][1]
        m_idx = m_start + tScS[i][0]
        m_valid = m_idx < seqlen
        n_valid = n_idx < seqlen
        if m_valid & n_valid:
            b = mBias_w[m_idx, n_idx]
            if b < threshold:
                acc_S[i] = masked_logit
        else:
            acc_S[i] = neg_inf


@cute.jit
def apply_swin_mask_u8_gmem(
    acc_S: cute.Tensor,
    tScS: cute.Tensor,
    mMask_w: cute.Tensor,
    m_start: Int32,
    n_start: Int32,
    seqlen: Int32,
    n_always_valid: cutlass.Constexpr[bool] = False,
    rows_all_valid: bool = False,
) -> None:
    """Apply uint8 Swin mask from gmem (1 = masked, 0 = allowed).

    The packed mask ``(N, N)`` is shared across all (window, head) CTAs, so the
    1-byte loads stay L2-resident. ``n_always_valid`` drops the column bound
    check for single-pass (``tile_n == seqlen``). ``rows_all_valid`` is a runtime
    predicate: when the whole m-block lies within ``seqlen`` (every m-block except
    the partial last one), the per-element row-bounds branch is skipped entirely.
    """
    neg_inf = -Float32.inf
    masked_logit = Float32(WINDOW_ATTN_MASKED_LOGIT)
    zero_u8 = Uint8(0)

    if cutlass.const_expr(n_always_valid):
        if rows_all_valid:
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                if mMask_w[m_start + tScS[i][0], n_start + tScS[i][1]] != zero_u8:
                    acc_S[i] = masked_logit
        else:
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                m_idx = m_start + tScS[i][0]
                if m_idx < seqlen:
                    if mMask_w[m_idx, n_start + tScS[i][1]] != zero_u8:
                        acc_S[i] = masked_logit
                else:
                    acc_S[i] = neg_inf
    else:
        for i in cutlass.range(cute.size(acc_S), unroll_full=True):
            n_idx = n_start + tScS[i][1]
            m_idx = m_start + tScS[i][0]
            if (m_idx < seqlen) & (n_idx < seqlen):
                if mMask_w[m_idx, n_idx] != zero_u8:
                    acc_S[i] = masked_logit
            else:
                acc_S[i] = neg_inf


def swin_attn_mask_u8(bias: torch.Tensor) -> torch.Tensor:
    """Pack float Swin bias ``(nW, N, N)`` into uint8 mask for CuTe kernels."""
    return (bias < WINDOW_ATTN_MASK_THRESHOLD).to(WINDOW_ATTN_MASK_U8_DTYPE)


@dataclass
class WindowOnlineSoftmax:
    """Online softmax state for tiled KV; compatible with SDPA scale in log2 domain."""

    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor

    @staticmethod
    def create(scale_log2: Float32, num_rows: cutlass.Constexpr[int]):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return WindowOnlineSoftmax(scale_log2, num_rows, row_max, row_sum)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        use_fastmath: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_rmem_tensor_like(self.row_max, Float32)
        row_max, row_sum = self.row_max, self.row_sum
        scale_log2 = self.scale_log2
        neg_inf = -Float32.inf
        zero = Float32.zero

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()

            row_max_cur = _fmax_reduce_row(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
            )
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if row_max_cur == neg_inf:
                row_sum[r] = zero
                row_scale[r] = 1.0
                acc_S_mn[r, None].store(acc_S_row * zero)
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled,
                    fastmath=use_fastmath,
                )
                if cutlass.const_expr(is_first):
                    acc_S_row_sum = _fadd_reduce_row(acc_S_row_exp, init_val=None)
                    row_scale[r] = 1.0
                else:
                    if row_max_prev == neg_inf:
                        row_scale[r] = zero
                    else:
                        row_scale[r] = cute.math.exp2(
                            (row_max_prev - row_max_cur) * scale_log2,
                            fastmath=use_fastmath,
                        )
                    acc_S_row_sum = _fadd_reduce_row(
                        acc_S_row_exp, init_val=row_sum[r] * row_scale[r]
                    )

                row_sum[r] = acc_S_row_sum
                acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self,
        final_scale: Float32 = 1.0,
        use_fastmath: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        row_sum, row_max = self.row_sum, self.row_max
        scale_log2 = self.scale_log2
        rs = row_sum.load()
        res = cute.make_rmem_tensor(rs.shape, Float32)
        res.store(rs)
        for i in cutlass.range_constexpr(cute.size(res.shape)):
            v = res[i]
            v = v + cute.arch.shuffle_sync_bfly(v, offset=1)
            v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
            res[i] = v
        row_sum.store(res.load())
        row_scale = cute.make_rmem_tensor_like(row_max, Float32)
        LN2 = math.log(2.0)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=use_fastmath))
                * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
