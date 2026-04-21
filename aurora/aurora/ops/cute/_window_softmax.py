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
from cutlass import Float32

from quack import layout_utils


@cute.jit
def _fmax_reduce_row(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    res = cute.make_fragment(x.shape, Float32)
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
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max, row_sum = self.row_max, self.row_sum
        scale_log2 = self.scale_log2

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()

            row_max_cur = _fmax_reduce_row(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
            )
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_sum = _fadd_reduce_row(acc_S_row_exp, init_val=None)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = _fadd_reduce_row(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r]
                )

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0) -> cute.Tensor:
        row_sum, row_max = self.row_sum, self.row_max
        scale_log2 = self.scale_log2
        rs = row_sum.load()
        res = cute.make_fragment(rs.shape, Float32)
        res.store(rs)
        for i in cutlass.range_constexpr(cute.size(res.shape)):
            v = res[i]
            v = v + cute.arch.shuffle_sync_bfly(v, offset=1)
            v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
            res[i] = v
        row_sum.store(res.load())
        row_scale = cute.make_fragment_like(row_max, Float32)
        LN2 = math.log(2.0)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
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
