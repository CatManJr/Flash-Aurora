# Copyright (c) 2025, Tri Dao.  (gemm helpers — same as upstream flash-attn ampere_helpers.)
# SPDX-License-Identifier: MIT
"""Minimal CuTeDSL helpers for window attention (no flash_attn dependency).

Torch ↔ CuTe bridge, SMEM swizzle atom, and SMEM GEMM loops (Ampere-style MMA).
"""
from __future__ import annotations

from typing import Callable, Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def assume_strides_aligned(t: cute.Tensor):
    divby = 128 // t.element_type.width
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1]
    )
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t: cute.Tensor | None) -> cute.Tensor | None:
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(
    t: torch.Tensor,
    *,
    assumed_align: int = 16,
    leading_dim: int = -1,
    enable_tvm_ffi: bool = True,
) -> cute.Tensor:
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        cutlass.const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size), order=(1, 0)
        ),
    )


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if cutlass.const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if cutlass.const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    """C-side store from MMA accumulators; use arch<=89 for CopyUniversal (no StMatrix)."""
    if cutlass.const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    return cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
        element_type,
    )


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(swap_AB):
        gemm(
            tiled_mma,
            acc,
            tCrB,
            tCrA,
            tCsB,
            tCsA,
            smem_thr_copy_B,
            smem_thr_copy_A,
            hook_fn,
            A_in_regs=B_in_regs,
            B_in_regs=A_in_regs,
            swap_AB=False,
        )
    else:
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        if cutlass.const_expr(not A_in_regs):
            cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
        if cutlass.const_expr(not B_in_regs):
            cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                if cutlass.const_expr(not A_in_regs):
                    cute.copy(
                        smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1]
                    )
                if cutlass.const_expr(not B_in_regs):
                    cute.copy(
                        smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
                    )
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            if cutlass.const_expr(k == 0 and hook_fn is not None):
                hook_fn()


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
            )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()
