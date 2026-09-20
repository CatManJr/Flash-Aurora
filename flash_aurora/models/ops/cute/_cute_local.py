"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Minimal CuTeDSL helpers for window attention (vendored; no flash_attn import).

Torch <-> CuTe bridge, SMEM swizzle atom, and SMEM GEMM loops (Ampere-style MMA).

References:
- flash-attn ``flash_attn/cute/ampere_helpers.py`` (Tri Dao) - SMEM layout atoms and GEMM loops.
- CUTLASS CuTe DSL (NVIDIA) - ``cutlass.cute`` / ``cutlass.utils`` APIs.
"""
from __future__ import annotations

from typing import Callable, Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, TFloat32
from cutlass.cute.runtime import from_dlpack

# TF32 keeps 11 of the 24 significand bits of an FP32 value, so clearing the low
# 13 mantissa bits yields a value that the TF32 MMA reproduces exactly and leaves
# a residual ``x - hi`` that is itself exact in FP32.  The DSL's ``TFloat32(x)``
# conversion cannot be used for this: it is a pure type change that leaves the
# bits untouched (the MMA hardware does the truncation), so the mask is the only
# way to materialise the high part as a value we can subtract.
_TF32_DROPPED_MANTISSA_BITS = 13
_TF32_HI_MASK = -(1 << _TF32_DROPPED_MANTISSA_BITS)


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
def predicate_d(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    """Predicate head-dim OOB for identity tiles shaped (Dh, N)."""
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][0], limit)
    return tApA


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    tApA = cute.make_rmem_tensor(
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


@cute.jit
def split_tf32_hi_lo(frag: cute.Tensor) -> tuple[cute.Tensor, cute.Tensor]:
    """Split an FP32-valued MMA fragment into two TF32 operands ``hi`` and ``lo``.

    ``hi`` survives the TF32 MMA's truncation unchanged and ``lo = frag - hi`` is
    exact in FP32, so ``hi + lo`` reproduces the input bit for bit.  ``lo`` still
    carries 13 significant bits where TF32 holds 11; that 2-bit loss is what caps
    3xTF32 near 22 significand bits instead of FP32's 24.
    """
    # Two constraints shape this.  Allocations agreeing in both dtype and layout get
    # merged downstream, and recasting one of a merged pair changes the other, so the
    # two scratch tensors deliberately differ in element type.  And the MMA accepts
    # only tf32-typed operands, so the results are handed back as TFloat32 views
    # created last, after every read of the input has been taken.
    frag_values = cute.recast_tensor(frag, dtype=Float32).load()
    frag_bits = cute.recast_tensor(frag, dtype=Int32).load()

    hi_bits = cute.make_rmem_tensor_like(frag, Int32)
    hi_bits.store(frag_bits & Int32(_TF32_HI_MASK))

    lo_values = cute.make_rmem_tensor_like(frag, Float32)
    lo_values.store(frag_values - cute.recast_tensor(hi_bits, dtype=Float32).load())

    return (
        cute.recast_tensor(hi_bits, dtype=TFloat32),
        cute.recast_tensor(lo_values, dtype=TFloat32),
    )


@cute.jit
def relayout_acc_to_frgA_tf32(acc: cute.Tensor, tidx: Int32) -> cute.Tensor:
    """Rearrange an ``m16n8`` accumulator into the ``m16n8k8`` TF32 A-fragment layout.

    The BF16 kernel feeds its QK accumulator straight into the PV MMA because
    ``m16n8k16``'s A layout happens to coincide with ``m16n8``'s C layout.  TF32
    has no k16 shape, and under ``m16n8k8`` lane ``t`` of each four-lane group
    holds C columns ``2t`` and ``2t+1`` but needs A columns ``t`` and ``t+4``, so
    the values must cross lanes.  Column ``t`` lives on lane ``t >> 1`` of the
    group and column ``t+4`` on lane ``(t >> 1) + 2``; which of that lane's two
    registers to take is decided by the parity of ``t``.

    Both shuffles run in uniform control flow because the source lane and the
    destination lane can have different parities, so a full warp mask is required.

    The result carries ``acc``'s own dtype (FP32).  ``cute.gemm`` accepts TF32
    operands as either ``tf32`` or ``i32`` but requires A and B to agree, and
    ``make_fragment_A``/``make_fragment_B`` both hand back ``i32`` - the raw
    32-bit register view.  A caller feeding this straight into a single MMA
    therefore has to ``cute.recast_tensor(..., Int32)`` first; the 3xTF32 caller
    does not, because :func:`split_tf32_hi_lo` rebuilds both operands as
    ``TFloat32`` anyway.
    """
    lane = tidx % 32
    src_lo = (lane & ~Int32(3)) | ((lane & Int32(3)) >> 1)
    src_hi = src_lo + 2
    take_odd = (lane & Int32(1)) != 0

    frgA = cute.make_rmem_tensor_like(acc)
    for m in cutlass.range_constexpr(cute.size(acc.shape[1])):
        for n in cutlass.range_constexpr(cute.size(acc.shape[2])):
            row_lo_even = cute.arch.shuffle_sync(acc[(0, 0), m, n], src_lo)
            row_lo_odd = cute.arch.shuffle_sync(acc[(1, 0), m, n], src_lo)
            row_hi_even = cute.arch.shuffle_sync(acc[(0, 1), m, n], src_lo)
            row_hi_odd = cute.arch.shuffle_sync(acc[(1, 1), m, n], src_lo)
            col4_lo_even = cute.arch.shuffle_sync(acc[(0, 0), m, n], src_hi)
            col4_lo_odd = cute.arch.shuffle_sync(acc[(1, 0), m, n], src_hi)
            col4_hi_even = cute.arch.shuffle_sync(acc[(0, 1), m, n], src_hi)
            col4_hi_odd = cute.arch.shuffle_sync(acc[(1, 1), m, n], src_hi)

            if take_odd:
                frgA[(0, 0), m, n] = row_lo_odd
                frgA[(1, 0), m, n] = row_hi_odd
                frgA[(0, 1), m, n] = col4_lo_odd
                frgA[(1, 1), m, n] = col4_hi_odd
            else:
                frgA[(0, 0), m, n] = row_lo_even
                frgA[(1, 0), m, n] = row_hi_even
                frgA[(0, 1), m, n] = col4_lo_even
                frgA[(1, 1), m, n] = col4_hi_even
    return frgA


@cute.jit
def mma_3xtf32(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
) -> None:
    """Emulate one FP32 product with three TF32 MMAs (CUTLASS 3xTF32 / fast-F32).

    Expanding ``(a_hi + a_lo)(b_hi + b_lo)`` leaves ``a_lo*b_lo`` at a relative
    magnitude of ~2^-22, below what the FP32 accumulator can hold, so three of the
    four terms suffice.  Cross terms are issued before the dominant ``a_hi*b_hi``
    term, matching CUTLASS's ordering.
    """
    a_hi, a_lo = split_tf32_hi_lo(a)
    b_hi, b_lo = split_tf32_hi_lo(b)
    cute.gemm(tiled_mma, acc, a_lo, b_hi, acc)
    cute.gemm(tiled_mma, acc, a_hi, b_lo, acc)
    cute.gemm(tiled_mma, acc, a_hi, b_hi, acc)


@cute.jit
def gemm_3xtf32(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    """:func:`gemm` with each k-step's product emulated by :func:`mma_3xtf32`."""
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            cute.copy(
                smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1]
            )
            cute.copy(
                smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
            )
        mma_3xtf32(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k])
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()


@cute.jit
def gemm_rs_3xtf32(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    """:func:`gemm_rs` with each k-step's product emulated by :func:`mma_3xtf32`."""
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
            )
        mma_3xtf32(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k])
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()
