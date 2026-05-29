"""FP32/TF32 window attention v2: hetero/sm120 staging path (CuTeDSL).

TF32 truncates mantissa inside MMA only; softmax/accumulation stay FP32.
Epilogue: direct MMA → GMEM (no SMEM staging).
"""
# V2 NOTE: This file intentionally starts as a behavior-equivalent copy.
# TMA producer/consumer changes should follow CUTLASS DSL's
# blackwell_geforce/dense_gemm.py recipe: SM90-style PipelineTmaAsync,
# hopper_helpers swizzled SMEM layouts, one DMA warp, and four MMA warps.
import math
from dataclasses import dataclass as _dataclass
from typing import Optional

import torch

try:
    import cuda.bindings.driver as cuda  # noqa: F401

    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Constexpr, Float32, TFloat32, Int32
    from cutlass.cute.nvgpu import cpasync, warp
    from cutlass.cutlass_dsl import BaseDSL

    import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
    from cutlass.cute.atom import make_atom
    from cutlass.cute.core import _pack_shape
    from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op, MmaF16BF16Trait

    from quack import layout_utils

    from ._cute_local import (
        assume_tensor_aligned,
        gemm,
        gemm_rs,
        get_smem_layout_atom,
        get_smem_store_atom,
        make_tiled_copy_A,
        make_tiled_copy_B,
        predicate_k,
        to_cute_tensor,
    )
    from ._window_softmax import WindowOnlineSoftmax
    from ._smem_utils import _choose_tile_n_tf32

    @_dataclass(frozen=True)
    class MmaTF32Op(warp.WarpMmaOp):
        """SM80 TF32 warp MMA (``mma.sync...f32.tf32.tf32.f32``)."""

        shape_mnk: warp.Shape = (16, 8, 8)

        def _make_trait(self, *, loc=None, ip=None, **kwargs):
            shape = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
            ty = _cute_nvgpu_ir.MmaAtomSM80Type.get(
                shape.type.attribute,
                TFloat32.mlir_type,
                TFloat32.mlir_type,
                Float32.mlir_type,
            )
            return MmaF16BF16Trait(make_atom(ty, loc=loc, ip=ip))

        def _verify_fragment_A(self, inp, *, loc=None, ip=None):
            pass

        def _verify_fragment_B(self, inp, *, loc=None, ip=None):
            pass

    _CUTE_AVAILABLE = True
    _tf32_v2_compile_cache: dict = {}

except ImportError:
    _CUTE_AVAILABLE = False
    _tf32_v2_compile_cache = {}


class WindowAttnFwdTF32V2:
    """Forward-only TF32 window attention for SM80+."""

    _NUM_THREADS: int = 128

    def __init__(
        self,
        head_dim: int,
        seq_len: int,
        has_bias: bool = False,
        tile_m: int = 64,
        tile_n: Optional[int] = None,
        num_stages: int = 1,
    ):
        assert _CUTE_AVAILABLE, "CuTeDSL / cutlass / quack not found"

        if tile_n is None:
            tile_n = _choose_tile_n_tf32(seq_len, head_dim=head_dim, tile_m=tile_m)

        self.head_dim = head_dim
        self.seq_len = seq_len
        self.has_bias = has_bias
        self.num_threads = self._NUM_THREADS
        self.tile_m = min(tile_m, seq_len)
        self.tile_n = min(tile_n, seq_len)
        self.single_kv_tile = self.tile_n >= seq_len
        self.num_stages = 1 if self.single_kv_tile else 2

        hdim_align = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_align) * hdim_align)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.dtype = Float32
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        stream: cuda.CUstream = None,
    ) -> None:
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        _tr = [2, 3, 1, 0]
        mQ_t, mK_t, mO_t = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=_tr)) for t in (mQ, mK, mO)
        ]
        # V in gmem is BF16 (host cast); PV MMA uses m16n8k16 on swizzled smem like BF16 FA.
        mV_t = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=_tr))

        N = mQ.shape[2]
        H = mQ.shape[1]
        Bwin = mQ.shape[0]
        num_m_blocks = (N + self.tile_m - 1) // self.tile_m

        num_stages = self.num_stages
        sQK_atom = get_smem_layout_atom(Float32, self.tile_hdim)
        sV_atom = get_smem_layout_atom(BFloat16, self.tile_hdim)

        sQ_layout = cute.tile_to_shape(sQK_atom, (self.tile_m, self.tile_hdim), (0, 1))
        sK_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        sV_layout = cute.tile_to_shape(
            sV_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        sO_layout = cute.tile_to_shape(sQK_atom, (self.tile_m, self.tile_hdim), (0, 1))

        _mma_op_qk = MmaTF32Op()
        _mma_op_pv = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        num_warps = self.num_threads // 32
        tiled_mma_qk = cute.make_tiled_mma(
            _mma_op_qk, (num_warps, 1, 1), permutation_mnk=(num_warps * 16, 16, 8)
        )
        tiled_mma_pv = cute.make_tiled_mma(
            _mma_op_pv, (num_warps, 1, 1), permutation_mnk=(num_warps * 16, 16, 16)
        )

        _bits_f32 = 128
        _elems_f32 = _bits_f32 // Float32.width
        _bits_bf16 = 128
        _elems_bf16 = _bits_bf16 // BFloat16.width
        atom_async_f32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=_bits_f32,
        )
        atom_async_bf16 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            BFloat16,
            num_bits_per_copy=_bits_bf16,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=_bits_f32
        )

        _sQK_dim1 = sQK_atom.outer.shape[1] // _elems_f32
        _sV_dim1 = sV_atom.outer.shape[1] // _elems_bf16
        vQK = cute.make_layout((1, _elems_f32))
        vV = cute.make_layout((1, _elems_bf16))

        def _tv(dim1):
            return cute.make_ordered_layout((self.num_threads // dim1, dim1), order=(1, 0))

        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_f32, _tv(_sQK_dim1), vQK)
        gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_f32, _tv(_sQK_dim1), vQK)
        gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_bf16, _tv(_sV_dim1), vV)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_store, _tv(_sQK_dim1), vQK)

        sQ_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(sQ_layout)], 1024]
        sK_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(sK_layout)], 1024]
        sV_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sV_layout)], 1024]

        @cute.struct
        class SharedStorage:
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct

        self.kernel(
            mQ_t,
            mK_t,
            mV_t,
            mO_t,
            mBias,
            softmax_scale_log2,
            N,
            H,
            sQ_layout,
            sK_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_K,
            gmem_tiled_copy_V,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=[num_m_blocks, Bwin * H, 1],
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        seqlen: Int32,
        H: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, by, _ = cute.arch.block_idx()

        head_id = by % H
        bwin_id = by // H

        mQ_cur = mQ[None, None, head_id, bwin_id]
        mK_cur = mK[None, None, head_id, bwin_id]
        mV_cur = mV[None, None, head_id, bwin_id]
        mO_cur = mO[None, None, head_id, bwin_id]

        blkQ = (self.tile_m, self.tile_hdim)
        blkKV = (self.tile_n, self.tile_hdim)
        gQ = cute.local_tile(mQ_cur, blkQ, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkKV, (None, 0))
        gV = cute.local_tile(mV_cur, blkKV, (None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)

        tQsQ = gmem_thr_copy_Q.partition_D(sQ)
        tQgQ = gmem_thr_copy_Q.partition_S(gQ)
        tKsK = gmem_thr_copy_K.partition_D(sK)
        tKgK = gmem_thr_copy_K.partition_S(gK)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        tVgV = gmem_thr_copy_V.partition_S(gV)

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)

        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))

        acc_O = cute.make_rmem_tensor(
            thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim)), Float32
        )
        acc_O.fill(0.0)

        smem_cp_QK = cute.make_copy_atom(
            warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4), TFloat32
        )
        smem_cp_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16
        )
        smem_thr_cp_Q = make_tiled_copy_A(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_K = make_tiled_copy_B(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_V = make_tiled_copy_B(smem_cp_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_cp_Q.partition_S(sQ)
        tSsK = smem_thr_cp_K.partition_S(sK)
        tOsVt = smem_thr_cp_V.partition_S(sVt)

        n_rows = acc_O.shape[0][0] * acc_O.shape[1]
        softmax = WindowOnlineSoftmax.create(softmax_scale_log2, n_rows)
        softmax.reset()

        cQ = cute.make_identity_tensor(blkQ)
        tQcQ = gmem_thr_copy_Q.partition_S(cQ)
        t0QcQ = gmem_tiled_copy_Q.get_slice(0).partition_S(cQ)
        tQpQ = predicate_k(tQcQ, limit=self.head_dim)

        cKV = cute.make_identity_tensor(blkKV)
        tKcK = gmem_thr_copy_K.partition_S(cKV)
        t0KcK = gmem_tiled_copy_K.get_slice(0).partition_S(cKV)
        tKpK = predicate_k(tKcK, limit=self.head_dim)

        tVcV = gmem_thr_copy_V.partition_S(cKV)
        t0VcV = gmem_tiled_copy_V.get_slice(0).partition_S(cKV)
        tVpV = tKpK

        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if t0QcQ[0, m, 0][0] < seqlen - m_block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy_Q,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
        cute.arch.cp_async_commit_group()

        n_block_max = (seqlen + self.tile_n - 1) // self.tile_n

        self._load_K(
            gmem_tiled_copy_K,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            n_block=0,
            smem_stage=0,
            seqlen=seqlen,
            need_predicates=True,
        )
        if cutlass.const_expr(not self.single_kv_tile):
            self._load_V(
                gmem_tiled_copy_V,
                tVgV,
                tVsV,
                tVcV,
                t0VcV,
                tVpV,
                n_block=0,
                seqlen=seqlen,
                smem_stage=0,
            )
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        if cutlass.const_expr(mBias is not None):
            nW = mBias.shape[0]
            win_id = bwin_id % nW
            mBias_w = mBias[win_id, None, None]

        if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        acc_S = cute.make_rmem_tensor(
            thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_S.fill(0.0)

        gemm(
            tiled_mma_qk,
            acc_S,
            tSrQ,
            tSrK,
            tSsQ,
            tSsK[None, None, None, 0],
            smem_thr_cp_Q,
            smem_thr_cp_K,
        )

        if n_block_max > 1:
            self._load_K(
                gmem_tiled_copy_K,
                tKgK,
                tKsK,
                tKcK,
                t0KcK,
                tKpK,
                n_block=1,
                smem_stage=1 if cutlass.const_expr(not self.single_kv_tile) else 0,
                seqlen=seqlen,
                need_predicates=False,
            )
            if cutlass.const_expr(not self.single_kv_tile):
                self._load_V(
                    gmem_tiled_copy_V,
                    tVgV,
                    tVsV,
                    tVcV,
                    t0VcV,
                    tVpV,
                    n_block=1,
                    seqlen=seqlen,
                    smem_stage=1,
                )
            cute.arch.cp_async_commit_group()

        if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
            m_start = m_block * self.tile_m
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                n_idx = tScS[i][1]
                if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                    if n_idx >= seqlen:
                        acc_S[i] = -Float32.inf
                if cutlass.const_expr(mBias is not None):
                    m_idx = m_start + tScS[i][0]
                    m_valid = m_idx < seqlen
                    n_valid = n_idx < seqlen
                    both = m_valid & n_valid
                    m_safe = m_idx if m_valid else Int32(0)
                    n_safe = n_idx if n_valid else Int32(0)
                    acc_S[i] = acc_S[i] + (mBias_w[m_safe, n_safe] if both else Float32(0.0))

        row_scale = softmax.online_softmax(acc_S, is_first=True, check_inf=True)

        if cutlass.const_expr(self.single_kv_tile):
            self._load_V(
                gmem_tiled_copy_V,
                tVgV,
                tVsV,
                tVcV,
                t0VcV,
                tVpV,
                n_block=0,
                seqlen=seqlen,
                smem_stage=0,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        acc_S_bf16 = cute.make_rmem_tensor_like(acc_S, BFloat16)
        acc_S_bf16.store(acc_S.load().to(BFloat16))
        tOrP = layout_utils.reshape_acc_to_frgA(acc_S_bf16)

        gemm_rs(
            tiled_mma_pv,
            acc_O,
            tOrP,
            tOrVt,
            tOsVt[None, None, None, 0],
            smem_thr_cp_V,
        )
        if cutlass.const_expr(self.single_kv_tile):
            cute.arch.sync_threads()

        if cutlass.const_expr(not self.single_kv_tile):
            for n_tile in cutlass.range(n_block_max - 1, unroll=1):
                n_block = n_tile + 1
                stage_cur = n_block % 2
                stage_nxt = (n_block + 1) % 2

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                acc_S = cute.make_rmem_tensor(
                    thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
                )
                acc_S.fill(0.0)

                gemm(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    tSsQ,
                    tSsK[None, None, None, stage_cur],
                    smem_thr_cp_Q,
                    smem_thr_cp_K,
                )

                if n_block < n_block_max - 1:
                    self._load_K(
                        gmem_tiled_copy_K,
                        tKgK,
                        tKsK,
                        tKcK,
                        t0KcK,
                        tKpK,
                        n_block=n_block + 1,
                        smem_stage=stage_nxt,
                        seqlen=seqlen,
                        need_predicates=False,
                    )
                    self._load_V(
                        gmem_tiled_copy_V,
                        tVgV,
                        tVsV,
                        tVcV,
                        t0VcV,
                        tVpV,
                        n_block=n_block + 1,
                        seqlen=seqlen,
                        smem_stage=stage_nxt,
                    )
                    cute.arch.cp_async_commit_group()

                if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
                    m_start = m_block * self.tile_m
                    n_start = n_block * self.tile_n
                    for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                        n_idx = n_start + tScS[i][1]
                        if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                            if n_idx >= seqlen:
                                acc_S[i] = -Float32.inf
                        if cutlass.const_expr(mBias is not None):
                            m_idx = m_start + tScS[i][0]
                            m_valid = m_idx < seqlen
                            n_valid = n_idx < seqlen
                            both = m_valid & n_valid
                            m_safe = m_idx if m_valid else Int32(0)
                            n_safe = n_idx if n_valid else Int32(0)
                            acc_S[i] = acc_S[i] + (
                                mBias_w[m_safe, n_safe] if both else Float32(0.0)
                            )

                row_scale = softmax.online_softmax(acc_S, is_first=False, check_inf=True)
                softmax.rescale_O(acc_O, row_scale)

                acc_S_bf16 = cute.make_rmem_tensor_like(acc_S, BFloat16)
                acc_S_bf16.store(acc_S.load().to(BFloat16))
                tOrP = layout_utils.reshape_acc_to_frgA(acc_S_bf16)

                gemm_rs(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt,
                    tOsVt[None, None, None, stage_cur],
                    smem_thr_cp_V,
                )

        final_row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, final_row_scale)

        rO = cute.make_rmem_tensor_like(acc_O, Float32)
        rO.store(acc_O.load())

        arch_v = self.arch.major * 10 + self.arch.minor
        smem_cp_O = get_smem_store_atom(arch_v, Float32)
        smem_thr_cp_O = cute.make_tiled_copy_C(smem_cp_O, tiled_mma_pv).get_slice(tidx)
        sO = storage.sQ.get_tensor(sO_layout)
        taccOrO = smem_thr_cp_O.retile(rO)
        taccOsO = smem_thr_cp_O.partition_D(sO)
        cute.copy(smem_cp_O, taccOrO, taccOsO)
        cute.arch.sync_threads()

        gO = cute.local_tile(mO_cur, blkQ, (m_block, 0))
        gmem_thr_cp_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_cp_O.partition_S(sO)
        tOrO = cute.make_rmem_tensor_like(tOsO, Float32)
        cute.autovec_copy(tOsO, tOrO)
        tOgO = gmem_thr_cp_O.partition_D(gO)
        cO = cute.make_identity_tensor(blkQ)
        tOcO = gmem_thr_cp_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = predicate_k(tOcO, limit=self.head_dim)
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None]
                    if cutlass.const_expr(self.check_hdim_oob) else None,
                )

    @cute.jit
    def _load_K(
        self,
        gmem_tiled_copy_K: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        n_block: Int32,
        smem_stage: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr[bool],
    ):
        is_even_n = cutlass.const_expr(self.tile_n % gmem_tiled_copy_K.tiler_mn[0].shape == 0)
        if cutlass.const_expr(need_predicates or not is_even_n):
            seqlen_limit = seqlen - n_block * self.tile_n - tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy_K,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None, smem_stage],
                        pred=tKpK[None, n, None]
                        if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
        else:
            cute.copy(
                gmem_tiled_copy_K,
                tKgK[None, None, None, n_block],
                tKsK[None, None, None, smem_stage],
            )

    @cute.jit
    def _load_V(
        self,
        gmem_tiled_copy_V: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        n_block: Int32,
        seqlen: Int32,
        smem_stage: Int32,
    ):
        is_even_n = cutlass.const_expr(self.tile_n % gmem_tiled_copy_V.tiler_mn[0].shape == 0)
        has_partial_tile = cutlass.const_expr(self.seq_len % self.tile_n != 0)
        if cutlass.const_expr(is_even_n and not has_partial_tile):
            cute.copy(
                gmem_tiled_copy_V,
                tVgV[None, None, None, n_block],
                tVsV[None, None, None, smem_stage],
            )
        else:
            seqlen_limit = seqlen - n_block * self.tile_n - tVcV[0][0]
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if t0VcV[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy_V,
                        tVgV[None, n, None, n_block],
                        tVsV[None, n, None, smem_stage],
                        pred=tVpV[None, n, None]
                        if cutlass.const_expr(self.check_hdim_oob) else None,
                    )


def _get_or_compile_tf32_v2(
    head_dim: int,
    seq_len: int,
    has_bias: bool,
    tile_m: int,
    tile_n: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    bias_or_none: Optional[torch.Tensor],
):
    single_kv = tile_n >= seq_len
    compile_key = (head_dim, seq_len, has_bias, tile_m, tile_n, single_kv, "hybrid_pv_bf16_v4")
    if compile_key in _tf32_v2_compile_cache:
        return _tf32_v2_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdTF32V2(
        head_dim=head_dim,
        seq_len=seq_len,
        has_bias=has_bias,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    q_ct = to_cute_tensor(q)
    k_ct = to_cute_tensor(k)
    v_ct = to_cute_tensor(v)
    o_ct = to_cute_tensor(o)
    bias_ct = to_cute_tensor(bias_or_none) if bias_or_none is not None else None

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        q_ct,
        k_ct,
        v_ct,
        o_ct,
        bias_ct,
        Float32(1.0),
        stream,
        options="--enable-tvm-ffi",
    )
    _tf32_v2_compile_cache[compile_key] = compiled
    return compiled
