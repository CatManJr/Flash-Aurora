"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

BF16 window attention (CuTeDSL, SM80+).

WindowAttnFwdBf16: 128-thread cp.async, single KV tile (typical N=144).
WindowAttnFwdBf16Stream: 160-thread TMA path when tile_n < seq_len.
uint8 Swin mask read directly from gmem (L2-resident) in both kernels.

References:
- flash-attn ``flash_attn/cute/flash_fwd.py`` (Tri Dao) - FMHA mainloop / masking layout.
- :mod:`aurora.ops.cute._blackwell_load` - CUTLASS Blackwell GeForce TMA example (see module doc).
- :mod:`aurora.ops.cute._cute_local`, :mod:`aurora.ops.cute._window_softmax`.
"""
import math
import os
from typing import Optional

import torch


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

try:
    import cuda.bindings.driver as cuda  # noqa: F401

    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cutlass import Constexpr, Float32, BFloat16, Int32, Int64
    from cutlass.cute.nvgpu import cpasync, warp
    from cutlass.cutlass_dsl import BaseDSL

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
    from ._window_softmax import (
        WINDOW_ATTN_MASKED_BIAS,
        WindowOnlineSoftmax,
        apply_partial_kv_mask,
        apply_swin_mask_u8_gmem,
    )
    from ._smem_utils import _choose_tile_n
    from ._blackwell_load import (
        make_kv_mainloop_pipeline,
        make_kv_tma_smem_layouts,
        make_tma_atom_and_tensor,
    )

    _CUTE_AVAILABLE = True
    _bf16_compile_cache: dict = {}
    _bf16_qkvpacked_compile_cache: dict = {}
    _bf16_stream_compile_cache: dict = {}

except ImportError:
    _CUTE_AVAILABLE = False
    _bf16_compile_cache = {}
    _bf16_qkvpacked_compile_cache = {}
    _bf16_stream_compile_cache = {}


class WindowAttnFwdBf16:
    """BF16 forward window attention on SM80+."""

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
            tile_n = _choose_tile_n(seq_len, head_dim=head_dim, tile_m=tile_m)

        self.head_dim = head_dim
        self.seq_len = seq_len
        self.has_bias = has_bias
        self.num_threads = self._NUM_THREADS
        self.tile_m = min(tile_m, seq_len)
        self.tile_n = min(tile_n, seq_len)
        self.single_kv_tile = self.tile_n >= seq_len
        # Single-pass: 1 stage; K+V preloaded with Q, one cp.async wait before QK.
        # Multi-pass: 2-stage K+V double-buffer - prefetch K[n+1]+V[n+1] behind compute.
        self.num_stages = 1 if self.single_kv_tile else 2

        hdim_align = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_align) * hdim_align)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.dtype = BFloat16
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
        mQ_t, mK_t, mV_t, mO_t = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=_tr)) for t in (mQ, mK, mV, mO)
        ]

        N = mQ.shape[2]
        H = mQ.shape[1]
        Bwin = mQ.shape[0]
        num_m_blocks = (N + self.tile_m - 1) // self.tile_m

        num_stages = self.num_stages
        sQK_atom = get_smem_layout_atom(BFloat16, self.tile_hdim)
        sV_atom = get_smem_layout_atom(BFloat16, self.tile_hdim)

        sQ_layout = cute.tile_to_shape(sQK_atom, (self.tile_m, self.tile_hdim), (0, 1))
        sK_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        sV_layout = cute.tile_to_shape(sV_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2))
        sO_layout = cute.tile_to_shape(sV_atom, (self.tile_m, self.tile_hdim), (0, 1))

        _mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        num_warps = self.num_threads // 32
        _mma_args = dict(permutation_mnk=(num_warps * 16, 16, 16))
        tiled_mma_qk = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)
        tiled_mma_pv = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)

        _bits = 128
        _elems = _bits // BFloat16.width
        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            BFloat16,
            num_bits_per_copy=_bits,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=_bits
        )

        _sQK_dim1 = sQK_atom.outer.shape[1] // _elems
        _sV_dim1 = sV_atom.outer.shape[1] // _elems
        vQKV = cute.make_layout((1, _elems))

        def _tv(dim1):
            return cute.make_ordered_layout((self.num_threads // dim1, dim1), order=(1, 0))

        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async, _tv(_sV_dim1), vQKV)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_store, _tv(_sV_dim1), vQKV)

        sQ_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sQ_layout)], 1024]
        sK_struct = cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sK_layout)], 1024]
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
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
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
        # Preload V[0] with K[0] in the same commit group (single- and multi-pass).
        # PV can start right after QK+softmax with no extra V async wait/sync.
        self._load_V(
            gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
            n_block=0, seqlen=seqlen, smem_stage=0,
        )
        cute.arch.cp_async_commit_group()

        # cS/tScS needed for OOB masking (partial last tile) and/or bias addition.
        if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)

        # One wait for Q+K+V (replaces wait_group(1)+sync then wait_group(0)+sync).
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if cutlass.const_expr(mBias is not None):
            nW = mBias.shape[0]
            win_id = bwin_id % nW
            mMask_w = mBias[win_id, None, None]
            mask_bias_unscaled = Float32(WINDOW_ATTN_MASKED_BIAS * math.sqrt(self.head_dim))

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
            # Multi-pass: also prefetch V[1] so the next iteration has both K+V ready.
            if cutlass.const_expr(not self.single_kv_tile):
                self._load_V(
                    gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                    n_block=1, seqlen=seqlen, smem_stage=1,
                )
            cute.arch.cp_async_commit_group()

        # Apply OOB masking for n >= seqlen (partial last tile) and/or bias.
        # For the first tile n_block=0, n_start=0.
        if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
            m_start = m_block * self.tile_m
            if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                apply_partial_kv_mask(acc_S, tScS, Int32(0), seqlen)
            if cutlass.const_expr(mBias is not None):
                apply_swin_mask_u8_gmem(
                    acc_S, tScS, mMask_w, m_start, Int32(0), seqlen,
                    mask_bias_unscaled,
                    n_always_valid=self.single_kv_tile,
                    rows_all_valid=(m_start + self.tile_m <= seqlen),
                )

        row_scale = softmax.online_softmax(
            acc_S, is_first=True, use_fastmath=self.single_kv_tile
        )

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
        # Epilogue writes sO (reuses sQ); PV only reads sV - separate smem regions, no CTA sync.

        if cutlass.const_expr(not self.single_kv_tile):
            for n_tile in cutlass.range(n_block_max - 1, unroll=1):
                n_block = n_tile + 1
                # Block n is always in stage (n % 2): established at prefetch time.
                stage_cur = n_block % 2
                stage_nxt = (n_block + 1) % 2

                # K[n]+V[n] were prefetched into stage_cur; wait for both.
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()  # 1 sync/pass (was 3)

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

                # Prefetch K[n+1]+V[n+1] into stage_nxt - overlaps with QK GEMM+softmax.
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
                        gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                        n_block=n_block + 1, seqlen=seqlen, smem_stage=stage_nxt,
                    )
                    cute.arch.cp_async_commit_group()

                # Apply OOB masking for n >= seqlen (partial last tile) and/or bias.
                if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
                    m_start = m_block * self.tile_m
                    n_start = n_block * self.tile_n
                    if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                        apply_partial_kv_mask(acc_S, tScS, n_start, seqlen)
                    if cutlass.const_expr(mBias is not None):
                        apply_swin_mask_u8_gmem(
                            acc_S, tScS, mMask_w, m_start, n_start, seqlen,
                            mask_bias_unscaled,
                        )

                row_scale = softmax.online_softmax(
                    acc_S, is_first=False, use_fastmath=self.single_kv_tile
                )
                softmax.rescale_O(acc_O, row_scale)

                # PV GEMM with V[n] in stage_cur (already in SMEM; no extra wait).
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
                # No sync_threads(): stage_cur and stage_nxt are different SMEM buffers.

        final_row_scale = softmax.finalize(use_fastmath=self.single_kv_tile)
        softmax.rescale_O(acc_O, final_row_scale)

        rO = cute.make_rmem_tensor_like(acc_O, BFloat16)
        rO.store(acc_O.load().to(BFloat16))

        arch_v = self.arch.major * 10 + self.arch.minor
        smem_cp_O = get_smem_store_atom(min(arch_v, 89), BFloat16)
        smem_thr_cp_O = cute.make_tiled_copy_C(smem_cp_O, tiled_mma_pv).get_slice(tidx)
        sO = storage.sQ.get_tensor(sO_layout)
        taccOrO = smem_thr_cp_O.retile(rO)
        taccOsO = smem_thr_cp_O.partition_D(sO)
        cute.copy(smem_cp_O, taccOrO, taccOsO)
        cute.arch.sync_threads()

        gO = cute.local_tile(mO_cur, blkQ, (m_block, 0))
        gmem_thr_cp_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_cp_O.partition_S(sO)
        tOrO = cute.make_rmem_tensor_like(tOsO, BFloat16)
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
        # has_partial_tile: seq_len is not a multiple of tile_n, so the last tile is
        # partial. Always use the predicated path in that case to avoid OOB gmem reads.
        has_partial_tile = cutlass.const_expr(self.seq_len % self.tile_n != 0)
        if cutlass.const_expr(need_predicates or not is_even_n or has_partial_tile):
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
        # Fast path only when tile_n divides seq_len exactly; otherwise the last tile
        # is partial and needs bounds checking to avoid OOB gmem reads.
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


def _get_or_compile_bf16(
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
    compile_key = (head_dim, seq_len, has_bias, tile_m, tile_n, "bf16_cpasync")
    if compile_key in _bf16_compile_cache:
        return _bf16_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdBf16(
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
    _bf16_compile_cache[compile_key] = compiled
    return compiled


def _get_or_compile_bf16_qkvpacked(
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
    output_layout: str = "bhnd",
):
    """Compile BF16 kernel for non-contiguous Q/K/V views derived from packed qkv.

    The kernel body is the same as the regular BF16 path, but this separate cache
    prevents a contiguous-layout compile from being reused for qkv-packed strides.
    """
    compile_key = (
        head_dim, seq_len, has_bias, tile_m, tile_n,
        output_layout, "bf16_qkvpacked",
    )
    if compile_key in _bf16_qkvpacked_compile_cache:
        return _bf16_qkvpacked_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdBf16(
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
    _bf16_qkvpacked_compile_cache[compile_key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Multipass (tile_n < seq_len)
# ---------------------------------------------------------------------------

class WindowAttnFwdBf16Stream:
    """BF16 multipass: 4 MMA warps + 1 DMA warp, TMA K/V pipeline."""

    _NUM_THREADS: int = 128
    _SM120_MMA_WARPS: int = 4
    _SM120_DMA_WARPS: int = 1
    _SM120_THREADS: int = (_SM120_MMA_WARPS + _SM120_DMA_WARPS) * 32   # 160
    _SM120_MMA_ATOM_LAYOUT: tuple[int, int, int] = (2, 2, 1)
    # Warp-specialization register split (setmaxregister). Tunable via env for
    # sm120 sweeps; DMA warp only issues TMA so it can shed registers to let the
    # MMA warps (and thus occupancy) grow. Defaults are the validated baseline.
    _SM120_LOAD_REGS: int = _env_int("AURORA_SM120_LOAD_REGS", 40)
    _SM120_MMA_REGS: int = _env_int("AURORA_SM120_MMA_REGS", 232)

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
            tile_n = _choose_tile_n(seq_len, head_dim=head_dim, tile_m=tile_m)

        self.head_dim = head_dim
        self.seq_len = seq_len
        self.has_bias = has_bias
        self.num_threads = self._NUM_THREADS
        self.tile_m = min(tile_m, seq_len)
        self.tile_n = min(tile_n, seq_len)
        self.single_kv_tile = self.tile_n >= seq_len
        # Single-pass: 1 stage (full SMEM, no double-buffer overhead).
        # Multi-pass: 2-stage K+V double-buffer for prefetch overlap.
        self.num_stages = 1 if self.single_kv_tile else 2

        hdim_align = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_align) * hdim_align)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.dtype = BFloat16
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        self.use_sm120_hetero_pipeline = (
            self.arch.major * 10 + self.arch.minor == 120
            and self.head_dim == 64
            and self.tile_hdim == 64
        )
        # SM120: 4 MMA warps + 1 DMA warp = 160 threads
        if self.use_sm120_hetero_pipeline:
            self.num_threads = self._SM120_THREADS

    def _sm120_threads(self) -> int:
        return self._SM120_THREADS

    def _sm120_mma_threads(self) -> int:
        return self._SM120_MMA_WARPS * 32

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
        mQ_t, mK_t, mV_t, mO_t = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=_tr))
            for t in (mQ, mK, mV, mO)
        ]

        N = mQ.shape[2]
        H = mQ.shape[1]
        Bwin = mQ.shape[0]
        num_m_blocks = (N + self.tile_m - 1) // self.tile_m

        num_stages = self.num_stages
        sQK_atom = get_smem_layout_atom(BFloat16, self.tile_hdim)
        sV_atom = get_smem_layout_atom(BFloat16, self.tile_hdim)

        sQ_layout = cute.tile_to_shape(sQK_atom, (self.tile_m, self.tile_hdim), (0, 1))
        sO_layout = cute.tile_to_shape(sV_atom, (self.tile_m, self.tile_hdim), (0, 1))
        sK_tma_layout, sV_tma_layout = make_kv_tma_smem_layouts(
            mK_t, mV_t, self.tile_m, self.tile_n, self.tile_hdim,
            BFloat16, num_stages,
        )

        _mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        # MMA tiling is fixed to 4 consumer warps, regardless of total CTA size.
        num_mma_warps = self._SM120_MMA_WARPS
        _mma_args = dict(permutation_mnk=(num_mma_warps * 16, 16, 16))
        tiled_mma_qk = cute.make_tiled_mma(_mma_op, (num_mma_warps, 1, 1), **_mma_args)
        tiled_mma_pv = cute.make_tiled_mma(_mma_op, (num_mma_warps, 1, 1), **_mma_args)

        _bits = 128
        _elems = _bits // BFloat16.width
        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            BFloat16,
            num_bits_per_copy=_bits,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=_bits
        )

        _sQK_dim1 = sQK_atom.outer.shape[1] // _elems
        _sV_dim1 = sV_atom.outer.shape[1] // _elems
        vQKV = cute.make_layout((1, _elems))

        def _tv(dim1):
            # Q and O gmem copies only involve the 4 MMA consumer warps (128 threads).
            return cute.make_ordered_layout(
                (self._sm120_mma_threads() // dim1, dim1), order=(1, 0)
            )

        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_store, _tv(_sV_dim1), vQKV)

        tma_atom_K, tma_tensor_K = make_tma_atom_and_tensor(
            mK_t, sK_tma_layout, (self.tile_n, self.tile_hdim)
        )
        tma_atom_V, tma_tensor_V = make_tma_atom_and_tensor(
            mV_t, sV_tma_layout, (self.tile_n, self.tile_hdim)
        )

        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sQ_layout)], 1024
        ]
        sK_tma_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sK_tma_layout)], 1024
        ]
        sV_tma_struct = cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(sV_tma_layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[Int64, num_stages * 2]
            sQ: sQ_struct
            sK_tma: sK_tma_struct
            sV_tma: sV_tma_struct

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
            sO_layout,
            sK_tma_layout,
            sV_tma_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
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
        sO_layout: cute.ComposedLayout,
        sK_tma_layout: cute.ComposedLayout,
        sV_tma_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        tma_tensor_V: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, by, _ = cute.arch.block_idx()

        head_id = by % H
        bwin_id = by // H

        mQ_cur = mQ[None, None, head_id, bwin_id]
        mO_cur = mO[None, None, head_id, bwin_id]

        blkQ = (self.tile_m, self.tile_hdim)
        blkKV = (self.tile_n, self.tile_hdim)

        gQ = cute.local_tile(mQ_cur, blkQ, (m_block, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Combined K+V byte-count drives the TMA transaction barrier.
        mainloop_pipeline = make_kv_mainloop_pipeline(
            self.num_stages,
            self._SM120_MMA_WARPS,
            sK_tma_layout,
            sV_tma_layout,
            storage.mainloop_pipeline_array_ptr.data_ptr(),
            BFloat16,
        )
        pipeline.sync(barrier_id=1)

        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK_tma.get_tensor(sK_tma_layout.outer, swizzle=sK_tma_layout.inner)
        sV = storage.sV_tma.get_tensor(sV_tma_layout.outer, swizzle=sV_tma_layout.inner)
        sVt = layout_utils.transpose_view(sV)

        # Pre-compute sO before any dynamic branch: the DSL cannot carry the
        # SharedStorage Python class through an if/else join point because it is
        # not representable as an MLIR IR value.  Computing sO here converts
        # the struct-field reference into a plain cute.Tensor (IR value) so
        # the branches never need to access 'storage' or 'SharedStorage' directly.
        sO = storage.sQ.get_tensor(sO_layout)

        gK_tma = cute.local_tile(tma_tensor_K, blkKV, (None, 0, None, None))
        gV_tma = cute.local_tile(tma_tensor_V, blkKV, (None, 0, None, None))
        tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(gK_tma, 0, 2),
        )
        tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(gV_tma, 0, 2),
        )

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        n_block_max = (seqlen + self.tile_n - 1) // self.tile_n

        # Pre-compute O epilogue atoms (all threads; DMA warp computes but never uses).
        # This avoids SharedStorage / tiled_mma references inside the dynamic branches.
        arch_v = self.arch.major * 10 + self.arch.minor
        smem_cp_O = get_smem_store_atom(min(arch_v, 89), BFloat16)
        smem_thr_cp_O = cute.make_tiled_copy_C(smem_cp_O, tiled_mma_pv).get_slice(tidx)
        gO = cute.local_tile(mO_cur, blkQ, (m_block, 0))
        gmem_thr_cp_O = gmem_tiled_copy_O.get_slice(tidx)

        # -- Q LOAD --------------------------------------------------------------
        # gmem_tiled_copy_Q is sized for 128 threads (MMA warps) only.
        # DMA warp (warp 4, tidx 128-159) must NOT call get_slice(tidx): the TV
        # layout wraps around and would double-write Q elements.  Instead, the DMA
        # warp issues a null cp.async commit to keep the group-count balanced.
        if warp_idx < self._SM120_MMA_WARPS:
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            tQsQ = gmem_thr_copy_Q.partition_D(sQ)
            tQgQ = gmem_thr_copy_Q.partition_S(gQ)
            cQ = cute.make_identity_tensor(blkQ)
            tQcQ = gmem_thr_copy_Q.partition_S(cQ)
            t0QcQ = gmem_tiled_copy_Q.get_slice(0).partition_S(cQ)
            tQpQ = predicate_k(tQcQ, limit=self.head_dim)
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if t0QcQ[0, m, 0][0] < seqlen - m_block * self.tile_m - tQcQ[0][0]:
                    cute.copy(
                        gmem_thr_copy_Q,
                        tQgQ[None, m, None],
                        tQsQ[None, m, None],
                        pred=tQpQ[None, m, None]
                        if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
        cute.arch.cp_async_commit_group()   # MMA: Q group; DMA: empty group (NOP)
        cute.arch.cp_async_wait_group(1)    # NOP for DMA warp (no pending async ops)
        cute.arch.sync_threads()            # SYNC-1: all 160 threads

        cute.arch.cp_async_wait_group(0)    # Q fully arrived for MMA warps; NOP for DMA
        cute.arch.sync_threads()            # SYNC-2: all 160 threads

        # -- WARP DIVERGENCE - TWO SEPARATE IF-STATEMENTS ------------------------
        #
        # The DSL requires that values at the join point of an if/else are
        # representable as IR values.  SharedStorage is a Python class that fails
        # this requirement.  Using two *separate* if-statements (no explicit else)
        # side-steps the join-point analysis entirely.
        #
        # sync_threads() calls are placed in the FLAT kernel body *between* the
        # two if-blocks so every thread hits the EXACT SAME program-counter
        # location - a requirement for correct CTA-wide barriers in CUDA.
        #
        # Barrier counting (must match between the two branches):
        #   Single-pass: DMA=(SYNC-3 + SYNC-4) = 2, MMA=(SYNC-3 + SYNC-4) = 2
        #   Multi-pass:  DMA=(SYNC-4) = 1,          MMA=(SYNC-4) = 1

        # -- DMA PRODUCER WARP (warp 4) ------------------------------------------
        if warp_idx == self._SM120_MMA_WARPS:
            cute.arch.setmaxregister_decrease(self._SM120_LOAD_REGS)
            for n in cutlass.range(n_block_max, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK_tma[(None, n, head_id, bwin_id)],
                    tKsK_tma[(None, mainloop_producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_V,
                    tVgV_tma[(None, n, head_id, bwin_id)],
                    tVsV_tma[(None, mainloop_producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()
            # Wait until all consumers have released every stage.
            mainloop_pipeline.producer_tail(mainloop_producer_state)

        # -- MMA CONSUMER WARPS (warps 0-3) --------------------------------------
        if warp_idx < self._SM120_MMA_WARPS:
            cute.arch.setmaxregister_increase(self._SM120_MMA_REGS)

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
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
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

            if cutlass.const_expr(mBias is not None):
                nW = mBias.shape[0]
                win_id = bwin_id % nW
                mMask_w = mBias[win_id, None, None]
                mask_bias_unscaled = Float32(WINDOW_ATTN_MASKED_BIAS * math.sqrt(self.head_dim))

            # cS/tScS: coordinate tensors for OOB masking and/or attention bias.
            if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
                cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS = thr_mma_qk.partition_C(cS)

            # -- TILE 0 ----------------------------------------------------------
            mainloop_pipeline.consumer_wait(mainloop_consumer_state)
            stage_cur = mainloop_consumer_state.index

            acc_S = cute.make_rmem_tensor(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            )
            acc_S.fill(0.0)
            gemm(
                tiled_mma_qk, acc_S, tSrQ, tSrK,
                tSsQ, tSsK[None, None, None, stage_cur],
                smem_thr_cp_Q, smem_thr_cp_K,
            )

            if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
                m_start = m_block * self.tile_m
                if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                    apply_partial_kv_mask(acc_S, tScS, Int32(0), seqlen)
                if cutlass.const_expr(mBias is not None):
                    apply_swin_mask_u8_gmem(
                        acc_S, tScS, mMask_w, m_start, Int32(0), seqlen,
                        mask_bias_unscaled,
                        n_always_valid=self.single_kv_tile,
                        rows_all_valid=(m_start + self.tile_m <= seqlen),
                    )

            softmax.online_softmax(
                acc_S, is_first=True, use_fastmath=self.single_kv_tile
            )

            acc_S_bf16 = cute.make_rmem_tensor_like(acc_S, BFloat16)
            acc_S_bf16.store(acc_S.load().to(BFloat16))
            tOrP = layout_utils.reshape_acc_to_frgA(acc_S_bf16)
            gemm_rs(
                tiled_mma_pv, acc_O, tOrP, tOrVt,
                tOsVt[None, None, None, stage_cur], smem_thr_cp_V,
            )

            # Release stage before the flat-body SYNC-3 (single-pass) so that
            # producer_tail can unblock the DMA warp - avoids deadlock with SYNC-3.
            mainloop_pipeline.consumer_release(mainloop_consumer_state)
            mainloop_consumer_state.advance()

            # -- MULTI-PASS LOOP --------------------------------------------------
            if cutlass.const_expr(not self.single_kv_tile):
                for n_tile in cutlass.range(n_block_max - 1, unroll=1):
                    n_block = n_tile + 1

                    mainloop_pipeline.consumer_wait(mainloop_consumer_state)
                    stage_cur = mainloop_consumer_state.index

                    acc_S = cute.make_rmem_tensor(
                        thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
                    )
                    acc_S.fill(0.0)
                    gemm(
                        tiled_mma_qk, acc_S, tSrQ, tSrK,
                        tSsQ, tSsK[None, None, None, stage_cur],
                        smem_thr_cp_Q, smem_thr_cp_K,
                    )

                    if cutlass.const_expr(
                        self.seq_len % self.tile_n != 0 or mBias is not None
                    ):
                        m_start = m_block * self.tile_m
                        n_start = n_block * self.tile_n
                        if cutlass.const_expr(self.seq_len % self.tile_n != 0):
                            apply_partial_kv_mask(acc_S, tScS, n_start, seqlen)
                        if cutlass.const_expr(mBias is not None):
                            apply_swin_mask_u8_gmem(
                                acc_S, tScS, mMask_w, m_start, n_start, seqlen,
                                mask_bias_unscaled,
                            )

                    row_scale = softmax.online_softmax(
                        acc_S, is_first=False, use_fastmath=self.single_kv_tile
                    )
                    softmax.rescale_O(acc_O, row_scale)

                    acc_S_bf16 = cute.make_rmem_tensor_like(acc_S, BFloat16)
                    acc_S_bf16.store(acc_S.load().to(BFloat16))
                    tOrP = layout_utils.reshape_acc_to_frgA(acc_S_bf16)
                    gemm_rs(
                        tiled_mma_pv, acc_O, tOrP, tOrVt,
                        tOsVt[None, None, None, stage_cur], smem_thr_cp_V,
                    )
                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

            # -- EPILOGUE (part 1): registers -> SMEM -----------------------------
            final_row_scale = softmax.finalize(use_fastmath=self.single_kv_tile)
            softmax.rescale_O(acc_O, final_row_scale)

            rO = cute.make_rmem_tensor_like(acc_O, BFloat16)
            rO.store(acc_O.load().to(BFloat16))

            # Write O from registers to shared memory (stmatrix).
            # sO is pre-computed (points into sQ SMEM region, no 'storage' ref here).
            taccOrO = smem_thr_cp_O.retile(rO)
            taccOsO = smem_thr_cp_O.partition_D(sO)
            cute.copy(smem_cp_O, taccOrO, taccOsO)

        # -- FLAT BODY: epilogue sync_threads (all 160 threads hit same PC) -------
        # SYNC-3 (single-pass only): guards the sV/sQ SMEM region between the PV
        # GEMM reads and the epilogue stmatrix writes (both happen inside the MMA
        # if-block above). DMA warp reaches here right after producer_tail.
        if cutlass.const_expr(self.single_kv_tile):
            cute.arch.sync_threads()  # SYNC-3
        # SYNC-4: ensures all stmatrix (O->SMEM) writes are globally visible
        # before any thread reads sO for the global copy below.
        cute.arch.sync_threads()  # SYNC-4

        # -- EPILOGUE (part 2): SMEM -> global (MMA warps only) -------------------
        # DMA warp skips this block; it has nothing more to do.
        if warp_idx < self._SM120_MMA_WARPS:
            tOsO = gmem_thr_cp_O.partition_S(sO)
            tOrO = cute.make_rmem_tensor_like(tOsO, BFloat16)
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


def _get_or_compile_bf16_stream(
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
    compile_key = (
        head_dim, seq_len, has_bias, tile_m, tile_n, "bf16_stream",
        WindowAttnFwdBf16Stream._SM120_LOAD_REGS,
        WindowAttnFwdBf16Stream._SM120_MMA_REGS,
    )
    if compile_key in _bf16_stream_compile_cache:
        return _bf16_stream_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdBf16Stream(
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
    _bf16_stream_compile_cache[compile_key] = compiled
    return compiled
