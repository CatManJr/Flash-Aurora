"""BF16 window attention v2: SM120 heterogeneous TMA pipeline (CuTeDSL).

Architecture: 4 MMA consumer warps + 1 DMA producer warp = 160 threads.
Pattern mirrors blackwell_geforce/dense_gemm.py:
  - DMA warp (warp 4): produces all K/V tiles via PipelineTmaAsync
  - MMA warps (0-3): consume K/V from SMEM pipeline, run QK+PV GEMM, write O

Key CuTeDSL constraint: the DSL's if/else requires all live Python objects at the
join point to be representable as IR values.  SharedStorage is a Python class that
cannot be represented as such.  To avoid this, we use two *separate* if-statements
(no explicit else) and pre-compute all SharedStorage-derived tensors (sO, etc.)
before any dynamic branch.  sync_threads() calls are placed in the flat kernel
body between the two if-blocks so every thread hits the SAME program-counter
location, which is what CTA-wide barriers require.
"""
import math
from typing import Optional

import torch

try:
    import cuda.bindings.driver as cuda  # noqa: F401

    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as cutlass_utils
    import cutlass.utils.hopper_helpers as sm90_utils
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
    from ._window_softmax import WindowOnlineSoftmax
    from ._smem_utils import _choose_tile_n

    _CUTE_AVAILABLE = True
    _bf16_v2_compile_cache: dict = {}

except ImportError:
    _CUTE_AVAILABLE = False
    _bf16_v2_compile_cache = {}


class WindowAttnFwdBf16V2:
    """BF16 forward window attention — SM120 heterogeneous TMA pipeline."""

    _NUM_THREADS: int = 128
    _SM120_MMA_WARPS: int = 4
    _SM120_DMA_WARPS: int = 1
    _SM120_THREADS: int = (_SM120_MMA_WARPS + _SM120_DMA_WARPS) * 32   # 160
    _SM120_MMA_ATOM_LAYOUT: tuple[int, int, int] = (2, 2, 1)
    _SM120_LOAD_REGS: int = 40
    _SM120_MMA_REGS: int = 232

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
        sm120_tma_tiler = (self.tile_m, self.tile_n, self.tile_hdim)
        sK_tma_layout = sm90_utils.make_smem_layout_b(
            cutlass_utils.LayoutEnum.from_tensor(mK_t),
            sm120_tma_tiler,
            BFloat16,
            num_stages,
        )
        sV_tma_layout = sm90_utils.make_smem_layout_b(
            cutlass_utils.LayoutEnum.from_tensor(mV_t),
            sm120_tma_tiler,
            BFloat16,
            num_stages,
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

        tma_atom_K, tma_tensor_K = self._make_tma_atom_and_tensor(
            mK_t, sK_tma_layout, (self.tile_n, self.tile_hdim)
        )
        tma_atom_V, tma_tensor_V = self._make_tma_atom_and_tensor(
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

    @staticmethod
    def _make_tma_atom_and_tensor(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
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
        tma_copy_bytes = cute.size_in_bytes(
            BFloat16, cute.slice_(sK_tma_layout, (None, None, 0))
        ) + cute.size_in_bytes(BFloat16, cute.slice_(sV_tma_layout, (None, None, 0)))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self._SM120_MMA_WARPS
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
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

        # ── Q LOAD ──────────────────────────────────────────────────────────────
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

        # ── WARP DIVERGENCE — TWO SEPARATE IF-STATEMENTS ────────────────────────
        #
        # The DSL requires that values at the join point of an if/else are
        # representable as IR values.  SharedStorage is a Python class that fails
        # this requirement.  Using two *separate* if-statements (no explicit else)
        # side-steps the join-point analysis entirely.
        #
        # sync_threads() calls are placed in the FLAT kernel body *between* the
        # two if-blocks so every thread hits the EXACT SAME program-counter
        # location — a requirement for correct CTA-wide barriers in CUDA.
        #
        # Barrier counting (must match between the two branches):
        #   Single-pass: DMA=(SYNC-3 + SYNC-4) = 2, MMA=(SYNC-3 + SYNC-4) = 2
        #   Multi-pass:  DMA=(SYNC-4) = 1,          MMA=(SYNC-4) = 1

        # ── DMA PRODUCER WARP (warp 4) ──────────────────────────────────────────
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

        # ── MMA CONSUMER WARPS (warps 0-3) ──────────────────────────────────────
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
                mBias_w = mBias[win_id, None, None]

            # cS/tScS: coordinate tensors for OOB masking and/or attention bias.
            if cutlass.const_expr(self.seq_len % self.tile_n != 0 or mBias is not None):
                cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS = thr_mma_qk.partition_C(cS)

            # ── TILE 0 ──────────────────────────────────────────────────────────
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
                        acc_S[i] = acc_S[i] + (
                            mBias_w[m_safe, n_safe] if both else Float32(0.0)
                        )

            softmax.online_softmax(acc_S, is_first=True, check_inf=True)

            acc_S_bf16 = cute.make_rmem_tensor_like(acc_S, BFloat16)
            acc_S_bf16.store(acc_S.load().to(BFloat16))
            tOrP = layout_utils.reshape_acc_to_frgA(acc_S_bf16)
            gemm_rs(
                tiled_mma_pv, acc_O, tOrP, tOrVt,
                tOsVt[None, None, None, stage_cur], smem_thr_cp_V,
            )

            # Release stage before the flat-body SYNC-3 (single-pass) so that
            # producer_tail can unblock the DMA warp — avoids deadlock with SYNC-3.
            mainloop_pipeline.consumer_release(mainloop_consumer_state)
            mainloop_consumer_state.advance()

            # ── MULTI-PASS LOOP ──────────────────────────────────────────────────
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
                        tiled_mma_pv, acc_O, tOrP, tOrVt,
                        tOsVt[None, None, None, stage_cur], smem_thr_cp_V,
                    )
                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

            # ── EPILOGUE (part 1): registers → SMEM ─────────────────────────────
            final_row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, final_row_scale)

            rO = cute.make_rmem_tensor_like(acc_O, BFloat16)
            rO.store(acc_O.load().to(BFloat16))

            # Write O from registers to shared memory (stmatrix).
            # sO is pre-computed (points into sQ SMEM region, no 'storage' ref here).
            taccOrO = smem_thr_cp_O.retile(rO)
            taccOsO = smem_thr_cp_O.partition_D(sO)
            cute.copy(smem_cp_O, taccOrO, taccOsO)

        # ── FLAT BODY: epilogue sync_threads (all 160 threads hit same PC) ───────
        # SYNC-3 (single-pass only): guards the sV/sQ SMEM region between the PV
        # GEMM reads and the epilogue stmatrix writes (both happen inside the MMA
        # if-block above). DMA warp reaches here right after producer_tail.
        if cutlass.const_expr(self.single_kv_tile):
            cute.arch.sync_threads()  # SYNC-3
        # SYNC-4: ensures all stmatrix (O→SMEM) writes are globally visible
        # before any thread reads sO for the global copy below.
        cute.arch.sync_threads()  # SYNC-4

        # ── EPILOGUE (part 2): SMEM → global (MMA warps only) ───────────────────
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


def _get_or_compile_bf16_v2(
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
    compile_key = (head_dim, seq_len, has_bias, tile_m, tile_n)
    if compile_key in _bf16_v2_compile_cache:
        return _bf16_v2_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdBf16V2(
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
    _bf16_v2_compile_cache[compile_key] = compiled
    return compiled
