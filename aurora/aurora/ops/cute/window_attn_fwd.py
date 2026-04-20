"""Copyright (c) Catman Jr. Licensed under the MIT license.

Modified from flash-attn/flash_attn/cute/window_attn_fwd.py.

See flash-attn/flash_attn/cute/window_attn_fwd.py for original license.
"""
from __future__ import annotations

import contextlib
import math
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CuTeDSL imports — guarded so the file is importable on CPU-only machines
# ---------------------------------------------------------------------------
try:
    import cuda.bindings.driver as cuda  # noqa: F401

    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, BFloat16, TFloat32, Int32
    from cutlass.cute.nvgpu import cpasync, warp
    from cutlass.cutlass_dsl import BaseDSL

    from flash_attn.cute import ampere_helpers as _sm80
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor, assume_tensor_aligned
    from flash_attn.cute import utils as _fa_utils
    from flash_attn.cute.cache_utils import get_jit_cache
    from flash_attn.cute.softmax import Softmax
    from flash_attn.cute.named_barrier import NamedBarrierFwd
    from quack import layout_utils

    _CUTE_AVAILABLE = True

    # ---- Custom MMA ops not exposed by the public Python warp API ----------
    import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
    from cutlass.cute.atom import make_atom
    from cutlass.cute.core import _pack_shape
    from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Trait
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class MmaTF32Op(warp.WarpMmaOp):
        """SM80 TF32 warp-level MMA.

        Underlying PTX instruction::

            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32

        A/B operands are TF32 (32-bit storage, 10-bit mantissa used during
        multiply).  Accumulator is FP32.  This bypasses the Python guard in
        :class:`cutlass.cute.nvgpu.warp.MmaF16BF16Op` which rejects TFloat32.
        """
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

except ImportError:
    _CUTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WinAttnPrecision(Enum):
    """Precision mode for :func:`window_attn_fwd_cute`.

    ``BF16_MIXED``
        BF16 I/O, FP32 accumulators — CuTeDSL kernel using the SM80
        ``mma.sync.m16n8k16.bf16.bf16.f32`` tensor-core instruction.
    ``TF32_ACC_FP32``
        FP32 I/O, TF32 matmul — CuTeDSL kernel using the SM80
        ``mma.sync.m16n8k8.tf32.tf32.f32`` tensor-core instruction.
        Inputs are truncated to 10-bit mantissa during multiply;
        accumulation stays in full FP32.

    For strict FP32 (no TF32 approximation) use
    :func:`torch.nn.functional.scaled_dot_product_attention` directly, or
    pass ``fp32_precision="strict"`` to :func:`window_attn_dispatch`.
    """

    BF16_MIXED    = "bf16_mixed"     # BF16 I/O, FP32 accumulators, CuTeDSL kernel
    TF32_ACC_FP32 = "tf32_acc_fp32"  # FP32 I/O, TF32 matmul,       CuTeDSL kernel


# ---------------------------------------------------------------------------
# SMEM-budget-aware tile-size selection
# ---------------------------------------------------------------------------

def _get_smem_budget_bytes() -> int:
    """Return the per-block dynamic SMEM limit (with optin) in bytes.

    Queries GPU properties via PyTorch when CUDA is available and maps the
    SM architecture to the known optin limit.  Falls back to the conservative
    default (48 KB) when no GPU is present.

    SM-architecture → optin SMEM per block
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SM70 (V100)        : 96 KB
    SM80 / SM86 (A100, RTX 3090)  : 100 KB
    SM89 (L40, RTX 4090)          :  99 KB
    SM90 (H100)        : 228 KB  (limited to 164 KB per block in practice)
    SM100 (B200)       : 228 KB
    SM120 (RTX Pro 6000 Blackwell): 99 KB   ← current target
    """
    if not torch.cuda.is_available():
        return 48 * 1024

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        major, minor = props.major, props.minor
        sm = major * 10 + minor
        if sm >= 100:               # SM100+ (B200, GB10x)
            return 164 * 1024       # practical per-block limit
        elif sm >= 90:              # SM90 (H100)
            return 164 * 1024
        elif sm >= 80:              # SM80/86/89 (A100, RTX 30/40xx, L40)
            return 99 * 1024
        elif sm >= 70:              # SM70 (V100)
            return 96 * 1024
    except Exception:
        pass

    return 48 * 1024  # conservative fallback


def _choose_tile_n(
    seq_len: int,
    head_dim: int = 64,
    tile_m: int = 64,
    smem_budget_bytes: Optional[int] = None,
) -> int:
    """Choose the largest tile_n that fits in SMEM for BF16 (2-byte elements).

    Aims for single-pass attention (tile_n ≥ seq_len) when SMEM allows.
    Falls back to a streaming multi-pass tile when N is too large.

    SMEM layout (num_stages=1)::

        sQ : tile_m  × head_dim × 2B
        sK : tile_n  × head_dim × 2B
        sV : tile_n  × head_dim × 2B
        ──────────────────────────────────────
        budget = sQ + sK + sV

    Solving for tile_n::

        tile_n = (budget − tile_m × head_dim × 2) / (head_dim × 4)

    Rounded down to a multiple of 8 (MMA-N dimension of BF16 m16n8k16 atom).
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes   = tile_m  * head_dim * 2          # BF16 = 2 bytes/element
    kv_row_bytes = head_dim * 2
    max_tile_n = (smem_budget_bytes - sQ_bytes) // (2 * kv_row_bytes)
    max_tile_n = max((max_tile_n // 8) * 8, 8)  # align to MMA-N=8
    return min(seq_len, max_tile_n)


def _choose_tile_n_tf32(
    seq_len: int,
    head_dim: int = 64,
    tile_m: int = 64,
    smem_budget_bytes: Optional[int] = None,
) -> int:
    """Choose the largest tile_n that fits in SMEM for TF32/FP32 (4-byte elements).

    Same formula as :func:`_choose_tile_n` but with 4-byte elements.
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes     = tile_m  * head_dim * 4        # Float32 = 4 bytes/element
    kv_row_bytes = head_dim * 4
    max_tile_n   = (smem_budget_bytes - sQ_bytes) // (2 * kv_row_bytes)
    max_tile_n   = max((max_tile_n // 8) * 8, 8)
    return min(seq_len, max_tile_n)


# ---------------------------------------------------------------------------
# BF16 CuTeDSL kernel
# ---------------------------------------------------------------------------

class WindowAttnFwdBf16:
    """Forward-only BF16 window attention for SM80 / SM120.

    Input / output layout : (Bwin, H, N, Dh) in BFloat16.
    Bias layout           : (nW, N, N)  in Float32, or ``None``.
                            Window index ``bwin_id % nW`` selects the mask.

    The kernel is compiled once per ``(head_dim, seq_len, has_bias,
    tile_m, tile_n)`` via ``cute.compile`` and cached by the dispatch
    function :func:`window_attn_fwd_cute`.

    SMEM layout (num_stages = 1, tile_m = 64, Dh = 64)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * sQ : BF16  64×64   =  8 KB (fixed)
    * sK : BF16  tile_n×64  (dynamic)
    * sV : BF16  tile_n×64  (dynamic)
    * sO : reuses sQ buffer after Q is consumed
    * tile_n is chosen by :func:`_choose_tile_n` to maximise single-pass
      coverage within the available SMEM budget.
    * On 99 KB devices (SM120 / SM89): tile_n ≤ 360 → single-pass for N ≤ 360
    * On 100 KB devices (SM80 A100)  : tile_n ≤ 360 → single-pass for N ≤ 360
    * Conservative 48 KB fallback    : tile_n ≤ 160 → single-pass for N ≤ 160
    """

    _NUM_THREADS: int = 128  # 4 warps

    # ------------------------------------------------------------------
    def __init__(
        self,
        head_dim: int,
        seq_len: int,
        has_bias: bool = False,
        tile_m: int = 64,
        tile_n: Optional[int] = None,
        num_stages: int = 1,
    ):
        assert _CUTE_AVAILABLE, "CuTeDSL / flash-attn not found"

        if tile_n is None:
            tile_n = _choose_tile_n(seq_len, head_dim=head_dim, tile_m=tile_m)

        self.head_dim   = head_dim
        self.seq_len    = seq_len
        self.has_bias   = has_bias
        self.num_stages = num_stages
        self.num_threads = self._NUM_THREADS
        self.tile_m     = min(tile_m, seq_len)
        self.tile_n     = min(tile_n, seq_len)

        # Round head_dim to a 16-element multiple (16-byte align for BF16)
        hdim_align      = 16
        self.tile_hdim  = int(math.ceil(head_dim / hdim_align) * hdim_align)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.dtype      = BFloat16

        # ---- SMEM layout atoms ------------------------------------------
        sQK_atom = _sm80.get_smem_layout_atom(BFloat16, self.tile_hdim)
        sV_atom  = _sm80.get_smem_layout_atom(BFloat16, self.tile_hdim)

        self.sQ_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        self.sV_layout = cute.tile_to_shape(
            sV_atom,  (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        # sO reuses the sQ region (same shape) once Q is done
        self.sO_layout = cute.tile_to_shape(
            sV_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )

        # ---- MMA: BF16 inputs → FP32 accumulator -----------------------
        _mma_op  = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        num_warps = self.num_threads // 32
        _mma_args = dict(
            permutation_mnk=(num_warps * 16, 16, 16),
        )
        self.tiled_mma_qk = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)
        self.tiled_mma_pv = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)

        # ---- 128-bit async GMEM→SMEM copies  ---------------------------
        _bits  = 128
        _elems = _bits // BFloat16.width  # = 8 elements per 128-bit load

        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            BFloat16,
            num_bits_per_copy=_bits,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=_bits
        )

        _sQK_dim1 = sQK_atom.outer.shape[1] // _elems
        _sV_dim1  = sV_atom.outer.shape[1]  // _elems
        vQKV = cute.make_layout((1, _elems))

        def _tv(dim1):
            return cute.make_ordered_layout(
                (self.num_threads // dim1, dim1), order=(1, 0)
            )

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async, _tv(_sV_dim1),  vQKV)
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_store,  _tv(_sV_dim1),  vQKV)

        # ---- Shared-memory struct ----------------------------------------
        _mk_struct = lambda layout: cute.struct.Align[
            cute.struct.MemRange[BFloat16, cute.cosize(layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            sQ: _mk_struct(self.sQ_layout)
            sK: _mk_struct(self.sK_layout)
            sV: _mk_struct(self.sV_layout)

        self.SharedStorage = SharedStorage
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,    # (Bwin, H, N, Dh)  BF16
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],  # (nW, N, N)  Float32  or  None
        softmax_scale_log2: Float32,
        stream: cuda.CUstream = None,
    ) -> None:
        """JIT entry point compiled by ``cute.compile``.

        Performs a zero-copy layout transpose from the caller's
        (Bwin, H, N, Dh) into the kernel's (N, Dh, H, Bwin) so that
        ``mQ[None, None, head_id, bwin_id]`` yields the (N, Dh) 2-D slice
        for a given window-batch / head pair.
        """
        # (Bwin, H, N, Dh) → (N, Dh, H, Bwin)
        _tr = [2, 3, 1, 0]
        mQ_t, mK_t, mV_t, mO_t = [
            assume_tensor_aligned(
                cute.make_tensor(t.iterator, cute.select(t.layout, mode=_tr))
            )
            for t in (mQ, mK, mV, mO)
        ]

        N    = mQ.shape[2]
        H    = mQ.shape[1]
        Bwin = mQ.shape[0]
        num_m_blocks = (N + self.tile_m - 1) // self.tile_m

        self.kernel(
            mQ_t, mK_t, mV_t, mO_t, mBias,
            softmax_scale_log2,
            N, H,
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout,
            self.gmem_tiled_copy_Q, self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V, self.gmem_tiled_copy_O,
            self.tiled_mma_qk, self.tiled_mma_pv,
            self.SharedStorage,
        ).launch(
            grid=[num_m_blocks, Bwin * H, 1],
            block=[self.num_threads, 1, 1],
            smem=self.SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,    # (N, Dh, H, Bwin) — after transpose
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],   # (nW, N, N) Float32  or  None
        softmax_scale_log2: Float32,
        seqlen: Int32,
        H: Int32,
        sQ_layout, sK_layout, sV_layout, sO_layout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage,
    ):
        """kernel body.

        Grid  : (ceil(N/tile_m),  Bwin * H,  1)
        Block : (128, 1, 1)   — 4 warps

        Algorithm: FlashAttention-2 online softmax, single K/V pass when
        tile_n >= N (default for Aurora's N=144).
        """
        tidx, _, _  = cute.arch.thread_idx()
        m_block     = cute.arch.block_idx_x()   # Q-tile along seqlen
        by          = cute.arch.block_idx_y()   # flattened (bwin * H + head)

        head_id = by % H
        bwin_id = by // H

        # Slice to (N, Dh) for this (head, window-batch) pair
        mQ_cur = mQ[None, None, head_id, bwin_id]
        mK_cur = mK[None, None, head_id, bwin_id]
        mV_cur = mV[None, None, head_id, bwin_id]
        mO_cur = mO[None, None, head_id, bwin_id]

        blkQ  = (self.tile_m, self.tile_hdim)
        blkKV = (self.tile_n, self.tile_hdim)

        # Global tiles: gK / gV have an extra n-block trailing dimension
        gQ = cute.local_tile(mQ_cur, blkQ,  (m_block, 0))
        gK = cute.local_tile(mK_cur, blkKV, (None, 0))
        gV = cute.local_tile(mV_cur, blkKV, (None, 0))

        # ---- Shared memory -----------------------------------------------
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ  = storage.sQ.get_tensor(sQ_layout)
        sK  = storage.sK.get_tensor(sK_layout)
        sV  = storage.sV.get_tensor(sV_layout)
        sVt = layout_utils.transpose_view(sV)   # (Dh, N, stage) for P×V

        # ---- GMEM → SMEM tiled-copy partitioning -------------------------
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)

        tQsQ = gmem_thr_copy_Q.partition_D(sQ)
        tQgQ = gmem_thr_copy_Q.partition_S(gQ)
        tKsK = gmem_thr_copy_K.partition_D(sK)
        tKgK = gmem_thr_copy_K.partition_S(gK)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        tVgV = gmem_thr_copy_V.partition_S(gV)

        # ---- MMA fragments and accumulators ------------------------------
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)

        tSrQ  = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK  = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))

        acc_O = cute.make_fragment(
            thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim)), Float32
        )
        acc_O.fill(0.0)

        # ---- SMEM → register ldmatrix copy atoms -------------------------
        smem_cp_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
        )
        smem_cp_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16
        )
        smem_thr_cp_Q  = _fa_utils.make_tiled_copy_A(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_K  = _fa_utils.make_tiled_copy_B(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_V  = _fa_utils.make_tiled_copy_B(smem_cp_V,  tiled_mma_pv).get_slice(tidx)

        tSsQ  = smem_thr_cp_Q.partition_S(sQ)
        tSsK  = smem_thr_cp_K.partition_S(sK)
        tOsVt = smem_thr_cp_V.partition_S(sVt)

        # ---- Online softmax state ----------------------------------------
        n_rows = acc_O.shape[0][0] * acc_O.shape[1]
        arch_v = self.arch.major * 10 + self.arch.minor
        softmax = Softmax.create(
            scale_log2=softmax_scale_log2,
            num_rows=n_rows,
            arch=arch_v,
        )
        softmax.reset()

        # ---- Predicates for partial tiles --------------------------------
        cQ   = cute.make_identity_tensor(blkQ)
        tQcQ = gmem_thr_copy_Q.partition_S(cQ)
        t0QcQ = gmem_thr_copy_Q.get_slice(0).partition_S(cQ)
        tQpQ = _fa_utils.predicate_k(tQcQ, limit=self.head_dim)

        cKV  = cute.make_identity_tensor(blkKV)
        tKcK = gmem_thr_copy_K.partition_S(cKV)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cKV)
        tKpK = _fa_utils.predicate_k(tKcK, limit=self.head_dim)

        tVcV  = gmem_thr_copy_V.partition_S(cKV)
        t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cKV)
        tVpV  = tKpK   # same head_dim predicate

        # ---- Prologue: async load Q and first K block -------------------
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

        # Prefetch first K tile (n_block = 0) with predicates
        self._load_K(
            gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
            n_block=0, smem_stage=0, seqlen=seqlen, need_predicates=True,
        )
        cute.arch.cp_async_commit_group()

        # Wait for Q (the K wait will be inside the loop)
        cute.arch.cp_async_wait_group(1)
        cute.arch.syncthreads()

        # ---- Bias setup: coordinate tensor for acc_S indexing -----------
        # tScS[i] → (row_in_tile, col_in_tile) for acc_S element i.
        # We add mBias[win_id, m_global, n_global] to acc_S[i] when valid.
        if cutlass.const_expr(mBias is not None):
            nW      = mBias.shape[0]
            win_id  = bwin_id % nW
            mBias_w = mBias[win_id]   # (N, N)  — window-specific mask
            cS      = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS    = thr_mma_qk.partition_C(cS)  # coord per acc_S element

        # ---- Main loop: iterate K/V blocks (forward, 0 → n_block_max) --
        #
        # Pipeline for num_stages=1 (sK uses a single SMEM buffer):
        #
        #   prologue: async-load K[0], commit
        #   loop n:
        #     wait K[n]           # K[n] now safe to read
        #     QK GEMM             # reads sK[...,0]  (K[n])
        #     async-load K[n+1]   # writes sK[...,0] AFTER GEMM → no race
        #     commit K[n+1]
        #     bias + softmax
        #     async-load V[n]
        #     commit V[n]
        #     wait ALL groups     # K[n+1] and V[n] both ready
        #     PV GEMM
        #
        # Using cp_async_wait_group(0) collapses K and V waits into one
        # barrier.  For num_stages > 1 this loop would need proper
        # double-buffering; for Aurora's single-pass (tile_n=N=144) there
        # is only ONE iteration so the pipeline cost is zero.
        for n_block in cutlass.range_constexpr(n_block_max):
            is_first_n = cutlass.const_expr(n_block == 0)

            # --- 1. Wait for K[n] to arrive in sK[..., 0] ----------------
            cute.arch.cp_async_wait_group(0)
            cute.arch.syncthreads()

            # --- 2. Q×K GEMM (reads sK[..., 0] = K[n]) ------------------
            acc_S = cute.make_fragment(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            )
            acc_S.fill(0.0)

            _sm80.gemm(
                tiled_mma_qk, acc_S,
                tSrQ, tSrK,
                tSsQ, tSsK[None, None, 0],   # stage 0  (num_stages = 1)
                smem_thr_cp_Q, smem_thr_cp_K,
            )

            # --- 3. Prefetch K[n+1] into sK[..., 0] (AFTER GEMM) --------
            #    Safe: GEMM is done, sK is free for next K tile.
            if cutlass.const_expr(n_block < n_block_max - 1):
                self._load_K(
                    gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
                    n_block=n_block + 1, smem_stage=0,
                    seqlen=seqlen, need_predicates=False,
                )
                cute.arch.cp_async_commit_group()

            # --- 4. Add (nW, N, N) bias via MMA-C coordinate tensor ------
            # Each thread owns ~4 elements of acc_S.  We load the matching
            # FP32 bias values directly from global memory using the
            # MMA-layout coordinate tensor tScS.  For Aurora's single-pass
            # (tile_n=144=N) this costs ~4 scattered 4-byte GM loads total
            # per thread — negligible vs. the GEMM latency.
            if cutlass.const_expr(mBias is not None):
                m_start = m_block * self.tile_m
                n_start = n_block * self.tile_n
                for i in cutlass.range_constexpr(cute.size(acc_S)):
                    m_idx = m_start + tScS[i][0]
                    n_idx = n_start + tScS[i][1]
                    # Guard invalid positions (last partial M-tile).
                    if m_idx < seqlen and n_idx < seqlen:
                        acc_S[i] = acc_S[i] + mBias_w[m_idx, n_idx]

            # --- 5. Online softmax update (FlashAttention-2) -------------
            row_scale = softmax.online_softmax(
                acc_S, is_first=is_first_n, check_inf=True
            )
            if cutlass.const_expr(not is_first_n):
                softmax.rescale_O(acc_O, row_scale)

            # --- 6. Async load V[n] into sV[..., 0] ----------------------
            self._load_V(
                gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                n_block=n_block, seqlen=seqlen,
            )
            cute.arch.cp_async_commit_group()

            # --- 7. Wait for ALL pending groups (K[n+1] and V[n]) --------
            cute.arch.cp_async_wait_group(0)
            cute.arch.syncthreads()

            # --- 8. P×V GEMM  (acc_O += P_bf16 @ V^T) -------------------
            acc_S_bf16 = cute.make_fragment_like(acc_S, BFloat16)
            acc_S_bf16.store(acc_S.load().to(BFloat16))

            _sm80.gemm_rs(
                tiled_mma_pv, acc_O,
                acc_S_bf16, tOrVt,
                tOsVt[None, None, 0],
                smem_thr_cp_V,
            )
            cute.arch.syncthreads()

        # ---- Finalize softmax and convert to BF16 ------------------------
        final_row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, final_row_scale)

        rO = cute.make_fragment_like(acc_O, BFloat16)
        rO.store(acc_O.load().to(BFloat16))

        # ---- Epilogue: rmem → smem (MMA-C store atom) -------------------
        smem_cp_O = _fa_utils.get_smem_store_atom(arch_v, BFloat16)
        smem_thr_cp_O = cute.make_tiled_copy_C(smem_cp_O, tiled_mma_pv).get_slice(tidx)

        # Reuse sQ shared-memory buffer as sO (same shape / size)
        sO = storage.sQ.get_tensor(sO_layout)
        taccOrO = smem_thr_cp_O.retile(rO)
        taccOsO = smem_thr_cp_O.partition_D(sO)
        cute.copy(smem_cp_O, taccOrO, taccOsO)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_threads,
        )

        # ---- Epilogue: smem → gmem (universal store) --------------------
        gO = cute.local_tile(mO_cur, blkQ, (m_block, 0))
        gmem_thr_cp_O  = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_cp_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, BFloat16)
        cute.autovec_copy(tOsO, tOrO)

        tOgO  = gmem_thr_cp_O.partition_D(gO)
        cO    = cute.make_identity_tensor(blkQ)
        tOcO  = gmem_thr_cp_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO  = _fa_utils.predicate_k(tOcO, limit=self.head_dim)

        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None]
                    if cutlass.const_expr(self.check_hdim_oob) else None,
                )

    # ------------------------------------------------------------------
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
        """Async cp.async load of K tile ``n_block`` into stage ``smem_stage``."""
        is_even_n = cutlass.const_expr(
            self.tile_n % gmem_tiled_copy_K.tiler_mn[0].shape == 0
        )
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

    # ------------------------------------------------------------------
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
    ):
        """Async cp.async load of V tile ``n_block`` into stage 0."""
        is_even_n = cutlass.const_expr(
            self.tile_n % gmem_tiled_copy_V.tiler_mn[0].shape == 0
        )
        if cutlass.const_expr(is_even_n):
            cute.copy(
                gmem_tiled_copy_V,
                tVgV[None, None, None, n_block],
                tVsV[None, None, None, 0],
            )
        else:
            seqlen_limit = seqlen - n_block * self.tile_n - tVcV[0][0]
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if t0VcV[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy_V,
                        tVgV[None, n, None, n_block],
                        tVsV[None, n, None, 0],
                        pred=tVpV[None, n, None]
                        if cutlass.const_expr(self.check_hdim_oob) else None,
                    )


# ---------------------------------------------------------------------------
# TF32 CuTeDSL kernel
# ---------------------------------------------------------------------------

class WindowAttnFwdTF32:
    """Forward-only TF32 window attention for SM80+.

    Input / output layout : (Bwin, H, N, Dh) in Float32.
    Bias layout           : (nW, N, N)  in Float32, or ``None``.

    Uses the SM80 ``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``
    tensor-core instruction: FP32 inputs are truncated to 10-bit mantissa
    *during multiply* (TF32 approximation), while accumulation remains in
    full FP32.  This yields ~8× higher math throughput vs. SIMT FP32 at the
    cost of slight numeric deviation (≈ 1e-3 relative error vs. strict FP32).

    SMEM layout (num_stages=1, tile_m=64, Dh=64, 99 KB budget)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * sQ : FP32  64×64   = 16 KB
    * sK : FP32  N×64    (tile_n rows)
    * sV : FP32  N×64    (tile_n rows)
    * sO : reuses sQ buffer after Q is consumed
    * tile_n is chosen by :func:`_choose_tile_n_tf32` to maximise single-pass
      coverage within the available SMEM budget (up to ~160 on 99 KB devices).
    """

    _NUM_THREADS: int = 128  # 4 warps

    def __init__(
        self,
        head_dim: int,
        seq_len: int,
        has_bias: bool = False,
        tile_m: int = 64,
        tile_n: Optional[int] = None,
        num_stages: int = 1,
    ):
        assert _CUTE_AVAILABLE, "CuTeDSL / flash-attn not found"

        if tile_n is None:
            tile_n = _choose_tile_n_tf32(seq_len, head_dim=head_dim, tile_m=tile_m)

        self.head_dim    = head_dim
        self.seq_len     = seq_len
        self.has_bias    = has_bias
        self.num_stages  = num_stages
        self.num_threads = self._NUM_THREADS
        self.tile_m      = min(tile_m, seq_len)
        self.tile_n      = min(tile_n, seq_len)

        # Round head_dim to 8-element multiple (32-byte align for FP32)
        hdim_align       = 8
        self.tile_hdim   = int(math.ceil(head_dim / hdim_align) * hdim_align)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.dtype       = Float32   # SMEM / GMEM element type

        # ---- SMEM layout atoms ------------------------------------------
        # Float32 and TFloat32 have the same width (32-bit), so the swizzle
        # pattern is identical.  We use Float32 for SMEM layout for clarity.
        sQK_atom = _sm80.get_smem_layout_atom(Float32, self.tile_hdim)
        sV_atom  = _sm80.get_smem_layout_atom(Float32, self.tile_hdim)

        self.sQ_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            sQK_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        self.sV_layout = cute.tile_to_shape(
            sV_atom, (self.tile_n, self.tile_hdim, num_stages), (0, 1, 2)
        )
        # sO reuses sQ buffer after Q is consumed
        self.sO_layout = cute.tile_to_shape(
            sV_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )

        # ---- MMA: TFloat32 inputs → FP32 accumulator ---------------------
        _mma_op   = MmaTF32Op()   # mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        num_warps = self.num_threads // 32
        _mma_args = dict(permutation_mnk=(num_warps * 16, 8, 8))
        self.tiled_mma_qk = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)
        self.tiled_mma_pv = cute.make_tiled_mma(_mma_op, (num_warps, 1, 1), **_mma_args)

        # ---- 128-bit async GMEM→SMEM copies (Float32) --------------------
        _bits  = 128
        _elems = _bits // Float32.width   # = 4 FP32 per 128-bit load

        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=_bits,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=_bits
        )

        _sQK_dim1 = sQK_atom.outer.shape[1] // _elems
        _sV_dim1  = sV_atom.outer.shape[1]  // _elems
        vQKV = cute.make_layout((1, _elems))

        def _tv(dim1):
            return cute.make_ordered_layout(
                (self.num_threads // dim1, dim1), order=(1, 0)
            )

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async, _tv(_sQK_dim1), vQKV)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async, _tv(_sV_dim1),  vQKV)
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_store,  _tv(_sV_dim1),  vQKV)

        # ---- Shared-memory struct ----------------------------------------
        _mk_struct = lambda layout: cute.struct.Align[
            cute.struct.MemRange[Float32, cute.cosize(layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            sQ: _mk_struct(self.sQ_layout)
            sK: _mk_struct(self.sK_layout)
            sV: _mk_struct(self.sV_layout)

        self.SharedStorage = SharedStorage
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,    # (Bwin, H, N, Dh)  Float32
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],   # (nW, N, N)  Float32  or  None
        softmax_scale_log2: Float32,
        stream: cuda.CUstream = None,
    ) -> None:
        # (Bwin, H, N, Dh) → (N, Dh, H, Bwin)
        _tr = [2, 3, 1, 0]
        mQ_t, mK_t, mV_t, mO_t = [
            assume_tensor_aligned(
                cute.make_tensor(t.iterator, cute.select(t.layout, mode=_tr))
            )
            for t in (mQ, mK, mV, mO)
        ]

        N    = mQ.shape[2]
        H    = mQ.shape[1]
        Bwin = mQ.shape[0]
        num_m_blocks = (N + self.tile_m - 1) // self.tile_m

        self.kernel(
            mQ_t, mK_t, mV_t, mO_t, mBias,
            softmax_scale_log2,
            N, H,
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout,
            self.gmem_tiled_copy_Q, self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V, self.gmem_tiled_copy_O,
            self.tiled_mma_qk, self.tiled_mma_pv,
            self.SharedStorage,
        ).launch(
            grid=[num_m_blocks, Bwin * H, 1],
            block=[self.num_threads, 1, 1],
            smem=self.SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,    # (N, Dh, H, Bwin) — after transpose
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: Optional[cute.Tensor],   # (nW, N, N) Float32  or  None
        softmax_scale_log2: Float32,
        seqlen: Int32,
        H: Int32,
        sQ_layout, sK_layout, sV_layout, sO_layout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage,
    ):
        """TF32 attention kernel body.

        Grid  : (ceil(N/tile_m),  Bwin * H,  1)
        Block : (128, 1, 1)  — 4 warps

        Algorithm: FlashAttention-2 online softmax.
        FP32 inputs are truncated to TF32 precision during QK and PV GEMMs;
        softmax and accumulation remain in full FP32.
        """
        tidx, _, _  = cute.arch.thread_idx()
        m_block     = cute.arch.block_idx_x()
        by          = cute.arch.block_idx_y()

        head_id = by % H
        bwin_id = by // H

        mQ_cur = mQ[None, None, head_id, bwin_id]
        mK_cur = mK[None, None, head_id, bwin_id]
        mV_cur = mV[None, None, head_id, bwin_id]
        mO_cur = mO[None, None, head_id, bwin_id]

        blkQ  = (self.tile_m, self.tile_hdim)
        blkKV = (self.tile_n, self.tile_hdim)

        gQ = cute.local_tile(mQ_cur, blkQ,  (m_block, 0))
        gK = cute.local_tile(mK_cur, blkKV, (None, 0))
        gV = cute.local_tile(mV_cur, blkKV, (None, 0))

        # ---- Shared memory -----------------------------------------------
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ  = storage.sQ.get_tensor(sQ_layout)
        sK  = storage.sK.get_tensor(sK_layout)
        sV  = storage.sV.get_tensor(sV_layout)
        sVt = layout_utils.transpose_view(sV)   # (Dh, N, stage) for P×V

        # ---- GMEM → SMEM tiled-copy partitioning -------------------------
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)

        tQsQ = gmem_thr_copy_Q.partition_D(sQ)
        tQgQ = gmem_thr_copy_Q.partition_S(gQ)
        tKsK = gmem_thr_copy_K.partition_D(sK)
        tKgK = gmem_thr_copy_K.partition_S(gK)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        tVgV = gmem_thr_copy_V.partition_S(gV)

        # ---- MMA fragments and accumulators ------------------------------
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)

        tSrQ  = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK  = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))

        acc_O = cute.make_fragment(
            thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim)), Float32
        )
        acc_O.fill(0.0)

        # ---- SMEM → register copy atoms (universal LDS for FP32/TF32) ---
        # TFloat32 SMEM→register: 128-bit load = 4 TF32 values per copy.
        # CopyUniversalOp generates vectorized LDS instructions (no ldmatrix).
        smem_cp_QK = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), TFloat32, num_bits_per_copy=128
        )
        smem_cp_V = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), TFloat32, num_bits_per_copy=128
        )
        smem_thr_cp_Q  = _fa_utils.make_tiled_copy_A(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_K  = _fa_utils.make_tiled_copy_B(smem_cp_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_cp_V  = _fa_utils.make_tiled_copy_B(smem_cp_V,  tiled_mma_pv).get_slice(tidx)

        tSsQ  = smem_thr_cp_Q.partition_S(sQ)
        tSsK  = smem_thr_cp_K.partition_S(sK)
        tOsVt = smem_thr_cp_V.partition_S(sVt)

        # ---- Online softmax state ----------------------------------------
        n_rows = acc_O.shape[0][0] * acc_O.shape[1]
        arch_v = self.arch.major * 10 + self.arch.minor
        softmax = Softmax.create(
            scale_log2=softmax_scale_log2,
            num_rows=n_rows,
            arch=arch_v,
        )
        softmax.reset()

        # ---- Predicates for partial tiles --------------------------------
        cQ   = cute.make_identity_tensor(blkQ)
        tQcQ = gmem_thr_copy_Q.partition_S(cQ)
        t0QcQ = gmem_thr_copy_Q.get_slice(0).partition_S(cQ)
        tQpQ = _fa_utils.predicate_k(tQcQ, limit=self.head_dim)

        cKV  = cute.make_identity_tensor(blkKV)
        tKcK = gmem_thr_copy_K.partition_S(cKV)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cKV)
        tKpK = _fa_utils.predicate_k(tKcK, limit=self.head_dim)

        tVcV  = gmem_thr_copy_V.partition_S(cKV)
        t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cKV)
        tVpV  = tKpK

        # ---- Prologue: async load Q and first K block -------------------
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
            gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
            n_block=0, smem_stage=0, seqlen=seqlen, need_predicates=True,
        )
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        cute.arch.syncthreads()

        # ---- Bias setup -------------------------------------------------
        if cutlass.const_expr(mBias is not None):
            nW      = mBias.shape[0]
            win_id  = bwin_id % nW
            mBias_w = mBias[win_id]
            cS      = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS    = thr_mma_qk.partition_C(cS)

        # ---- Main loop: iterate K/V blocks ------------------------------
        for n_block in cutlass.range_constexpr(n_block_max):
            is_first_n = cutlass.const_expr(n_block == 0)

            # --- 1. Wait for K[n] ----------------------------------------
            cute.arch.cp_async_wait_group(0)
            cute.arch.syncthreads()

            # --- 2. Q×K GEMM (TF32 inputs, FP32 accumulator) -------------
            acc_S = cute.make_fragment(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            )
            acc_S.fill(0.0)

            _sm80.gemm(
                tiled_mma_qk, acc_S,
                tSrQ, tSrK,
                tSsQ, tSsK[None, None, 0],
                smem_thr_cp_Q, smem_thr_cp_K,
            )

            # --- 3. Prefetch K[n+1] (AFTER GEMM — sK is now free) --------
            if cutlass.const_expr(n_block < n_block_max - 1):
                self._load_K(
                    gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
                    n_block=n_block + 1, smem_stage=0,
                    seqlen=seqlen, need_predicates=False,
                )
                cute.arch.cp_async_commit_group()

            # --- 4. Add bias (nW, N, N) element-wise via MMA coords ------
            if cutlass.const_expr(mBias is not None):
                m_start = m_block * self.tile_m
                n_start = n_block * self.tile_n
                for i in cutlass.range_constexpr(cute.size(acc_S)):
                    m_idx = m_start + tScS[i][0]
                    n_idx = n_start + tScS[i][1]
                    if m_idx < seqlen and n_idx < seqlen:
                        acc_S[i] = acc_S[i] + mBias_w[m_idx, n_idx]

            # --- 5. Online softmax (FlashAttention-2) ---------------------
            row_scale = softmax.online_softmax(
                acc_S, is_first=is_first_n, check_inf=True
            )
            if cutlass.const_expr(not is_first_n):
                softmax.rescale_O(acc_O, row_scale)

            # --- 6. Async load V[n] --------------------------------------
            self._load_V(
                gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                n_block=n_block, seqlen=seqlen,
            )
            cute.arch.cp_async_commit_group()

            # --- 7. Wait for all pending async groups --------------------
            cute.arch.cp_async_wait_group(0)
            cute.arch.syncthreads()

            # --- 8. P×V GEMM: P is in FP32, cast to TF32 for MMA --------
            # FP32 → TF32 truncates the lower 13 mantissa bits.
            acc_S_tf32 = cute.make_fragment_like(acc_S, TFloat32)
            acc_S_tf32.store(acc_S.load().to(TFloat32))

            _sm80.gemm_rs(
                tiled_mma_pv, acc_O,
                acc_S_tf32, tOrVt,
                tOsVt[None, None, 0],
                smem_thr_cp_V,
            )
            cute.arch.syncthreads()

        # ---- Finalize softmax (FP32 accumulator, no conversion needed) ---
        final_row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, final_row_scale)

        rO = cute.make_fragment_like(acc_O, Float32)
        rO.store(acc_O.load())

        # ---- Epilogue: rmem → smem (FP32 universal store atom) ----------
        smem_cp_O = _fa_utils.get_smem_store_atom(arch_v, Float32)
        smem_thr_cp_O = cute.make_tiled_copy_C(smem_cp_O, tiled_mma_pv).get_slice(tidx)

        sO = storage.sQ.get_tensor(sO_layout)
        taccOrO = smem_thr_cp_O.retile(rO)
        taccOsO = smem_thr_cp_O.partition_D(sO)
        cute.copy(smem_cp_O, taccOrO, taccOsO)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_threads,
        )

        # ---- Epilogue: smem → gmem (universal store) --------------------
        gO = cute.local_tile(mO_cur, blkQ, (m_block, 0))
        gmem_thr_cp_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_cp_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, Float32)
        cute.autovec_copy(tOsO, tOrO)

        tOgO  = gmem_thr_cp_O.partition_D(gO)
        cO    = cute.make_identity_tensor(blkQ)
        tOcO  = gmem_thr_cp_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO  = _fa_utils.predicate_k(tOcO, limit=self.head_dim)

        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None]
                    if cutlass.const_expr(self.check_hdim_oob) else None,
                )

    # ------------------------------------------------------------------
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
        is_even_n = cutlass.const_expr(
            self.tile_n % gmem_tiled_copy_K.tiler_mn[0].shape == 0
        )
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

    # ------------------------------------------------------------------
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
    ):
        is_even_n = cutlass.const_expr(
            self.tile_n % gmem_tiled_copy_V.tiler_mn[0].shape == 0
        )
        if cutlass.const_expr(is_even_n):
            cute.copy(
                gmem_tiled_copy_V,
                tVgV[None, None, None, n_block],
                tVsV[None, None, None, 0],
            )
        else:
            seqlen_limit = seqlen - n_block * self.tile_n - tVcV[0][0]
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if t0VcV[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy_V,
                        tVgV[None, n, None, n_block],
                        tVsV[None, n, None, 0],
                        pred=tVpV[None, n, None]
                        if cutlass.const_expr(self.check_hdim_oob) else None,
                    )


# ---------------------------------------------------------------------------
# Compile cache (mirrors flash_attn.cute.cache_utils pattern)
# ---------------------------------------------------------------------------

_bf16_compile_cache: dict = {}


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
    """Return a compiled ``cute.compile`` callable, building it on first call."""
    compile_key = (head_dim, seq_len, has_bias, tile_m, tile_n)
    if compile_key in _bf16_compile_cache:
        return _bf16_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdBf16(
        head_dim=head_dim,
        seq_len=seq_len,
        has_bias=has_bias,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    # Build fake cute.Tensor descriptors for compilation
    q_ct  = to_cute_tensor(q)
    k_ct  = to_cute_tensor(k)
    v_ct  = to_cute_tensor(v)
    o_ct  = to_cute_tensor(o)
    bias_ct = to_cute_tensor(bias_or_none) if bias_or_none is not None else None

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        q_ct, k_ct, v_ct, o_ct,
        bias_ct,
        Float32(1.0),   # softmax_scale_log2 placeholder
        stream,
        options="--enable-tvm-ffi",
    )
    _bf16_compile_cache[compile_key] = compiled
    return compiled


_tf32_compile_cache: dict = {}


def _get_or_compile_tf32(
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
    """Return a compiled TF32 kernel callable, building it on first call."""
    compile_key = (head_dim, seq_len, has_bias, tile_m, tile_n)
    if compile_key in _tf32_compile_cache:
        return _tf32_compile_cache[compile_key]

    kernel_obj = WindowAttnFwdTF32(
        head_dim=head_dim,
        seq_len=seq_len,
        has_bias=has_bias,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    q_ct  = to_cute_tensor(q)
    k_ct  = to_cute_tensor(k)
    v_ct  = to_cute_tensor(v)
    o_ct  = to_cute_tensor(o)
    bias_ct = to_cute_tensor(bias_or_none) if bias_or_none is not None else None

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        q_ct, k_ct, v_ct, o_ct,
        bias_ct,
        Float32(1.0),
        stream,
        options="--enable-tvm-ffi",
    )
    _tf32_compile_cache[compile_key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# FP32 / TF32 SDPA helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _tf32_disabled():
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
    # Gather: each sample picks the window mask for its position
    # window index for sample b is b % nW
    # For arbitrary Bwin this requires a gather, but for the common case
    # where Bwin is a multiple of nW we can use expand directly.
    if Bwin % nW == 0:
        # (nW, N, N) → (nW, 1, N, N) → expand to (Bwin, 1, N, N)
        bias_expanded = bias.unsqueeze(1)  # (nW, 1, N, N)
        reps = Bwin // nW
        if reps > 1:
            bias_expanded = bias_expanded.repeat(reps, 1, 1, 1)
    else:
        # General case: index by b % nW
        win_ids = torch.arange(Bwin, device=bias.device) % nW
        bias_expanded = bias[win_ids].unsqueeze(1)  # (Bwin, 1, N, N)
    return bias_expanded


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

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
    """Window attention forward pass — CuTeDSL kernels only.

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
        GEMM tile sizes passed to the CuTeDSL kernel.

    Returns
    -------
    torch.Tensor
        Shape ``(Bwin, H, N, Dh)``, same dtype as inputs.
    """
    # Dtype guards fire before the CuTe availability check so that tests
    # without a CUDA/CuTe environment still get informative errors.
    if precision == WinAttnPrecision.TF32_ACC_FP32:
        assert q.dtype == torch.float32, "TF32_ACC_FP32 requires float32 tensors"
    else:
        assert q.dtype == torch.bfloat16, "BF16_MIXED requires bfloat16 tensors"
    assert _CUTE_AVAILABLE, "CuTeDSL / flash-attn package not found"

    Bwin, H, N, Dh = q.shape
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(Dh)

    # ---- TF32 CuTeDSL path ----------------------------------------------
    if precision == WinAttnPrecision.TF32_ACC_FP32:
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if bias is not None and not bias.is_contiguous():
            bias = bias.contiguous()

        out = torch.empty_like(q)
        import math as _math
        scale_log2 = Float32(_math.log2(_math.e) * scale_qk)

        _tile_n_tf32 = tile_n if tile_n is not None else _choose_tile_n_tf32(N, head_dim=Dh, tile_m=tile_m)
        has_bias = bias is not None

        compiled_fn = _get_or_compile_tf32(
            head_dim=Dh,
            seq_len=N,
            has_bias=has_bias,
            tile_m=tile_m,
            tile_n=_tile_n_tf32,
            q=q, k=k, v=v, o=out,
            bias_or_none=bias,
        )
        compiled_fn(q, k, v, out, bias, scale_log2)
        return out

    # ---- BF16 CuTeDSL path ----------------------------------------------
    # Ensure contiguous layout (required by CuTeDSL async copies)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    # Allocate output
    out = torch.empty_like(q)

    # log2(e) * scale so the kernel uses exp2 (faster than exp on GPU)
    import math as _math
    scale_log2 = Float32(_math.log2(_math.e) * scale_qk)

    _tile_n = tile_n if tile_n is not None else _choose_tile_n(N, head_dim=Dh, tile_m=tile_m)
    has_bias = bias is not None

    compiled_fn = _get_or_compile_bf16(
        head_dim=Dh,
        seq_len=N,
        has_bias=has_bias,
        tile_m=tile_m,
        tile_n=_tile_n,
        q=q, k=k, v=v, o=out,
        bias_or_none=bias,
    )

    compiled_fn(q, k, v, out, bias, scale_log2)
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
    ``use_cute=True`` + CUDA + BF16  ->  CuTeDSL BF16_MIXED kernel
    ``use_cute=True`` + CUDA + FP32 + ``fp32_precision="tf32"``
                                     ->  CuTeDSL TF32_ACC_FP32 kernel
    ``fp32_precision="strict"``      ->  torch SDPA (strict IEEE-754 FP32)
    everything else                  ->  torch SDPA

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
        Enable CuTeDSL kernels.  Set ``False`` to always fall back to
        ``F.scaled_dot_product_attention``.
    fp32_precision:
        For float32 inputs only.
        ``"tf32"``    Use the CuTeDSL TF32 tensor-core kernel (faster,
                      ≈1e-3 relative approximation error).
        ``"strict"``  Skip CuTeDSL; delegate to torch SDPA with TF32
                      disabled (exact IEEE-754 FP32, Aurora's native path).
    tile_m, tile_n:
        GEMM tile sizes forwarded to the CuTeDSL kernels.

    Returns
    -------
    torch.Tensor
        Shape ``(Bwin, H, N, Dh)``, same dtype as inputs.
    """
    if scale_qk is None:
        scale_qk = 1.0 / math.sqrt(q.shape[-1])

    Bwin, H, N, _ = q.shape

    # ---- CuTeDSL paths --------------------------------------------------
    use_cute_kernel = (
        use_cute
        and _CUTE_AVAILABLE
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
            tile_m=tile_m,
            tile_n=tile_n,
        )

    # ---- Fallback: Aurora's native SDPA path ----------------------------
    # Used for: strict FP32, CPU, non-CUDA, use_cute=False.
    # For strict FP32 we disable TF32 to match Aurora's original semantics.
    attn_mask = _expand_bias_for_sdpa(bias, Bwin, H, N) if bias is not None else None
    use_tf32 = not (q.dtype == torch.float32 and fp32_precision == "strict")
    ctx = contextlib.nullcontext() if use_tf32 else _tf32_disabled()
    with ctx:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale_qk
        )
