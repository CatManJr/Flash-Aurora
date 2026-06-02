"""Shared-memory budget helpers for window attention CuTeDSL kernels.
"""
from typing import Optional

import torch


def _get_smem_budget_bytes() -> int:
    """Return the per-block dynamic SMEM limit (with optin) in bytes.

    SM-architecture → optin SMEM per block
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SM70 (V100)                           :  96 KB
    SM80 / SM86 (A100, RTX 3090)          : 100 KB
    SM89 (L40, RTX 4090)                  :  99 KB
    SM90 (H100)                           : 164 KB (practical limit)
    SM100 (B100/B200, sm_100)             : 256 KB (safe tile sizing)
    SM120 (Blackwell GeForce, sm_120)     :  99 KB

    ``major * 10 + minor`` from PyTorch: 10.0 → 100, 12.0 → 120.  SM120 must use
    the smaller budget; treating it like SM100 would oversubscribe shared memory.
    """
    if not torch.cuda.is_available():
        return 48 * 1024

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        major, minor = props.major, props.minor
        sm = major * 10 + minor
        if sm >= 120:
            return 99 * 1024   # Blackwell GeForce (sm_120)
        elif sm >= 100:
            return 256 * 1024  # Datacenter Blackwell (sm_100, B200)
        elif sm >= 90:
            return 164 * 1024  # H100
        elif sm >= 80:
            return 99 * 1024   # A100, RTX 30/40xx, L40
        elif sm >= 70:
            return 96 * 1024   # V100
    except Exception:
        pass

    return 48 * 1024  # conservative fallback


def _choose_tile_n(
    seq_len: int,
    head_dim: int = 64,
    tile_m: int = 64,
    smem_budget_bytes: Optional[int] = None,
) -> int:
    """Choose tile_n for BF16 (2-byte elements).

    SMEM layout (num_stages=1)::

        sQ : tile_m  × head_dim × 2B
        sK : tile_n  × head_dim × 2B
        sV : tile_n  × head_dim × 2B

    Strategy
    --------
    Single-pass (seq_len fits in one tile)
        Fill the full SMEM budget — there is only one pass so high occupancy
        is not critical, and a larger tile means fewer MMA fragments overall.

    Multi-pass (streaming)
        Cap SMEM at half the budget so that *two* CTAs fit per SM.  The
        doubled occupancy hides memory latency and cuts wave-quantization
        overhead far more than a wider tile would help.

    Rounded down to a multiple of 16 (MMA-K for m16n8k16).
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes     = tile_m * head_dim * 2
    kv_row_bytes = head_dim * 2

    # Largest tile_n that fits in the full budget (mask is uint8 in gmem, not SMEM).
    max_tile_n_full = (smem_budget_bytes - sQ_bytes) // (2 * kv_row_bytes)
    max_tile_n_full = max((max_tile_n_full // 16) * 16, 16)

    if seq_len <= max_tile_n_full:
        # Single-pass: fill SMEM.
        capped = min(seq_len, max_tile_n_full)
        return max(16, (capped // 16) * 16) if capped >= 16 else max(capped, 8)

    # Multi-pass: use half the budget, with 2-stage K+V double-buffer.
    half_budget    = smem_budget_bytes // 2
    max_tile_n_half = max((half_budget - sQ_bytes) // (4 * kv_row_bytes), 0)
    max_tile_n_half = max((max_tile_n_half // 16) * 16, 16)
    capped = min(seq_len, max_tile_n_half)
    return max(16, (capped // 16) * 16) if capped >= 16 else max(capped, 8)


def _tf32_hybrid_smem_bytes(
    tile_n: int,
    head_dim: int,
    tile_m: int = 64,
    num_stages: int = 1,
    *,
    include_mask_tile: bool = True,
) -> int:
    """Bytes for TF32 hybrid kernel SMEM: FP32 ``sQ``/``sK``, BF16 ``sV``.

    ``include_mask_tile`` is ignored (uint8 Swin mask is read from gmem).
    """
    del include_mask_tile
    return (
        tile_m * head_dim * 4
        + tile_n * head_dim * 4 * num_stages
        + tile_n * head_dim * 2 * num_stages
    )


def _choose_tile_n_tf32(
    seq_len: int,
    head_dim: int = 64,
    tile_m: int = 64,
    smem_budget_bytes: Optional[int] = None,
) -> int:
    """Choose tile_n for TF32_ACC_FP32 (hybrid FP32 Q/K + BF16 V).

    SMEM layout (matches ``WindowAttnFwdTF32``)::

        sQ : tile_m  × head_dim × 4B
        sK : tile_n  × head_dim × 4B × num_stages
        sV : tile_n  × head_dim × 2B × num_stages

    Single-pass uses ``num_stages=1``; streaming uses ``num_stages=2`` (K/V
    double-buffer) with half the SMEM budget for occupancy, same as BF16.
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes = tile_m * head_dim * 4
    kv_single = head_dim * 6
    kv_stream = head_dim * 12

    max_tile_n_full = (smem_budget_bytes - sQ_bytes) // kv_single
    max_tile_n_full = max((max_tile_n_full // 16) * 16, 16)

    if seq_len <= max_tile_n_full:
        capped = min(seq_len, max_tile_n_full)
        return max(16, (capped // 16) * 16) if capped >= 16 else max(capped, 8)

    half_budget = smem_budget_bytes // 2
    max_tile_n_half = max((half_budget - sQ_bytes) // kv_stream, 0)
    max_tile_n_half = max((max_tile_n_half // 16) * 16, 16)
    capped = min(seq_len, max_tile_n_half)
    return max(16, (capped // 16) * 16) if capped >= 16 else max(capped, 8)
