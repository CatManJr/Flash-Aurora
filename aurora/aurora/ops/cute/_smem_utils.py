"""Shared-memory budget helpers for window attention CuTeDSL kernels.

No CuTe / MLIR dependency — importable on CPU-only machines.
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
    SM100 (B100/B200, sm_100)             : 164 KB (safe tile sizing)
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
            return 164 * 1024  # Datacenter Blackwell (sm_100, B200)
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
    """Choose the largest tile_n that fits in SMEM for BF16 (2-byte elements).

    Aims for single-pass attention (tile_n ≥ seq_len) when SMEM allows.
    Falls back to a streaming multi-pass tile when N is too large.

    SMEM layout (num_stages=1)::

        sQ : tile_m  × head_dim × 2B
        sK : tile_n  × head_dim × 2B
        sV : tile_n  × head_dim × 2B
        ──────────────────────────────
        budget = sQ + sK + sV

    Rounded down to a multiple of 16 (MMA-K for m16n8k16 / TF32 m16n8k8).
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes     = tile_m * head_dim * 2
    kv_row_bytes = head_dim * 2
    max_tile_n   = (smem_budget_bytes - sQ_bytes) // (2 * kv_row_bytes)
    max_tile_n   = max((max_tile_n // 16) * 16, 16)
    capped = min(seq_len, max_tile_n)
    if capped >= 16:
        return max(16, (capped // 16) * 16)
    return max(capped, 8)


def _choose_tile_n_tf32(
    seq_len: int,
    head_dim: int = 64,
    tile_m: int = 64,
    smem_budget_bytes: Optional[int] = None,
) -> int:
    """Choose the largest tile_n that fits in SMEM for TF32/FP32 (4-byte elements).

    Same logic as :func:`_choose_tile_n` but with 4-byte elements.
    Rounded down to a multiple of 16.
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_smem_budget_bytes()

    sQ_bytes     = tile_m * head_dim * 4
    kv_row_bytes = head_dim * 4
    max_tile_n   = (smem_budget_bytes - sQ_bytes) // (2 * kv_row_bytes)
    max_tile_n   = max((max_tile_n // 16) * 16, 16)
    capped = min(seq_len, max_tile_n)
    if capped >= 16:
        return max(16, (capped // 16) * 16)
    return max(capped, 8)
