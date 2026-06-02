"""Shared Blackwell GeForce (sm_120a) TMA + pipeline load helpers.

Extracted from the BF16 Stream kernel so the TMA K/V load recipe — sm90/Hopper
swizzled SMEM layouts, bulk-tensor TMA atoms, and the ``PipelineTmaAsync``
mainloop barrier — lives in one place. These mirror the data-movement stack in
``cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`` and are
dtype-generic so any precision (BF16 today, others later) can reuse them.

All helpers are thin wrappers over CuTe DSL APIs and inline cleanly into both
``@cute.jit`` host trace (layout/atom construction) and ``@cute.kernel`` device
bodies (pipeline creation).
"""
from __future__ import annotations

import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils


def make_kv_tma_smem_layouts(
    mK_t: cute.Tensor,
    mV_t: cute.Tensor,
    tile_m: int,
    tile_n: int,
    tile_hdim: int,
    dtype,
    num_stages: int,
) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
    """Staged (outer + swizzle) SMEM layouts for K and V TMA loads.

    Uses the Hopper ``make_smem_layout_b`` helper so the SMEM swizzle matches the
    TMA descriptor for a ``(tile_m, tile_n, tile_hdim)`` GEMM-B operand.
    """
    tiler = (tile_m, tile_n, tile_hdim)
    sK_tma_layout = sm90_utils.make_smem_layout_b(
        cutlass_utils.LayoutEnum.from_tensor(mK_t), tiler, dtype, num_stages,
    )
    sV_tma_layout = sm90_utils.make_smem_layout_b(
        cutlass_utils.LayoutEnum.from_tensor(mV_t), tiler, dtype, num_stages,
    )
    return sK_tma_layout, sV_tma_layout


def make_tma_atom_and_tensor(
    tensor: cute.Tensor,
    smem_layout_staged: cute.ComposedLayout,
    smem_tile: tuple[int, int],
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """Bulk-tensor (TMA) G2S copy atom + the TMA-coordinate tensor for ``tensor``."""
    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        tensor,
        smem_layout,
        smem_tile,
    )


def make_kv_mainloop_pipeline(
    num_stages: int,
    mma_warps: int,
    sK_tma_layout: cute.ComposedLayout,
    sV_tma_layout: cute.ComposedLayout,
    barrier_storage_ptr,
    dtype,
) -> pipeline.PipelineTmaAsync:
    """``PipelineTmaAsync`` whose transaction count covers one K + one V tile.

    Single producer (the DMA warp) and ``mma_warps`` consumers, matching the
    blackwell_geforce 1-DMA / N-MMA warp split.
    """
    tma_copy_bytes = cute.size_in_bytes(
        dtype, cute.slice_(sK_tma_layout, (None, None, 0))
    ) + cute.size_in_bytes(dtype, cute.slice_(sV_tma_layout, (None, None, 0)))
    return pipeline.PipelineTmaAsync.create(
        num_stages=num_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, mma_warps),
        tx_count=tma_copy_bytes,
        barrier_storage=barrier_storage_ptr,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
