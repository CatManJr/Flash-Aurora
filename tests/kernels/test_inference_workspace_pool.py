"""Copyright (c) Catman Jr. Licensed under the MIT license.

Tests for :class:`InferenceWorkspacePool` and backbone wiring (Stage D3).

Requires CUDA for backbone parity; pool unit tests run on CUDA when available.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from flash_aurora.aurora.model.swin3d import Swin3DTransformerBackbone
from flash_aurora.aurora.model.workspace_pool import InferenceWorkspacePool

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for backbone workspace pool tests",
)


@requires_cuda
def test_pool_reuses_storage() -> None:
    pool = InferenceWorkspacePool()
    shape = (2, 128, 512)
    a = pool.get("k", shape, device=torch.device("cuda"), dtype=torch.float32)
    b = pool.get("k", shape, device=torch.device("cuda"), dtype=torch.float32)
    assert a.data_ptr() == b.data_ptr()


@requires_cuda
def test_pool_reallocates_on_shape_change() -> None:
    pool = InferenceWorkspacePool()
    dev = torch.device("cuda")
    a = pool.get("k", (1, 10, 256), device=dev, dtype=torch.float32)
    b = pool.get("k", (1, 10, 512), device=dev, dtype=torch.float32)
    assert a.data_ptr() != b.data_ptr()
    assert b.numel() > a.numel()


@requires_cuda
def test_backbone_workspace_pool_matches_reference() -> None:
    torch.manual_seed(11)
    kwargs = dict(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode="single",
    )
    b_ref = Swin3DTransformerBackbone(**kwargs).cuda().eval()
    pool = InferenceWorkspacePool()
    b_pool = Swin3DTransformerBackbone(**kwargs, workspace_pool=pool).cuda().eval()
    b_pool.load_state_dict(b_ref.state_dict())

    C, H, W = 4, 32, 64
    L = C * H * W
    x = torch.randn(1, L, 256, device="cuda", dtype=torch.float32)
    lead = timedelta(hours=6)
    with torch.no_grad():
        y_ref = b_ref(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
        y_pool = b_pool(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
    torch.testing.assert_close(y_pool, y_ref, rtol=1e-5, atol=1e-5)
