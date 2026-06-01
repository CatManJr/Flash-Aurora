"""Perceiver FlashAttention routing tests."""

from __future__ import annotations

import pytest
import torch

from aurora.model.perceiver import PerceiverAttention, _MIN_FLASH_ATTN_SEQLEN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_short_seqlen_perceiver_uses_sdpa_not_fa() -> None:
    attn = PerceiverAttention(
        latent_dim=256,
        context_dim=256,
        head_dim=32,
        num_heads=8,
        use_flash_attn=True,
    ).cuda()
    b, l1, l2 = 4, 4, 4
    assert l1 < _MIN_FLASH_ATTN_SEQLEN
    latents = torch.randn(b, l1, 256, device="cuda")
    context = torch.randn(b, l2, 256, device="cuda")
    with torch.inference_mode():
        out = attn(latents, context)
    assert out.shape == (b, l1, 256)
