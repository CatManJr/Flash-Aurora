"""Tests for custom-op / model-path dtype adapters."""

from __future__ import annotations

import pytest
import torch

from aurora.model.custom_op_paths import (
    align_binary_activations,
    can_use_cute_qkvpacked,
    can_use_triton_adaln,
)


def test_align_binary_activations_promotes_to_bf16() -> None:
    fp32 = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    bf16 = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    left, right = align_binary_activations(fp32, bf16)
    assert left.dtype == torch.bfloat16
    assert right.dtype == torch.bfloat16


def test_can_use_cute_qkvpacked_under_autocast() -> None:
    qkv = torch.randn(4, 144, 768, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        assert can_use_cute_qkvpacked(
            qkv,
            num_heads=8,
            head_dim=32,
            cute_enabled=True,
            training=False,
            attn_dropout=0.0,
        )


def test_prepare_backbone_input_casts_to_bf16() -> None:
    from aurora.model.custom_op_paths import prepare_backbone_input

    x = torch.randn(2, 8, 16, device="cuda", dtype=torch.float32)
    y = prepare_backbone_input(x, torch.bfloat16)
    assert y.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_can_use_triton_adaln_requires_fp32() -> None:
    fp32 = torch.randn(2, 64, 128, device="cuda", dtype=torch.float32)
    bf16 = fp32.to(torch.bfloat16)
    assert can_use_triton_adaln(fp32, enabled=True, training=False, drop_path_is_identity=True)
    assert not can_use_triton_adaln(
        bf16, enabled=True, training=False, drop_path_is_identity=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_hybrid_routing() -> None:
    from aurora.model.custom_op_paths import backbone_matmul_context
    from aurora.model.swin3d import MLP

    with torch.inference_mode():
        with backbone_matmul_context(tf32=True, bf16=True):
            linear = torch.nn.Linear(32, 96, bias=True).cuda().float()
            x = torch.randn(4, 144, 32, device="cuda", dtype=torch.float32)
            assert linear(x).dtype == torch.float32
            mlp = MLP(32, hidden_features=64, out_features=32).cuda().float()
            y = mlp(x)
    assert y.dtype == torch.bfloat16


def test_backbone_bf16_routing() -> None:
    from aurora.model.custom_op_paths import backbone_matmul_context
    from aurora.model.swin3d import MLP

    with torch.inference_mode():
        with backbone_matmul_context(tf32=False, bf16=True):
            linear = torch.nn.Linear(32, 96, bias=True).cuda().float()
            x = torch.randn(4, 144, 32, device="cuda", dtype=torch.float32)
            assert linear(x).dtype == torch.bfloat16
            mlp = MLP(32, hidden_features=64, out_features=32).cuda().float()
            y = mlp(x)
    assert y.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cast_activation_dtype_is_noop_when_matched() -> None:
    from aurora.model.custom_op_paths import cast_activation_dtype

    x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
    y = cast_activation_dtype(x, torch.bfloat16)
    assert y is x
