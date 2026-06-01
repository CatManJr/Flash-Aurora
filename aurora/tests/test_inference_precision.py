"""Tests for Aurora inference precision router presets."""

from __future__ import annotations

import warnings

import pytest

import torch

from aurora.model.inference_precision import (
    AuroraInferenceConfig,
    AuroraInferencePrecision,
    apply_inference_config,
    parse_inference_precision,
    resolve_inference_config,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fp32", AuroraInferencePrecision.FP32),
        ("pytorch_autocast", AuroraInferencePrecision.PYTORCH_AUTOCAST),
        ("fast_fp32", AuroraInferencePrecision.FAST_FP32),
        ("tf32_1x", AuroraInferencePrecision.TF32_1X),
        ("bf16_mixed", AuroraInferencePrecision.BF16_MIXED),
    ],
)
def test_parse_inference_precision(raw: str, expected: AuroraInferencePrecision) -> None:
    assert parse_inference_precision(raw) == expected


def test_full_bf16_alias_maps_to_bf16_mixed() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert parse_inference_precision("full_bf16") == AuroraInferencePrecision.BF16_MIXED
    assert any("full_bf16 is deprecated" in str(w.message) for w in caught)


def test_fp32_preset_is_strict_pytorch() -> None:
    cfg = resolve_inference_config("fp32")
    assert cfg is not None
    assert cfg.autocast_backbone is False
    assert cfg.use_cute_window_attn is False
    assert cfg.use_triton_layout is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.backbone_matmul_bf16 is False


def test_pytorch_autocast_preset() -> None:
    cfg = resolve_inference_config("pytorch_autocast")
    assert cfg is not None
    assert cfg.autocast_backbone is True
    assert cfg.use_triton_layout is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.backbone_matmul_bf16 is False


def test_fast_fp32_preset_is_triton_with_native_perceiver() -> None:
    cfg = resolve_inference_config("fast_fp32")
    assert cfg is not None
    assert cfg.use_triton_layout is True
    assert cfg.use_triton_mlp is True
    assert cfg.use_cute_window_attn is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.backbone_matmul_bf16 is False


def test_tf32_1x_preset_adds_cute() -> None:
    cfg = resolve_inference_config("tf32_1x")
    assert cfg is not None
    assert cfg.use_cute_window_attn is True
    assert cfg.backbone_compute_dtype == "float32"
    assert cfg.backbone_matmul_bf16 is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.cuda_graph_scope == "backbone"


def test_bf16_mixed_preset_uses_explicit_bf16_backbone() -> None:
    cfg = resolve_inference_config("bf16_mixed")
    assert cfg is not None
    assert cfg.backbone_compute_dtype == "bfloat16"
    assert cfg.backbone_matmul_bf16 is True
    assert cfg.use_cute_window_attn is True
    assert cfg.autocast_backbone is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.cuda_graph_scope == "backbone"


def test_custom_ops_cannot_combine_with_autocast() -> None:
    cfg = AuroraInferenceConfig(
        precision=AuroraInferencePrecision.TF32_1X,
        autocast_backbone=True,
        backbone_compute_dtype="float32",
        backbone_matmul_bf16=False,
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    )
    with pytest.raises(ValueError, match="cannot be combined with backbone autocast"):
        cfg.validate()


def test_apply_inference_config_expands_constructor_kwargs() -> None:
    assert apply_inference_config("fast_fp32") == {
        "autocast": False,
        "backbone_compute_dtype": "float32",
        "backbone_matmul_bf16": False,
        "use_triton_layout": True,
        "use_triton_adaln": True,
        "use_triton_mlp": True,
        "use_cute_window_attn": False,
        "use_triton_perceiver_ln_fusion": False,
        "use_perceiver_flash_attn": False,
        "autocast_encoder_decoder": False,
    }


def test_fp32_rejects_cuda_graph_enable() -> None:
    with pytest.raises(ValueError, match="CUDA graph capture is not supported"):
        resolve_inference_config("fp32", enable_cuda_graph=True)


def test_aurora_constructor_applies_tf32_1x_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="tf32_1x")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.TF32_1X
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.use_triton_layout is True
    assert block.attn.use_cute_window_attn is True


def test_aurora_constructor_applies_bf16_mixed_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="bf16_mixed")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.BF16_MIXED
    assert model.inference_config.backbone_matmul_bf16 is True
    assert model.cute_window_attn_dtype == torch.bfloat16
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.attn.cute_window_attn_dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_matmul_context_routes_linear_to_bf16() -> None:
    from aurora.model.custom_op_paths import backbone_bf16_matmul_context

    linear = torch.nn.Linear(32, 16, bias=True).cuda().float()
    x = torch.randn(4, 32, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        with backbone_bf16_matmul_context(enabled=True):
            y = linear(x)
    assert y.dtype == torch.bfloat16
