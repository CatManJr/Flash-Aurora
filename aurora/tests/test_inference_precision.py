"""Tests for Aurora inference precision router presets."""

from __future__ import annotations

import pytest

from aurora.model.inference_precision import (
    AuroraInferencePrecision,
    apply_inference_config,
    parse_inference_precision,
    resolve_inference_config,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fp32", AuroraInferencePrecision.FP32),
        ("fast_fp32", AuroraInferencePrecision.FAST_FP32),
        ("bf16_mixed", AuroraInferencePrecision.BF16_MIXED),
        ("tf32", AuroraInferencePrecision.FAST_FP32),
        ("bfloat16", AuroraInferencePrecision.BF16_MIXED),
    ],
)
def test_parse_inference_precision_aliases(raw: str, expected: AuroraInferencePrecision) -> None:
    assert parse_inference_precision(raw) == expected


def test_fp32_preset_is_strict_pytorch() -> None:
    cfg = resolve_inference_config("fp32")
    assert cfg is not None
    assert cfg.autocast_backbone is False
    assert cfg.use_cute_window_attn is False
    assert cfg.use_triton_layout is False
    assert cfg.cuda_graph_scope == "off"


def test_fast_fp32_preset_enables_swin_kernels() -> None:
    cfg = resolve_inference_config("fast_fp32")
    assert cfg is not None
    assert cfg.autocast_backbone is False
    assert cfg.use_cute_window_attn is True
    assert cfg.use_triton_mlp is True
    assert cfg.cuda_graph_scope == "full_gpu"
    assert cfg.use_triton_perceiver_ln_fusion is False


def test_bf16_mixed_preset_enables_autocast() -> None:
    cfg = resolve_inference_config("bf16_mixed")
    assert cfg is not None
    assert cfg.autocast_backbone is True
    assert cfg.use_cute_window_attn is True
    assert cfg.cuda_graph_scope == "full_gpu"


def test_apply_inference_config_expands_constructor_kwargs() -> None:
    kw = apply_inference_config("fast_fp32")
    assert kw == {
        "autocast": False,
        "use_triton_layout": True,
        "use_triton_adaln": True,
        "use_triton_mlp": True,
        "use_cute_window_attn": True,
        "use_triton_perceiver_ln_fusion": False,
    }


def test_fp32_rejects_cuda_graph_enable() -> None:
    with pytest.raises(ValueError, match="CUDA graph capture is not supported"):
        resolve_inference_config("fp32", enable_cuda_graph=True)


def test_resolve_local_checkpoint_from_autodl_dir() -> None:
    from aurora.model.checkpoint_local import DEFAULT_CHECKPOINT_DIR, resolve_checkpoint_path

    path = resolve_checkpoint_path(
        filename="aurora-0.25-small-pretrained.ckpt",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        allow_hub_download=False,
    )
    assert path.is_file()
    assert path.parent == DEFAULT_CHECKPOINT_DIR.resolve()


def test_aurora_constructor_applies_fast_fp32_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="fast_fp32")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.FAST_FP32
    assert model.autocast is False
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.use_triton_layout is True
    assert block.attn.use_cute_window_attn is True
    assert model.encoder.level_agg.use_triton_ln_residual_fusion is False
