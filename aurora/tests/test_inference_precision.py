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
    assert cfg.use_triton_adaln is True
    assert cfg.use_triton_mlp is False
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
    assert cfg.backbone_matmul_tf32 is True
    assert cfg.window_attn_compute_dtype == "float32"
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.cuda_graph_scope == "backbone"


def test_bf16_mixed_preset_uses_explicit_bf16_backbone() -> None:
    cfg = resolve_inference_config("bf16_mixed")
    assert cfg is not None
    assert cfg.backbone_compute_dtype == "float32"
    assert cfg.window_attn_compute_dtype == "bfloat16"
    assert cfg.backbone_matmul_bf16 is True
    assert cfg.backbone_matmul_tf32 is True
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
        backbone_matmul_tf32=False,
        window_attn_compute_dtype="float32",
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
        "backbone_matmul_tf32": False,
        "window_attn_compute_dtype": "float32",
        "use_triton_layout": True,
        "use_triton_adaln": True,
        "use_triton_mlp": False,
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
    assert model.inference_config.backbone_matmul_tf32 is True
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.use_triton_layout is True
    assert block.attn.use_cute_window_attn is True


def test_aurora_constructor_applies_bf16_mixed_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="bf16_mixed")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.BF16_MIXED
    assert model.inference_config.backbone_matmul_bf16 is True
    assert model.inference_config.backbone_matmul_tf32 is True
    assert model.cute_window_attn_dtype == torch.bfloat16
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.attn.cute_window_attn_dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_fp32_backbone_matches_fp32_pytorch_path() -> None:
    """Triton layout+AdaLN with PyTorch GELU should match pure PyTorch backbone."""
    from datetime import timedelta

    from aurora.model.swin3d import Swin3DTransformerBackbone

    torch.manual_seed(0)
    kwargs = dict(
        embed_dim=128,
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        window_size=(2, 4, 4),
        use_lora=False,
    )
    b_fast = Swin3DTransformerBackbone(
        **kwargs,
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
    ).cuda().eval()
    b_ref = Swin3DTransformerBackbone(**kwargs).cuda().eval()
    b_ref.load_state_dict(b_fast.state_dict())

    C, H, W = 4, 16, 32
    L = C * H * W
    x = torch.randn(1, L, 128, device="cuda", dtype=torch.float32)
    lead = timedelta(hours=6)
    with torch.inference_mode():
        y_fast = b_fast(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
        y_ref = b_ref(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
    torch.testing.assert_close(y_fast, y_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_matmul_context_mlp_chain_and_norm_break() -> None:
    from aurora.model.custom_op_paths import backbone_matmul_context
    from aurora.model.swin3d import MLP

    mlp = MLP(32, hidden_features=64, out_features=16).cuda().float()
    norm = torch.nn.LayerNorm(16).cuda().float()
    x = torch.randn(4, 144, 32, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        with backbone_matmul_context(tf32=True, bf16=True):
            y = mlp(x)
            z = norm(y)
    assert y.dtype == torch.bfloat16
    assert z.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_matmul_context_hooks_qkv_linear() -> None:
    from aurora.model.custom_op_paths import backbone_bf16_matmul_context

    linear = torch.nn.Linear(128, 128 * 3, bias=True).cuda().float()
    x = torch.randn(8, 144, 128, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        with backbone_bf16_matmul_context(enabled=True):
            qkv = linear(x)
    assert qkv.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_tf32_matmul_context_enables_tf32_flags() -> None:
    from aurora.model.custom_op_paths import backbone_tf32_matmul_context

    with torch.inference_mode():
        with backbone_tf32_matmul_context(enabled=True):
            assert torch.get_float32_matmul_precision() == "high"
            assert torch.backends.cuda.matmul.allow_tf32 is True
    # Restored after context (default may vary; just ensure no exception on enter/exit).
