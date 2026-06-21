"""Tests for Aurora inference precision router presets."""

from __future__ import annotations

import pytest

import torch

from aurora.model.inference_precision import (
    AuroraInferenceConfig,
    AuroraInferencePrecision,
    BackboneMatmulLevel,
    EncoderDecoderMatmulLevel,
    apply_inference_config,
    build_inference_config,
    expand_precision_combos,
    parse_inference_precision,
    parse_precision_spec,
    resolve_inference_config,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fp32", AuroraInferencePrecision.FP32),
        ("pytorch_autocast", AuroraInferencePrecision.PYTORCH_AUTOCAST),
        ("fast_fp32", AuroraInferencePrecision.FAST_FP32),
        ("tf32", AuroraInferencePrecision.TF32),
        ("bf16_mixed", AuroraInferencePrecision.BF16_MIXED),
        ("bf16", AuroraInferencePrecision.BF16),
    ],
)
def test_parse_inference_precision(raw: str, expected: AuroraInferencePrecision) -> None:
    assert parse_inference_precision(raw) == expected


def test_unknown_precision_raises() -> None:
    with pytest.raises(ValueError, match="Unknown precision"):
        parse_inference_precision("tf32_1x")


def test_fp32_preset_is_strict_pytorch() -> None:
    cfg = resolve_inference_config("fp32")
    assert cfg is not None
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.FP32
    assert cfg.encoder_decoder_use_tensor_core is False
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
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.FP32
    assert cfg.encoder_decoder_use_tensor_core is False
    assert cfg.use_triton_layout is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.backbone_matmul_bf16 is False


def test_fast_fp32_preset_is_triton_with_native_perceiver() -> None:
    cfg = resolve_inference_config("fast_fp32")
    assert cfg is not None
    assert cfg.kernel_profile == "fast_fp32"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.FP32
    assert cfg.encoder_decoder_use_tensor_core is False
    assert cfg.use_triton_layout is True
    assert cfg.use_triton_adaln is True
    assert cfg.use_triton_mlp is False
    assert cfg.use_cute_window_attn is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.autocast_encoder_decoder is False
    assert cfg.backbone_matmul_bf16 is False


def test_tf32_preset_adds_cute() -> None:
    cfg = resolve_inference_config("tf32")
    assert cfg is not None
    assert cfg.kernel_profile == "tf32_backbone"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert cfg.use_cute_window_attn is True
    assert cfg.backbone_compute_dtype == "float32"
    assert cfg.backbone_matmul_bf16 is False
    assert cfg.backbone_matmul_tf32 is True
    assert cfg.window_attn_compute_dtype == "float32"
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.encoder_decoder_use_tensor_core is True
    assert cfg.cuda_graph_scope == "backbone"


def test_bf16_mixed_preset_is_attention_and_mlp_bf16_hybrid() -> None:
    cfg = resolve_inference_config("bf16_mixed")
    assert cfg is not None
    assert cfg.kernel_profile == "bf16_mixed_backbone"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.BF16_MIXED
    assert cfg.backbone_matmul_bf16 is True
    assert cfg.backbone_matmul_tf32 is True
    assert cfg.window_attn_compute_dtype == "bfloat16"
    assert cfg.use_cute_window_attn is True


def test_bf16_preset_is_full_bf16_linears() -> None:
    cfg = resolve_inference_config("bf16")
    assert cfg is not None
    assert cfg.kernel_profile == "bf16_mixed_backbone"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert cfg.backbone_compute_dtype == "float32"
    assert cfg.window_attn_compute_dtype == "bfloat16"
    assert cfg.backbone_matmul_bf16 is True
    assert cfg.backbone_matmul_tf32 is False
    assert cfg.use_cute_window_attn is True
    assert cfg.autocast_backbone is False
    assert cfg.use_perceiver_flash_attn is False
    assert cfg.encoder_decoder_use_tensor_core is True
    assert cfg.autocast_encoder_decoder is False
    assert cfg.cuda_graph_scope == "backbone"


def test_fast_fp32_plus_tensor_core_modifier() -> None:
    cfg = resolve_inference_config("fast_fp32+tensor_core")
    assert cfg is not None
    assert cfg.kernel_profile == "fast_fp32"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.FP32
    assert cfg.encoder_decoder_use_tensor_core is True
    assert cfg.use_cute_window_attn is False


def test_tf32_plus_no_tensor_core_modifier() -> None:
    cfg = resolve_inference_config("tf32+no_tensor_core")
    assert cfg is not None
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert cfg.encoder_decoder_use_tensor_core is False
    assert cfg.use_cute_window_attn is True


def test_backbone_matmul_level_override_on_fast_fp32() -> None:
    cfg = resolve_inference_config(
        "fast_fp32",
        backbone_matmul_level=BackboneMatmulLevel.TF32,
    )
    assert cfg is not None
    assert cfg.kernel_profile == "fast_fp32"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert cfg.backbone_matmul_tf32 is True
    assert cfg.use_cute_window_attn is False


def test_encoder_decoder_use_tensor_core_kwarg_override() -> None:
    cfg = resolve_inference_config(
        "fp32",
        encoder_decoder_use_tensor_core=True,
    )
    assert cfg is not None
    assert cfg.encoder_decoder_use_tensor_core is True
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.FP32


def test_custom_ops_cannot_combine_with_autocast() -> None:
    cfg = build_inference_config(
        precision=AuroraInferencePrecision.TF32,
        kernel_profile="tf32_backbone",
        backbone_matmul_level=BackboneMatmulLevel.TF32,
        encoder_decoder_matmul_level=EncoderDecoderMatmulLevel.FP32,
    )
    object.__setattr__(cfg, "autocast_backbone", True)
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
        "encoder_decoder_use_tensor_core": False,
        "backbone_matmul_level": "fp32",
        "encoder_decoder_matmul_level": "fp32",
        "inference_config_label": "fast_fp32",
    }


def test_parse_precision_combo_at_syntax() -> None:
    spec = parse_precision_spec("bf16@fp32")
    assert spec.named_preset is None
    assert spec.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert spec.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32


def test_parse_precision_combo_kv_syntax() -> None:
    spec = parse_precision_spec("backbone=tf32,encoder_decoder=fp32")
    assert spec.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert spec.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32


def test_encoder_decoder_bf16_combo_rejected() -> None:
    with pytest.raises(ValueError, match="encoder_decoder=bf16 is not supported"):
        parse_precision_spec("backbone=tf32,encoder_decoder=bf16")
    with pytest.raises(ValueError, match="encoder_decoder=bf16 is not supported"):
        parse_precision_spec("tf32@bf16")


def test_resolve_bf16_mixed_at_fp32_combo() -> None:
    cfg = resolve_inference_config("bf16_mixed@fp32")
    assert cfg is not None
    assert cfg.config_label == "bf16_mixed@fp32"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.BF16_MIXED
    assert cfg.backbone_matmul_bf16 is True
    assert cfg.backbone_matmul_tf32 is True
    assert cfg.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32


def test_bf16_mixed_vs_bf16_combo_differ_on_backbone_flags() -> None:
    mixed = resolve_inference_config("bf16_mixed@fp32")
    full = resolve_inference_config("bf16@fp32")
    assert mixed is not None and full is not None
    assert mixed.backbone_matmul_level == BackboneMatmulLevel.BF16_MIXED
    assert full.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert mixed.backbone_matmul_tf32 is True
    assert full.backbone_matmul_tf32 is False


def test_resolve_bf16_at_fp32_combo() -> None:
    cfg = resolve_inference_config("bf16@fp32")
    assert cfg is not None
    assert cfg.precision is None
    assert cfg.config_label == "bf16@fp32"
    assert cfg.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert cfg.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32
    assert cfg.encoder_decoder_use_tensor_core is False
    assert cfg.autocast_encoder_decoder is False


def test_expand_precision_combos_cartesian_product() -> None:
    combos = expand_precision_combos(["tf32", "bf16"], ["fp32", "tf32"])
    labels = [label for label, _ in combos]
    assert labels == ["tf32@fp32", "tf32@tf32", "bf16@fp32", "bf16@tf32"]
    assert len(combos) == 4


def test_aurora_constructor_combo_string() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="bf16@fp32")
    assert model.inference_config is not None
    assert model.inference_config.config_label == "bf16@fp32"
    assert model.inference_config.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert model.inference_config.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32
    assert model.encoder_decoder_use_tensor_core is False


def test_aurora_constructor_independent_level_kwargs() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(
        use_lora=False,
        inference_precision="tf32",
        encoder_decoder_matmul_level="fp32",
    )
    assert model.inference_config is not None
    assert model.inference_config.config_label == "tf32@fp32"
    assert model.inference_config.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert model.inference_config.encoder_decoder_matmul_level == EncoderDecoderMatmulLevel.FP32
    assert model.encoder_decoder_use_tensor_core is False


def test_fp32_rejects_cuda_graph_enable() -> None:
    with pytest.raises(ValueError, match="CUDA graph capture is not supported"):
        resolve_inference_config("fp32", enable_cuda_graph=True)


def test_aurora_prepare_encoder_batch_keeps_lat_lon_fp32() -> None:
    from aurora import Batch, Metadata
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, autocast=True).cuda().to(dtype=torch.bfloat16)
    batch = Batch(
        surf_vars={"2t": torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float32)},
        static_vars={"lsm": torch.randn(8, 16, device="cuda", dtype=torch.float32)},
        atmos_vars={"z": torch.randn(1, 2, 4, 8, 16, device="cuda", dtype=torch.float32)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 8, device="cuda", dtype=torch.float32),
            lon=torch.linspace(0, 360, 17, device="cuda", dtype=torch.float32)[:-1],
            atmos_levels=(100, 250, 500, 850),
            time=(),
        ),
    )
    _batch, transformed, _patch_res = model._prepare_encoder_batch(batch)
    assert transformed.metadata.lat.dtype == torch.float32
    assert transformed.metadata.lon.dtype == torch.float32


def test_aurora_constructor_applies_tf32_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="tf32")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.TF32
    assert model.inference_config.backbone_matmul_level == BackboneMatmulLevel.TF32
    assert model.inference_config.backbone_matmul_tf32 is True
    assert model.encoder_decoder_use_tensor_core is True
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.use_triton_layout is True
    assert block.attn.use_cute_window_attn is True


def test_aurora_constructor_applies_bf16_mixed_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="bf16_mixed")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.BF16_MIXED
    assert model.inference_config.backbone_matmul_level == BackboneMatmulLevel.BF16_MIXED
    assert model.inference_config.backbone_matmul_bf16 is True
    assert model.inference_config.backbone_matmul_tf32 is True
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.attn.cute_window_attn_dtype == torch.bfloat16
    assert block.attn.use_cute_window_attn is True


def test_aurora_constructor_applies_bf16_preset() -> None:
    from aurora.model.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision="bf16")
    assert model.inference_config is not None
    assert model.inference_config.precision == AuroraInferencePrecision.BF16
    assert model.inference_config.backbone_matmul_level == BackboneMatmulLevel.BF16
    assert model.inference_config.backbone_matmul_bf16 is True
    assert model.inference_config.backbone_matmul_tf32 is False
    assert model.encoder_decoder_use_tensor_core is True
    assert model.cute_window_attn_dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_encoder_decoder_routing_enables_tf32_flags() -> None:
    from aurora.model.custom_op_paths import run_with_encoder_decoder_routing

    def _noop() -> None:
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True

    with torch.inference_mode():
        run_with_encoder_decoder_routing(_noop, use_tensor_core=True)


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
def test_backbone_bf16_hybrid_hooks_qkv_linear_fp32() -> None:
    from aurora.model.custom_op_paths import backbone_matmul_context

    linear = torch.nn.Linear(128, 128 * 3, bias=True).cuda().float()
    x = torch.randn(8, 144, 128, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        with backbone_matmul_context(tf32=True, bf16=True):
            qkv = linear(x)
    assert qkv.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_hooks_qkv_linear_bf16() -> None:
    from aurora.model.custom_op_paths import backbone_bf16_matmul_context

    linear = torch.nn.Linear(128, 128 * 3, bias=True).cuda().float()
    x = torch.randn(8, 144, 128, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        with backbone_bf16_matmul_context(enabled=True):
            qkv = linear(x)
    assert qkv.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fused_attention_chain_on_full_bf16_context() -> None:
    from aurora.model.custom_op_paths import (
        backbone_bf16_matmul_context,
        use_bf16_fused_attention_chain,
    )

    with torch.inference_mode():
        with backbone_bf16_matmul_context(enabled=True):
            assert use_bf16_fused_attention_chain(
                use_cute_window_attn=True,
                cute_window_attn_dtype=torch.bfloat16,
                is_cuda=True,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_tf32_matmul_context_enables_tf32_flags() -> None:
    from aurora.model.custom_op_paths import backbone_tf32_matmul_context

    with torch.inference_mode():
        with backbone_tf32_matmul_context(enabled=True):
            assert torch.get_float32_matmul_precision() == "high"
            assert torch.backends.cuda.matmul.allow_tf32 is True
    # Restored after context (default may vary; just ensure no exception on enter/exit).
