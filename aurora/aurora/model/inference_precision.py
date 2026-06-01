"""Inference precision router for Aurora inference presets.

Six named paths:

1. ``fp32``             — PyTorch FP32 (Swin + native Perceiver)
2. ``pytorch_autocast`` — PyTorch backbone BF16 autocast (no custom Swin kernels)
3. ``fast_fp32``        — Triton Swin fusions + native Perceiver
4. ``tf32_1x``          — ``fast_fp32`` + CuTe 1×TF32 window attention
5. ``bf16_mixed``       — ``fast_fp32`` + explicit BF16 CuTe window attention
6. ``full_bf16``        — Full-model BF16 mixed precision + Perceiver FlashAttention

Custom Triton/CuTe Swin paths never run inside ``torch.autocast``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

BackboneComputeDtype = Literal["float32", "bfloat16"]


class AuroraInferencePrecision(str, Enum):
    """Named inference precision presets for Aurora."""

    FP32 = "fp32"
    """PyTorch FP32 Swin + naive Perceiver."""

    PYTORCH_AUTOCAST = "pytorch_autocast"
    """PyTorch backbone BF16 autocast; no Triton/CuTe Swin kernels."""

    FAST_FP32 = "fast_fp32"
    """Triton Swin fusions (FP32) + native Perceiver."""

    TF32_1X = "tf32_1x"
    """``fast_fp32`` + CuTe 1×TF32 window attention."""

    BF16_MIXED = "bf16_mixed"
    """``fast_fp32`` + CuTe BF16 window attention (BF16 only in CuTe; native Perceiver)."""

    FULL_BF16 = "full_bf16"
    """``bf16_mixed`` Swin + encoder/decoder BF16 autocast; Perceiver FA when seqlen≥16."""


CudaGraphScope = Literal["off", "backbone", "full_gpu"]


@dataclass(frozen=True)
class AuroraInferenceConfig:
    """Resolved inference settings for one precision preset."""

    precision: AuroraInferencePrecision
    autocast_backbone: bool
    backbone_compute_dtype: BackboneComputeDtype
    use_triton_layout: bool
    use_triton_adaln: bool
    use_triton_mlp: bool
    use_cute_window_attn: bool
    use_triton_perceiver_ln_fusion: bool
    use_perceiver_flash_attn: bool
    autocast_encoder_decoder: bool
    cuda_graph_scope: CudaGraphScope
    cuda_graph_recommended: bool

    def validate(self) -> None:
        if self.use_triton_perceiver_ln_fusion:
            raise ValueError(
                "Perceiver Triton LN fusion is disabled for inference presets; "
                "encoder/decoder remain PyTorch naive."
            )
        uses_custom_swin = any(
            (
                self.use_triton_layout,
                self.use_triton_adaln,
                self.use_triton_mlp,
                self.use_cute_window_attn,
            )
        )
        if uses_custom_swin and self.autocast_backbone:
            raise ValueError(
                "Custom Triton/CuTe Swin paths cannot be combined with backbone autocast."
            )
        if self.precision == AuroraInferencePrecision.FP32:
            if uses_custom_swin or self.autocast_backbone or self.use_perceiver_flash_attn:
                raise ValueError("FP32 preset must not enable Triton/CuTe/autocast/Perceiver FA.")
            if self.autocast_encoder_decoder:
                raise ValueError("FP32 preset must not enable encoder/decoder autocast.")
        if self.precision == AuroraInferencePrecision.PYTORCH_AUTOCAST:
            if uses_custom_swin or self.use_perceiver_flash_attn:
                raise ValueError("PYTORCH_AUTOCAST preset must not enable Triton/CuTe/Perceiver FA.")
            if self.autocast_encoder_decoder:
                raise ValueError("PYTORCH_AUTOCAST preset must not enable encoder/decoder autocast.")
            if not self.autocast_backbone:
                raise ValueError("PYTORCH_AUTOCAST preset requires autocast_backbone=True.")
        if self.precision == AuroraInferencePrecision.FAST_FP32:
            if self.autocast_backbone or self.use_cute_window_attn:
                raise ValueError("FAST_FP32 preset requires autocast off and CuTe off.")
            if not (self.use_triton_layout and self.use_triton_adaln and self.use_triton_mlp):
                raise ValueError("FAST_FP32 preset requires all Triton Swin fusions enabled.")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("FAST_FP32 preset requires native Perceiver (no FA, no E/D autocast).")
        if self.precision == AuroraInferencePrecision.TF32_1X:
            if self.autocast_backbone or self.backbone_compute_dtype != "float32":
                raise ValueError("TF32_1X preset requires FP32 backbone compute.")
            if not (
                self.use_triton_layout
                and self.use_triton_adaln
                and self.use_triton_mlp
                and self.use_cute_window_attn
            ):
                raise ValueError("TF32_1X preset requires Triton Swin + CuTe.")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("TF32_1X preset requires native Perceiver (no FA, no E/D autocast).")
        if self.precision == AuroraInferencePrecision.BF16_MIXED:
            if self.autocast_backbone or self.backbone_compute_dtype != "bfloat16":
                raise ValueError("BF16_MIXED preset requires explicit BF16 backbone compute.")
            if not (
                self.use_triton_layout
                and self.use_triton_adaln
                and self.use_triton_mlp
                and self.use_cute_window_attn
            ):
                raise ValueError("BF16_MIXED preset requires Triton Swin + CuTe.")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("BF16_MIXED preset requires native Perceiver (no FA, no E/D autocast).")
        if self.precision == AuroraInferencePrecision.FULL_BF16:
            if self.autocast_backbone or self.backbone_compute_dtype != "bfloat16":
                raise ValueError("FULL_BF16 preset requires explicit BF16 backbone compute.")
            if not self.autocast_encoder_decoder:
                raise ValueError("FULL_BF16 preset requires encoder/decoder BF16 autocast.")
            if not (
                self.use_triton_layout
                and self.use_triton_adaln
                and self.use_triton_mlp
                and self.use_cute_window_attn
                and self.use_perceiver_flash_attn
            ):
                raise ValueError("FULL_BF16 preset requires Triton Swin + CuTe + Perceiver FA.")


_PRESETS: dict[AuroraInferencePrecision, AuroraInferenceConfig] = {
    AuroraInferencePrecision.FP32: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FP32,
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        use_triton_layout=False,
        use_triton_adaln=False,
        use_triton_mlp=False,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    AuroraInferencePrecision.PYTORCH_AUTOCAST: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.PYTORCH_AUTOCAST,
        autocast_backbone=True,
        backbone_compute_dtype="float32",
        use_triton_layout=False,
        use_triton_adaln=False,
        use_triton_mlp=False,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    AuroraInferencePrecision.FAST_FP32: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FAST_FP32,
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    AuroraInferencePrecision.TF32_1X: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.TF32_1X,
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="full_gpu",
        cuda_graph_recommended=True,
    ),
    AuroraInferencePrecision.BF16_MIXED: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.BF16_MIXED,
        autocast_backbone=False,
        backbone_compute_dtype="bfloat16",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="full_gpu",
        cuda_graph_recommended=True,
    ),
    AuroraInferencePrecision.FULL_BF16: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FULL_BF16,
        autocast_backbone=False,
        backbone_compute_dtype="bfloat16",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=True,
        autocast_encoder_decoder=True,
        cuda_graph_scope="full_gpu",
        cuda_graph_recommended=True,
    ),
}


def parse_inference_precision(value: str | AuroraInferencePrecision) -> AuroraInferencePrecision:
    if isinstance(value, AuroraInferencePrecision):
        return value
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "fast_fp32_triton":
        normalized = "fast_fp32"
    if normalized in {"tf32", "1x_tf32"}:
        normalized = "tf32_1x"
    if normalized in {"fullbf16", "bf16_full"}:
        normalized = "full_bf16"
    try:
        return AuroraInferencePrecision(normalized)
    except ValueError as exc:
        valid = ", ".join(p.value for p in AuroraInferencePrecision)
        raise ValueError(f"Unknown inference precision {value!r}; expected one of: {valid}") from exc


def resolve_inference_config(
    precision: str | AuroraInferencePrecision | None,
    *,
    enable_cuda_graph: bool = False,
) -> AuroraInferenceConfig | None:
    """Return the preset config, optionally enabling CUDA graph capture."""
    if precision is None:
        return None
    preset = parse_inference_precision(precision)
    cfg = _PRESETS[preset]
    if not enable_cuda_graph:
        return cfg
    if cfg.cuda_graph_scope == "off":
        raise ValueError(
            f"CUDA graph capture is not supported for precision={preset.value!r}. "
            "Use tf32_1x, bf16_mixed, or full_bf16."
        )
    cfg.validate()
    return cfg


def apply_inference_config(
    precision: str | AuroraInferencePrecision,
    *,
    enable_cuda_graph: bool = False,
) -> dict[str, bool | str]:
    """Expand a precision preset into Aurora constructor kwargs."""
    cfg = resolve_inference_config(precision, enable_cuda_graph=enable_cuda_graph)
    assert cfg is not None
    cfg.validate()
    return {
        "autocast": cfg.autocast_backbone,
        "backbone_compute_dtype": cfg.backbone_compute_dtype,
        "use_triton_layout": cfg.use_triton_layout,
        "use_triton_adaln": cfg.use_triton_adaln,
        "use_triton_mlp": cfg.use_triton_mlp,
        "use_cute_window_attn": cfg.use_cute_window_attn,
        "use_triton_perceiver_ln_fusion": cfg.use_triton_perceiver_ln_fusion,
        "use_perceiver_flash_attn": cfg.use_perceiver_flash_attn,
        "autocast_encoder_decoder": cfg.autocast_encoder_decoder,
    }
