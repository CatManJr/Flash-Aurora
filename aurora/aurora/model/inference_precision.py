"""Inference precision router for Aurora inference presets.

Five named paths:

1. ``fp32``             — PyTorch FP32 (Swin + native Perceiver)
2. ``pytorch_autocast`` — PyTorch backbone BF16 autocast (no custom kernels)
3. ``fast_fp32``        — Triton layout + AdaLN (PyTorch GELU) + native Perceiver
4. ``tf32_1x``          — ``fast_fp32`` + TF32 backbone matmuls + CuTe TF32 window attention
5. ``bf16_mixed``       — ``fast_fp32`` + BF16 backbone matmuls + CuTe BF16 window attention

Custom Triton/CuTe Swin3D paths never run inside ``torch.autocast``. ``bf16_mixed`` runs BF16
Tensor Core only on MLP ``fc1→fc2`` (BF16 activations between them); QKV/proj and AdaLN modulation
stay FP32. ``F.layer_norm`` on BF16 MLP output uses FP32 statistics (like autocast).
"""

from __future__ import annotations

import warnings
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
    """Triton layout + AdaLN (PyTorch GELU, FP32) + native Perceiver."""

    TF32_1X = "tf32_1x"
    """``fast_fp32`` + TF32 ``F.linear`` matmuls + CuTe TF32 window attention (FP32 activations)."""

    BF16_MIXED = "bf16_mixed"
    """``fast_fp32`` + BF16 GEMM chain + FP32 LayerNorm + CuTe BF16 window attention."""


CudaGraphScope = Literal["off", "backbone", "full_gpu"]
"""``backbone`` captures Swin only (encoder/decoder stay eager). ``full_gpu`` is experimental."""


@dataclass(frozen=True)
class AuroraInferenceConfig:
    """Resolved inference settings for one precision preset."""

    precision: AuroraInferencePrecision
    autocast_backbone: bool
    backbone_compute_dtype: BackboneComputeDtype
    backbone_matmul_bf16: bool
    backbone_matmul_tf32: bool
    window_attn_compute_dtype: BackboneComputeDtype
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
        if self.backbone_matmul_bf16 and self.backbone_matmul_tf32:
            raise ValueError(
                "backbone_matmul_bf16 and backbone_matmul_tf32 are mutually exclusive."
            )
        if self.backbone_matmul_bf16 and self.backbone_compute_dtype != "float32":
            raise ValueError(
                "backbone_matmul_bf16 requires backbone_compute_dtype='float32' "
                "(FP32 activation chain; matmuls alone run BF16)."
            )
        if self.backbone_matmul_tf32 and self.backbone_compute_dtype != "float32":
            raise ValueError(
                "backbone_matmul_tf32 requires backbone_compute_dtype='float32' "
                "(FP32 activation chain; matmuls alone run TF32)."
            )
        if self.precision == AuroraInferencePrecision.FP32:
            if uses_custom_swin or self.autocast_backbone or self.use_perceiver_flash_attn:
                raise ValueError("FP32 preset must not enable Triton/CuTe/autocast/Perceiver FA.")
            if self.autocast_encoder_decoder or self.backbone_matmul_bf16 or self.backbone_matmul_tf32:
                raise ValueError("FP32 preset must not enable encoder/decoder autocast or approximate matmul.")
        if self.precision == AuroraInferencePrecision.PYTORCH_AUTOCAST:
            if uses_custom_swin or self.use_perceiver_flash_attn:
                raise ValueError("PYTORCH_AUTOCAST preset must not enable Triton/CuTe/Perceiver FA.")
            if self.autocast_encoder_decoder or self.backbone_matmul_bf16 or self.backbone_matmul_tf32:
                raise ValueError("PYTORCH_AUTOCAST preset must not enable encoder/decoder autocast or approximate matmul.")
            if not self.autocast_backbone:
                raise ValueError("PYTORCH_AUTOCAST preset requires autocast_backbone=True.")
        if self.precision == AuroraInferencePrecision.FAST_FP32:
            if self.autocast_backbone or self.use_cute_window_attn or self.backbone_matmul_bf16 or self.backbone_matmul_tf32:
                raise ValueError("FAST_FP32 preset requires autocast off, CuTe off, and approximate matmul off.")
            if not (self.use_triton_layout and self.use_triton_adaln):
                raise ValueError("FAST_FP32 preset requires Triton layout and AdaLN.")
            if self.use_triton_mlp:
                raise ValueError("FAST_FP32 preset must use PyTorch GELU (use_triton_mlp=False).")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("FAST_FP32 preset requires native Perceiver (no FA, no E/D autocast).")
        if self.precision == AuroraInferencePrecision.TF32_1X:
            if self.autocast_backbone or self.backbone_compute_dtype != "float32" or self.backbone_matmul_bf16:
                raise ValueError("TF32_1X preset requires FP32 backbone compute and no BF16 matmul.")
            if not self.backbone_matmul_tf32:
                raise ValueError("TF32_1X preset requires backbone_matmul_tf32=True.")
            if not (self.use_triton_layout and self.use_triton_adaln and self.use_cute_window_attn):
                raise ValueError("TF32_1X preset requires Triton layout/AdaLN + CuTe.")
            if self.use_triton_mlp:
                raise ValueError("TF32_1X preset must use PyTorch GELU (use_triton_mlp=False).")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("TF32_1X preset requires native Perceiver (no FA, no E/D autocast).")
        if self.precision == AuroraInferencePrecision.BF16_MIXED:
            if self.autocast_backbone:
                raise ValueError("BF16_MIXED preset must not use backbone autocast.")
            if not self.backbone_matmul_bf16 or self.backbone_matmul_tf32:
                raise ValueError("BF16_MIXED preset requires backbone_matmul_bf16=True and no TF32 matmul.")
            if self.window_attn_compute_dtype != "bfloat16":
                raise ValueError("BF16_MIXED preset requires CuTe BF16 window attention.")
            if not (self.use_triton_layout and self.use_triton_adaln and self.use_cute_window_attn):
                raise ValueError("BF16_MIXED preset requires Triton layout/AdaLN + CuTe.")
            if self.use_triton_mlp:
                raise ValueError("BF16_MIXED preset must use PyTorch GELU (use_triton_mlp=False).")
            if self.use_perceiver_flash_attn or self.autocast_encoder_decoder:
                raise ValueError("BF16_MIXED preset requires native Perceiver (no FA, no E/D autocast).")


_PRESETS: dict[AuroraInferencePrecision, AuroraInferenceConfig] = {
    AuroraInferencePrecision.FP32: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FP32,
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        backbone_matmul_bf16=False,
        backbone_matmul_tf32=False,
        window_attn_compute_dtype="float32",
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
        backbone_matmul_bf16=False,
        backbone_matmul_tf32=False,
        window_attn_compute_dtype="float32",
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
        backbone_matmul_bf16=False,
        backbone_matmul_tf32=False,
        window_attn_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
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
        backbone_matmul_bf16=False,
        backbone_matmul_tf32=True,
        window_attn_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="backbone",
        cuda_graph_recommended=True,
    ),
    AuroraInferencePrecision.BF16_MIXED: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.BF16_MIXED,
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        backbone_matmul_bf16=True,
        backbone_matmul_tf32=False,
        window_attn_compute_dtype="bfloat16",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        autocast_encoder_decoder=False,
        cuda_graph_scope="backbone",
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
    if normalized in {"full_bf16", "fullbf16", "bf16_full"}:
        warnings.warn(
            "full_bf16 is deprecated and removed; use bf16_mixed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "bf16_mixed"
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
            "Use tf32_1x or bf16_mixed."
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
        "backbone_matmul_bf16": cfg.backbone_matmul_bf16,
        "backbone_matmul_tf32": cfg.backbone_matmul_tf32,
        "window_attn_compute_dtype": cfg.window_attn_compute_dtype,
        "use_triton_layout": cfg.use_triton_layout,
        "use_triton_adaln": cfg.use_triton_adaln,
        "use_triton_mlp": cfg.use_triton_mlp,
        "use_cute_window_attn": cfg.use_cute_window_attn,
        "use_triton_perceiver_ln_fusion": cfg.use_triton_perceiver_ln_fusion,
        "use_perceiver_flash_attn": cfg.use_perceiver_flash_attn,
        "autocast_encoder_decoder": cfg.autocast_encoder_decoder,
    }
