"""Inference precision router for Aurora Swin3D kernel presets.

Perceiver encoder/decoder stay on the default PyTorch path (optional FlashAttention).
Custom Triton/CuTe optimizations apply to the Swin3D backbone only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AuroraInferencePrecision(str, Enum):
    """Named inference precision presets for Aurora."""

    FP32 = "fp32"
    """Strict FP32 Swin path: no Triton/CuTe kernels, no backbone autocast."""

    FAST_FP32 = "fast_fp32"
    """FP32 I/O with TF32 CuTe attention and Triton Swin fusions."""

    BF16_MIXED = "bf16_mixed"
    """BF16 backbone autocast with CuTe BF16 attention and Triton Swin fusions."""


CudaGraphScope = Literal["off", "backbone", "full_gpu"]


@dataclass(frozen=True)
class AuroraInferenceConfig:
    """Resolved Swin3D inference settings for one precision preset."""

    precision: AuroraInferencePrecision
    autocast_backbone: bool
    use_triton_layout: bool
    use_triton_adaln: bool
    use_triton_mlp: bool
    use_cute_window_attn: bool
    use_triton_perceiver_ln_fusion: bool
    cuda_graph_scope: CudaGraphScope
    cuda_graph_recommended: bool

    def validate(self) -> None:
        if self.use_triton_perceiver_ln_fusion:
            raise ValueError(
                "Perceiver Triton LN fusion is disabled for inference presets; "
                "encoder/decoder remain PyTorch naive."
            )
        if self.precision == AuroraInferencePrecision.FP32:
            if any(
                (
                    self.use_triton_layout,
                    self.use_triton_adaln,
                    self.use_triton_mlp,
                    self.use_cute_window_attn,
                    self.autocast_backbone,
                )
            ):
                raise ValueError("FP32 preset must not enable Triton/CuTe/autocast.")
        if self.precision == AuroraInferencePrecision.FAST_FP32 and self.autocast_backbone:
            raise ValueError("FAST_FP32 preset requires autocast_backbone=False.")
        if self.precision == AuroraInferencePrecision.BF16_MIXED and not self.autocast_backbone:
            raise ValueError("BF16_MIXED preset requires autocast_backbone=True.")


_PRESETS: dict[AuroraInferencePrecision, AuroraInferenceConfig] = {
    AuroraInferencePrecision.FP32: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FP32,
        autocast_backbone=False,
        use_triton_layout=False,
        use_triton_adaln=False,
        use_triton_mlp=False,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    AuroraInferencePrecision.FAST_FP32: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.FAST_FP32,
        autocast_backbone=False,
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        cuda_graph_scope="full_gpu",
        cuda_graph_recommended=True,
    ),
    AuroraInferencePrecision.BF16_MIXED: AuroraInferenceConfig(
        precision=AuroraInferencePrecision.BF16_MIXED,
        autocast_backbone=True,
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=True,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        cuda_graph_scope="full_gpu",
        cuda_graph_recommended=True,
    ),
}


def parse_inference_precision(value: str | AuroraInferencePrecision) -> AuroraInferencePrecision:
    if isinstance(value, AuroraInferencePrecision):
        return value
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "fp32": AuroraInferencePrecision.FP32,
        "float32": AuroraInferencePrecision.FP32,
        "strict_fp32": AuroraInferencePrecision.FP32,
        "fast_fp32": AuroraInferencePrecision.FAST_FP32,
        "tf32": AuroraInferencePrecision.FAST_FP32,
        "bf16": AuroraInferencePrecision.BF16_MIXED,
        "bf16_mixed": AuroraInferencePrecision.BF16_MIXED,
        "bfloat16": AuroraInferencePrecision.BF16_MIXED,
    }
    if normalized in aliases:
        return aliases[normalized]
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
            "Use fast_fp32 or bf16_mixed."
        )
    cfg.validate()
    return cfg


def apply_inference_config(
    precision: str | AuroraInferencePrecision,
    *,
    enable_cuda_graph: bool = False,
) -> dict[str, bool]:
    """Expand a precision preset into Aurora constructor kwargs."""
    cfg = resolve_inference_config(precision, enable_cuda_graph=enable_cuda_graph)
    assert cfg is not None
    cfg.validate()
    return {
        "autocast": cfg.autocast_backbone,
        "use_triton_layout": cfg.use_triton_layout,
        "use_triton_adaln": cfg.use_triton_adaln,
        "use_triton_mlp": cfg.use_triton_mlp,
        "use_cute_window_attn": cfg.use_cute_window_attn,
        "use_triton_perceiver_ln_fusion": cfg.use_triton_perceiver_ln_fusion,
    }
