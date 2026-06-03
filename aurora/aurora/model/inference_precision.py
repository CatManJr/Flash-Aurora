"""Inference precision router for Aurora inference presets.

Matmul routing is a **2D grid**:

* **Backbone** — :class:`BackboneMatmulLevel`: ``fp32`` | ``tf32`` | ``bf16_mixed``
* **Encoder/decoder** — ``encoder_decoder_use_tensor_core``: TF32 ``F.linear`` (FP32 activations)

Named presets are convenience points on the grid. Override E/D TC with ``+tensor_core`` /
``+no_tensor_core`` suffixes on the precision string, e.g. ``fast_fp32+tensor_core``.

Custom Triton/CuTe Swin paths never run inside ``torch.autocast``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Literal

BackboneComputeDtype = Literal["float32", "bfloat16"]
KernelProfile = Literal["baseline", "fast_fp32", "tf32_backbone", "bf16_mixed_backbone"]


class BackboneMatmulLevel(str, Enum):
    """Backbone ``F.linear`` matmul precision (activations stay FP32 unless autocast)."""

    FP32 = "fp32"
    """Strict FP32 cuBLAS on all backbone linears."""

    TF32 = "tf32"
    """TF32 Tensor Core on backbone linears (FP32 I/O)."""

    BF16_MIXED = "bf16_mixed"
    """TF32 on QKV/proj/patch; BF16 on MLP ``fc1``/``fc2`` (FP32 LayerNorm on MLP output)."""


class AuroraInferencePrecision(str, Enum):
    """Named inference precision presets (kernel profile + default matmul grid cell)."""

    FP32 = "fp32"
    PYTORCH_AUTOCAST = "pytorch_autocast"
    FAST_FP32 = "fast_fp32"
    TF32_1X = "tf32_1x"
    BF16_MIXED = "bf16_mixed"


CudaGraphScope = Literal["off", "backbone", "full_gpu"]


def backbone_matmul_flags(level: BackboneMatmulLevel) -> tuple[bool, bool]:
    """Return ``(backbone_matmul_bf16, backbone_matmul_tf32)`` for a backbone level."""
    if level == BackboneMatmulLevel.FP32:
        return False, False
    if level == BackboneMatmulLevel.TF32:
        return False, True
    return True, True


@dataclass(frozen=True)
class _KernelProfileSpec:
    autocast_backbone: bool
    backbone_compute_dtype: BackboneComputeDtype
    window_attn_compute_dtype: BackboneComputeDtype
    use_triton_layout: bool
    use_triton_adaln: bool
    use_triton_mlp: bool
    use_cute_window_attn: bool
    use_triton_perceiver_ln_fusion: bool
    use_perceiver_flash_attn: bool
    cuda_graph_scope: CudaGraphScope
    cuda_graph_recommended: bool


_KERNEL_PROFILES: dict[KernelProfile, _KernelProfileSpec] = {
    "baseline": _KernelProfileSpec(
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        window_attn_compute_dtype="float32",
        use_triton_layout=False,
        use_triton_adaln=False,
        use_triton_mlp=False,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    "fast_fp32": _KernelProfileSpec(
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        window_attn_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
        use_cute_window_attn=False,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        cuda_graph_scope="off",
        cuda_graph_recommended=False,
    ),
    "tf32_backbone": _KernelProfileSpec(
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        window_attn_compute_dtype="float32",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        cuda_graph_scope="backbone",
        cuda_graph_recommended=True,
    ),
    "bf16_mixed_backbone": _KernelProfileSpec(
        autocast_backbone=False,
        backbone_compute_dtype="float32",
        window_attn_compute_dtype="bfloat16",
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
        use_cute_window_attn=True,
        use_triton_perceiver_ln_fusion=False,
        use_perceiver_flash_attn=False,
        cuda_graph_scope="backbone",
        cuda_graph_recommended=True,
    ),
}


@dataclass(frozen=True)
class _PresetGridCell:
    kernel_profile: KernelProfile
    backbone_matmul_level: BackboneMatmulLevel
    encoder_decoder_use_tensor_core: bool
    autocast_backbone: bool | None = None


_PRESET_GRID: dict[AuroraInferencePrecision, _PresetGridCell] = {
    AuroraInferencePrecision.FP32: _PresetGridCell("baseline", BackboneMatmulLevel.FP32, False),
    AuroraInferencePrecision.PYTORCH_AUTOCAST: _PresetGridCell(
        "baseline", BackboneMatmulLevel.FP32, False, autocast_backbone=True
    ),
    AuroraInferencePrecision.FAST_FP32: _PresetGridCell(
        "fast_fp32", BackboneMatmulLevel.FP32, False
    ),
    AuroraInferencePrecision.TF32_1X: _PresetGridCell(
        "tf32_backbone", BackboneMatmulLevel.TF32, True
    ),
    AuroraInferencePrecision.BF16_MIXED: _PresetGridCell(
        "bf16_mixed_backbone", BackboneMatmulLevel.BF16_MIXED, True
    ),
}


@dataclass(frozen=True)
class AuroraInferenceConfig:
    """Resolved inference settings for one precision preset."""

    precision: AuroraInferencePrecision
    kernel_profile: KernelProfile
    backbone_matmul_level: BackboneMatmulLevel
    encoder_decoder_use_tensor_core: bool
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
        if self.autocast_encoder_decoder and self.encoder_decoder_use_tensor_core:
            raise ValueError(
                "autocast_encoder_decoder and encoder_decoder_use_tensor_core are mutually exclusive."
            )
        if self.encoder_decoder_use_tensor_core and self.backbone_compute_dtype != "float32":
            raise ValueError(
                "encoder_decoder_use_tensor_core requires backbone_compute_dtype='float32'."
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
        bf16, tf32 = backbone_matmul_flags(self.backbone_matmul_level)
        if self.backbone_matmul_bf16 != bf16 or self.backbone_matmul_tf32 != tf32:
            raise ValueError("backbone matmul flags disagree with backbone_matmul_level.")
        if self.backbone_matmul_level == BackboneMatmulLevel.BF16_MIXED:
            if self.kernel_profile != "bf16_mixed_backbone":
                raise ValueError(
                    "backbone_matmul_level=bf16_mixed requires kernel_profile=bf16_mixed_backbone."
                )
            if self.window_attn_compute_dtype != "bfloat16":
                raise ValueError("bf16_mixed backbone requires CuTe BF16 window attention.")
        if self.backbone_matmul_level == BackboneMatmulLevel.TF32 and not self.backbone_matmul_tf32:
            raise ValueError("backbone_matmul_level=tf32 requires backbone_matmul_tf32=True.")
        if self.backbone_matmul_level == BackboneMatmulLevel.FP32 and (
            self.backbone_matmul_bf16 or self.backbone_matmul_tf32
        ):
            raise ValueError("backbone_matmul_level=fp32 forbids approximate backbone matmul.")
        if self.kernel_profile == "baseline":
            if uses_custom_swin or self.use_perceiver_flash_attn:
                raise ValueError("baseline profile must not enable Triton/CuTe/Perceiver FA.")
            if self.autocast_backbone and self.precision != AuroraInferencePrecision.PYTORCH_AUTOCAST:
                raise ValueError(
                    "baseline profile enables backbone autocast only for PYTORCH_AUTOCAST."
                )
            if self.backbone_matmul_level != BackboneMatmulLevel.FP32:
                raise ValueError("baseline profile requires strict FP32 backbone matmul.")
        if self.kernel_profile == "fast_fp32":
            if self.use_cute_window_attn:
                raise ValueError("fast_fp32 profile must not enable CuTe window attention.")
            if self.use_triton_mlp:
                raise ValueError("fast_fp32 profile must use PyTorch GELU (use_triton_mlp=False).")
        if self.kernel_profile == "tf32_backbone":
            if not self.use_cute_window_attn or self.window_attn_compute_dtype != "float32":
                raise ValueError("tf32_backbone profile requires CuTe TF32 window attention.")
        if self.kernel_profile == "bf16_mixed_backbone":
            if not self.use_cute_window_attn or self.window_attn_compute_dtype != "bfloat16":
                raise ValueError("bf16_mixed_backbone profile requires CuTe BF16 window attention.")
        if self.precision == AuroraInferencePrecision.PYTORCH_AUTOCAST:
            if uses_custom_swin or self.use_perceiver_flash_attn:
                raise ValueError("PYTORCH_AUTOCAST must not enable Triton/CuTe/Perceiver FA.")
            if self.encoder_decoder_use_tensor_core or self.backbone_matmul_level != BackboneMatmulLevel.FP32:
                raise ValueError("PYTORCH_AUTOCAST must not enable approximate matmul.")
            if not self.autocast_backbone:
                raise ValueError("PYTORCH_AUTOCAST requires autocast_backbone=True.")


def build_inference_config(
    *,
    precision: AuroraInferencePrecision,
    kernel_profile: KernelProfile,
    backbone_matmul_level: BackboneMatmulLevel,
    encoder_decoder_use_tensor_core: bool,
    autocast_encoder_decoder: bool = False,
    autocast_backbone: bool | None = None,
) -> AuroraInferenceConfig:
    """Compose config from kernel profile × backbone matmul level × E/D tensor core."""
    prof = _KERNEL_PROFILES[kernel_profile]
    bf16, tf32 = backbone_matmul_flags(backbone_matmul_level)
    return AuroraInferenceConfig(
        precision=precision,
        kernel_profile=kernel_profile,
        backbone_matmul_level=backbone_matmul_level,
        encoder_decoder_use_tensor_core=encoder_decoder_use_tensor_core,
        autocast_backbone=prof.autocast_backbone if autocast_backbone is None else autocast_backbone,
        backbone_compute_dtype=prof.backbone_compute_dtype,
        backbone_matmul_bf16=bf16,
        backbone_matmul_tf32=tf32,
        window_attn_compute_dtype=prof.window_attn_compute_dtype,
        use_triton_layout=prof.use_triton_layout,
        use_triton_adaln=prof.use_triton_adaln,
        use_triton_mlp=prof.use_triton_mlp,
        use_cute_window_attn=prof.use_cute_window_attn,
        use_triton_perceiver_ln_fusion=prof.use_triton_perceiver_ln_fusion,
        use_perceiver_flash_attn=prof.use_perceiver_flash_attn,
        autocast_encoder_decoder=autocast_encoder_decoder,
        cuda_graph_scope=prof.cuda_graph_scope,
        cuda_graph_recommended=prof.cuda_graph_recommended,
    )


def _config_from_grid_cell(
    preset: AuroraInferencePrecision,
    cell: _PresetGridCell,
) -> AuroraInferenceConfig:
    return build_inference_config(
        precision=preset,
        kernel_profile=cell.kernel_profile,
        backbone_matmul_level=cell.backbone_matmul_level,
        encoder_decoder_use_tensor_core=cell.encoder_decoder_use_tensor_core,
        autocast_backbone=cell.autocast_backbone,
    )


_PRESETS: dict[AuroraInferencePrecision, AuroraInferenceConfig] = {
    p: _config_from_grid_cell(p, cell) for p, cell in _PRESET_GRID.items()
}


def _split_precision_modifiers(raw: str) -> tuple[str, bool | None]:
    """Parse ``fast_fp32+tensor_core`` / ``bf16_mixed+no_tensor_core`` style strings."""
    normalized = raw.strip().lower().replace("-", "_")
    use_tensor_core: bool | None = None
    for suffix, value in (
        ("+tensor_core", True),
        ("+tc", True),
        ("+no_tensor_core", False),
        ("+no_tc", False),
    ):
        if suffix in normalized:
            if use_tensor_core is not None:
                raise ValueError(f"Conflicting tensor-core modifiers in {raw!r}.")
            use_tensor_core = value
            normalized = normalized.replace(suffix, "")
    return normalized.strip("_"), use_tensor_core


def parse_inference_precision(value: str | AuroraInferencePrecision) -> AuroraInferencePrecision:
    if isinstance(value, AuroraInferencePrecision):
        return value
    base, _ = _split_precision_modifiers(value)
    if base == "fast_fp32_triton":
        base = "fast_fp32"
    if base in {"tf32", "1x_tf32"}:
        base = "tf32_1x"
    if base in {"full_bf16", "fullbf16", "bf16_full"}:
        warnings.warn(
            "full_bf16 is deprecated and removed; use bf16_mixed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        base = "bf16_mixed"
    try:
        return AuroraInferencePrecision(base)
    except ValueError as exc:
        valid = ", ".join(p.value for p in AuroraInferencePrecision)
        raise ValueError(f"Unknown inference precision {value!r}; expected one of: {valid}") from exc


def resolve_inference_config(
    precision: str | AuroraInferencePrecision | None,
    *,
    enable_cuda_graph: bool = False,
    encoder_decoder_use_tensor_core: bool | None = None,
    backbone_matmul_level: BackboneMatmulLevel | str | None = None,
) -> AuroraInferenceConfig | None:
    """Return preset config with optional E/D tensor-core and backbone-level overrides."""
    if precision is None:
        return None

    if isinstance(precision, AuroraInferencePrecision):
        preset = precision
        tc_override = encoder_decoder_use_tensor_core
    else:
        base, tc_suffix = _split_precision_modifiers(str(precision))
        preset = parse_inference_precision(base)
        tc_override = tc_suffix if tc_suffix is not None else encoder_decoder_use_tensor_core

    cell = _PRESET_GRID[preset]
    level = cell.backbone_matmul_level
    if backbone_matmul_level is not None:
        level = (
            backbone_matmul_level
            if isinstance(backbone_matmul_level, BackboneMatmulLevel)
            else BackboneMatmulLevel(str(backbone_matmul_level).strip().lower())
        )
    use_tc = cell.encoder_decoder_use_tensor_core if tc_override is None else tc_override

    cfg = build_inference_config(
        precision=preset,
        kernel_profile=cell.kernel_profile,
        backbone_matmul_level=level,
        encoder_decoder_use_tensor_core=use_tc,
        autocast_backbone=cell.autocast_backbone,
    )
    if enable_cuda_graph and cfg.cuda_graph_scope == "off":
        raise ValueError(
            f"CUDA graph capture is not supported for precision={preset.value!r}. "
            "Use tf32_1x or bf16_mixed (or a profile with cuda_graph_scope='backbone')."
        )
    cfg.validate()
    return cfg


def apply_inference_config(
    precision: str | AuroraInferencePrecision,
    *,
    enable_cuda_graph: bool = False,
    encoder_decoder_use_tensor_core: bool | None = None,
    backbone_matmul_level: BackboneMatmulLevel | str | None = None,
) -> dict[str, bool | str]:
    """Expand a precision preset into Aurora constructor kwargs."""
    cfg = resolve_inference_config(
        precision,
        enable_cuda_graph=enable_cuda_graph,
        encoder_decoder_use_tensor_core=encoder_decoder_use_tensor_core,
        backbone_matmul_level=backbone_matmul_level,
    )
    assert cfg is not None
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
        "encoder_decoder_use_tensor_core": cfg.encoder_decoder_use_tensor_core,
        "backbone_matmul_level": cfg.backbone_matmul_level.value,
    }
