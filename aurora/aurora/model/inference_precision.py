"""Inference precision router for Aurora inference presets.

Two independent axes (combinable):

* **Backbone** — :class:`BackboneMatmulLevel`: ``fp32`` | ``tf32`` | ``bf16_mixed`` | ``bf16``
* **Encoder/decoder** — :class:`EncoderDecoderMatmulLevel`: ``fp32`` | ``tf32`` only
  (no E/D BF16 autocast: Perceiver needs FP32 ``lat``/``lon`` and AdaLN paths).

Named presets (``fp32``, ``fast_fp32``, ``tf32``, ``bf16_mixed``, ``bf16``, …) set both axes.
Override either axis explicitly or use a combo string (``backbone@encoder_decoder``).
The left token is always a **backbone matmul level**; the right is **encoder/decoder only**:

* ``bf16_mixed@fp32`` — hybrid backbone (BF16 attention QKV/proj + MLP) + strict FP32 Perceiver
* ``bf16@fp32`` — full backbone BF16 linears + strict FP32 Perceiver
* ``bf16_mixed@tf32`` — hybrid backbone + Perceiver TF32 tensor cores (same E/D as preset ``bf16_mixed``)
* ``bf16@tf32`` — full backbone BF16 + Perceiver TF32 (same E/D as preset ``bf16``)
* ``backbone=bf16_mixed,encoder_decoder=fp32`` — equivalent to ``bf16_mixed@fp32``

Do not confuse backbone ``bf16`` / ``bf16_mixed`` with a non-existent encoder/decoder ``bf16`` level.

Custom Triton/CuTe Swin paths never run inside ``torch.autocast``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

BackboneComputeDtype = Literal["float32", "bfloat16"]
KernelProfile = Literal["baseline", "fast_fp32", "tf32_backbone", "bf16_mixed_backbone"]


class BackboneMatmulLevel(str, Enum):
    """Backbone ``F.linear`` matmul precision (activations stay FP32 unless autocast)."""

    FP32 = "fp32"
    TF32 = "tf32"
    BF16_MIXED = "bf16_mixed"
    BF16 = "bf16"


class EncoderDecoderMatmulLevel(str, Enum):
    """Perceiver encoder/decoder matmul precision (``lat``/``lon`` and AdaLN stay FP32)."""

    FP32 = "fp32"
    TF32 = "tf32"


class AuroraInferencePrecision(str, Enum):
    """Named presets bundling kernel profile + default backbone × E/D matmul levels."""

    FP32 = "fp32"
    PYTORCH_AUTOCAST = "pytorch_autocast"
    FAST_FP32 = "fast_fp32"
    TF32 = "tf32"
    BF16_MIXED = "bf16_mixed"
    BF16 = "bf16"


CudaGraphScope = Literal["off", "backbone", "full_gpu"]

_BACKBONE_LEVEL_TO_KERNEL: dict[BackboneMatmulLevel, KernelProfile] = {
    BackboneMatmulLevel.FP32: "fast_fp32",
    BackboneMatmulLevel.TF32: "tf32_backbone",
    BackboneMatmulLevel.BF16_MIXED: "bf16_mixed_backbone",
    BackboneMatmulLevel.BF16: "bf16_mixed_backbone",
}


def backbone_matmul_flags(level: BackboneMatmulLevel) -> tuple[bool, bool]:
    """Return ``(backbone_matmul_bf16, backbone_matmul_tf32)``."""
    if level == BackboneMatmulLevel.FP32:
        return False, False
    if level == BackboneMatmulLevel.TF32:
        return False, True
    if level == BackboneMatmulLevel.BF16_MIXED:
        return True, True
    return True, False


def encoder_decoder_matmul_flags(level: EncoderDecoderMatmulLevel) -> tuple[bool, bool]:
    """Return ``(use_tensor_core, autocast_encoder_decoder)``."""
    if level == EncoderDecoderMatmulLevel.FP32:
        return False, False
    return True, False


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
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel
    autocast_backbone: bool | None = None


_PRESET_GRID: dict[AuroraInferencePrecision, _PresetGridCell] = {
    AuroraInferencePrecision.FP32: _PresetGridCell(
        "baseline",
        BackboneMatmulLevel.FP32,
        EncoderDecoderMatmulLevel.FP32,
    ),
    AuroraInferencePrecision.PYTORCH_AUTOCAST: _PresetGridCell(
        "baseline",
        BackboneMatmulLevel.FP32,
        EncoderDecoderMatmulLevel.FP32,
        autocast_backbone=True,
    ),
    AuroraInferencePrecision.FAST_FP32: _PresetGridCell(
        "fast_fp32",
        BackboneMatmulLevel.FP32,
        EncoderDecoderMatmulLevel.FP32,
    ),
    AuroraInferencePrecision.TF32: _PresetGridCell(
        "tf32_backbone",
        BackboneMatmulLevel.TF32,
        EncoderDecoderMatmulLevel.TF32,
    ),
    AuroraInferencePrecision.BF16_MIXED: _PresetGridCell(
        "bf16_mixed_backbone",
        BackboneMatmulLevel.BF16_MIXED,
        EncoderDecoderMatmulLevel.TF32,
    ),
    AuroraInferencePrecision.BF16: _PresetGridCell(
        "bf16_mixed_backbone",
        BackboneMatmulLevel.BF16,
        EncoderDecoderMatmulLevel.TF32,
    ),
}


@dataclass(frozen=True)
class ParsedPrecisionSpec:
    """Result of :func:`parse_precision_spec` (optional per-axis overrides)."""

    named_preset: AuroraInferencePrecision | None = None
    backbone_matmul_level: BackboneMatmulLevel | None = None
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel | None = None


@dataclass(frozen=True)
class AuroraInferenceConfig:
    """Resolved inference settings."""

    precision: AuroraInferencePrecision | None
    config_label: str
    kernel_profile: KernelProfile
    backbone_matmul_level: BackboneMatmulLevel
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel
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
        ed_tc, ed_ac = encoder_decoder_matmul_flags(self.encoder_decoder_matmul_level)
        if self.encoder_decoder_use_tensor_core != ed_tc:
            raise ValueError("encoder_decoder_use_tensor_core disagrees with encoder_decoder_matmul_level.")
        if self.autocast_encoder_decoder != ed_ac:
            raise ValueError("autocast_encoder_decoder disagrees with encoder_decoder_matmul_level.")
        if self.autocast_encoder_decoder and self.encoder_decoder_use_tensor_core:
            raise ValueError(
                "encoder_decoder TF32 tensor core cannot be combined with E/D BF16 autocast."
            )
        if self.encoder_decoder_use_tensor_core and self.backbone_compute_dtype != "float32":
            raise ValueError(
                "encoder_decoder TF32 requires backbone_compute_dtype='float32'."
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
        if self.backbone_matmul_level in (
            BackboneMatmulLevel.BF16_MIXED,
            BackboneMatmulLevel.BF16,
        ):
            if self.kernel_profile != "bf16_mixed_backbone":
                raise ValueError("bf16/bf16_mixed backbone requires kernel_profile=bf16_mixed_backbone.")
            if self.window_attn_compute_dtype != "bfloat16":
                raise ValueError("bf16 backbone requires CuTe BF16 window attention.")
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
            if self.encoder_decoder_matmul_level != EncoderDecoderMatmulLevel.FP32:
                raise ValueError("PYTORCH_AUTOCAST requires encoder_decoder_matmul_level=fp32.")
            if self.backbone_matmul_level != BackboneMatmulLevel.FP32:
                raise ValueError("PYTORCH_AUTOCAST requires backbone_matmul_level=fp32.")
            if not self.autocast_backbone:
                raise ValueError("PYTORCH_AUTOCAST requires autocast_backbone=True.")


def _parse_matmul_level(
    token: str,
    enum_cls: type[BackboneMatmulLevel] | type[EncoderDecoderMatmulLevel],
) -> BackboneMatmulLevel | EncoderDecoderMatmulLevel:
    try:
        return enum_cls(token.strip().lower())  # type: ignore[return-value]
    except ValueError as exc:
        valid = ", ".join(m.value for m in enum_cls)  # type: ignore[attr-defined]
        raise ValueError(f"Unknown matmul level {token!r}; expected one of: {valid}") from exc


def _parse_backbone_token(token: str) -> BackboneMatmulLevel:
    t = token.strip().lower()
    try:
        return BackboneMatmulLevel(t)
    except ValueError:
        pass
    try:
        preset = AuroraInferencePrecision(t)
    except ValueError as exc:
        valid_bb = ", ".join(m.value for m in BackboneMatmulLevel)
        valid_p = ", ".join(p.value for p in AuroraInferencePrecision)
        raise ValueError(
            f"Unknown backbone token {token!r}; expected backbone level ({valid_bb}) "
            f"or named preset ({valid_p})."
        ) from exc
    return _PRESET_GRID[preset].backbone_matmul_level


def _parse_encoder_decoder_token(token: str) -> EncoderDecoderMatmulLevel:
    t = token.strip().lower().replace("encoder_decoder", "ed").replace("enc_dec", "ed")
    if t.startswith("ed="):
        t = t[3:]
    if t == "bf16":
        raise ValueError(
            "encoder_decoder=bf16 is not supported: Perceiver requires FP32 lat/lon and "
            "FP32 AdaLN; use backbone bf16 or bf16_mixed for Swin BF16 matmuls."
        )
    try:
        return EncoderDecoderMatmulLevel(t)
    except ValueError:
        pass
    try:
        preset = AuroraInferencePrecision(t)
    except ValueError as exc:
        valid_ed = ", ".join(m.value for m in EncoderDecoderMatmulLevel)
        valid_p = ", ".join(p.value for p in AuroraInferencePrecision)
        raise ValueError(
            f"Unknown encoder/decoder token {token!r}; expected E/D level ({valid_ed}) "
            f"or named preset ({valid_p})."
        ) from exc
    return _PRESET_GRID[preset].encoder_decoder_matmul_level


def parse_precision_spec(value: str) -> ParsedPrecisionSpec:
    """Parse a preset name or ``backbone@encoder_decoder`` combo string."""
    raw = value.strip().lower().replace("-", "_")
    if "@" in raw:
        bb_tok, ed_tok = raw.split("@", 1)
        return ParsedPrecisionSpec(
            backbone_matmul_level=_parse_backbone_token(bb_tok),
            encoder_decoder_matmul_level=_parse_encoder_decoder_token(ed_tok),
        )
    if "," in raw and ("backbone=" in raw or "encoder_decoder=" in raw or "ed=" in raw):
        bb_tok: str | None = None
        ed_tok: str | None = None
        for part in raw.split(","):
            part = part.strip()
            if part.startswith("backbone="):
                bb_tok = part.split("=", 1)[1]
            elif part.startswith(("encoder_decoder=", "ed=", "enc_dec=")):
                ed_tok = part.split("=", 1)[1]
        if bb_tok is None or ed_tok is None:
            raise ValueError(
                f"Combo string {value!r} must include backbone= and encoder_decoder= (or ed=)."
            )
        return ParsedPrecisionSpec(
            backbone_matmul_level=_parse_backbone_token(bb_tok),
            encoder_decoder_matmul_level=_parse_encoder_decoder_token(ed_tok),
        )
    try:
        return ParsedPrecisionSpec(named_preset=AuroraInferencePrecision(raw))
    except ValueError as exc:
        valid_p = ", ".join(p.value for p in AuroraInferencePrecision)
        raise ValueError(
            f"Unknown precision {value!r}; use a named preset ({valid_p}) or "
            "backbone@encoder_decoder (e.g. bf16@fp32)."
        ) from exc


def inference_config_label(
    *,
    backbone_matmul_level: BackboneMatmulLevel,
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel,
    named_preset: AuroraInferencePrecision | None = None,
) -> str:
    if named_preset is not None:
        cell = _PRESET_GRID[named_preset]
        if (
            cell.backbone_matmul_level == backbone_matmul_level
            and cell.encoder_decoder_matmul_level == encoder_decoder_matmul_level
        ):
            return named_preset.value
    return f"{backbone_matmul_level.value}@{encoder_decoder_matmul_level.value}"


def describe_backbone_matmul_level(level: BackboneMatmulLevel) -> str:
    if level == BackboneMatmulLevel.FP32:
        return "backbone matmul FP32 (strict, no TF32/BF16 hooks)"
    if level == BackboneMatmulLevel.TF32:
        return "backbone matmul TF32 tensor cores"
    if level == BackboneMatmulLevel.BF16_MIXED:
        return "backbone matmul hybrid: BF16 attention QKV/proj + BF16 MLP; TF32 elsewhere"
    return (
        "backbone matmul full BF16 (all Swin linears + fused CuTe attn; "
        "quant-prep / GeForce-oriented)"
    )


def describe_encoder_decoder_matmul_level(level: EncoderDecoderMatmulLevel) -> str:
    if level == EncoderDecoderMatmulLevel.FP32:
        return "encoder/decoder matmul FP32 (native Perceiver SDPA)"
    return "encoder/decoder matmul TF32 tensor cores (native Perceiver SDPA)"


def describe_inference_config(cfg: AuroraInferenceConfig) -> str:
    """Human-readable precision summary (no preset aliases)."""
    parts = [
        describe_backbone_matmul_level(cfg.backbone_matmul_level),
        describe_encoder_decoder_matmul_level(cfg.encoder_decoder_matmul_level),
    ]
    if cfg.autocast_backbone:
        parts.append("backbone torch.autocast BF16")
    prof = _KERNEL_PROFILES[cfg.kernel_profile]
    if prof.use_triton_layout or prof.use_triton_adaln:
        extras: list[str] = []
        if prof.use_triton_layout:
            extras.append("layout")
        if prof.use_triton_adaln:
            extras.append("AdaLN")
        parts.append("Triton " + "+".join(extras))
    else:
        parts.append("PyTorch Swin (no Triton/CuTe)")
    if prof.use_cute_window_attn:
        parts.append(f"CuTe window attention ({prof.window_attn_compute_dtype})")
    elif cfg.kernel_profile != "baseline":
        parts.append("PyTorch window SDPA")
    parts.append(f"backbone activations {prof.backbone_compute_dtype}")
    return "; ".join(parts)


def kernel_profile_for_backbone(
    level: BackboneMatmulLevel,
    *,
    named_preset: AuroraInferencePrecision | None = None,
) -> KernelProfile:
    if named_preset == AuroraInferencePrecision.PYTORCH_AUTOCAST:
        return "baseline"
    return _BACKBONE_LEVEL_TO_KERNEL[level]


def build_inference_config(
    *,
    precision: AuroraInferencePrecision | None = None,
    kernel_profile: KernelProfile,
    backbone_matmul_level: BackboneMatmulLevel,
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel,
    autocast_encoder_decoder: bool | None = None,
    autocast_backbone: bool | None = None,
) -> AuroraInferenceConfig:
    """Compose config from kernel profile × backbone level × encoder/decoder level."""
    prof = _KERNEL_PROFILES[kernel_profile]
    bb_bf16, bb_tf32 = backbone_matmul_flags(backbone_matmul_level)
    ed_tc, ed_ac = encoder_decoder_matmul_flags(encoder_decoder_matmul_level)
    if autocast_encoder_decoder is not None:
        ed_ac = autocast_encoder_decoder
    label = inference_config_label(
        backbone_matmul_level=backbone_matmul_level,
        encoder_decoder_matmul_level=encoder_decoder_matmul_level,
        named_preset=precision,
    )
    return AuroraInferenceConfig(
        precision=precision,
        config_label=label,
        kernel_profile=kernel_profile,
        backbone_matmul_level=backbone_matmul_level,
        encoder_decoder_matmul_level=encoder_decoder_matmul_level,
        encoder_decoder_use_tensor_core=ed_tc,
        autocast_backbone=prof.autocast_backbone if autocast_backbone is None else autocast_backbone,
        backbone_compute_dtype=prof.backbone_compute_dtype,
        backbone_matmul_bf16=bb_bf16,
        backbone_matmul_tf32=bb_tf32,
        window_attn_compute_dtype=prof.window_attn_compute_dtype,
        use_triton_layout=prof.use_triton_layout,
        use_triton_adaln=prof.use_triton_adaln,
        use_triton_mlp=prof.use_triton_mlp,
        use_cute_window_attn=prof.use_cute_window_attn,
        use_triton_perceiver_ln_fusion=prof.use_triton_perceiver_ln_fusion,
        use_perceiver_flash_attn=prof.use_perceiver_flash_attn,
        autocast_encoder_decoder=ed_ac,
        cuda_graph_scope=prof.cuda_graph_scope,
        cuda_graph_recommended=prof.cuda_graph_recommended,
    )


def _split_precision_modifiers(raw: str) -> tuple[str, bool | None]:
    """Parse trailing ``+tensor_core`` / ``+no_tensor_core`` on a preset or combo string."""
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


def _encoder_decoder_level_from_tensor_core_flag(
    use_tensor_core: bool | None,
) -> EncoderDecoderMatmulLevel | None:
    if use_tensor_core is None:
        return None
    return EncoderDecoderMatmulLevel.TF32 if use_tensor_core else EncoderDecoderMatmulLevel.FP32


def parse_inference_precision(value: str | AuroraInferencePrecision) -> AuroraInferencePrecision:
    if isinstance(value, AuroraInferencePrecision):
        return value
    base, _ = _split_precision_modifiers(value)
    spec = parse_precision_spec(base)
    if spec.named_preset is None:
        raise ValueError(
            f"{value!r} is a precision combo, not a single named preset; "
            "use resolve_inference_config() instead."
        )
    return spec.named_preset


def resolve_inference_config(
    precision: str | AuroraInferencePrecision | None = None,
    *,
    enable_cuda_graph: bool = False,
    backbone_matmul_level: BackboneMatmulLevel | str | None = None,
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel | str | None = None,
    encoder_decoder_use_tensor_core: bool | None = None,
    kernel_profile: KernelProfile | str | None = None,
) -> AuroraInferenceConfig | None:
    """Resolve config from a named preset and/or independent backbone × E/D levels."""
    if precision is None and backbone_matmul_level is None and encoder_decoder_matmul_level is None:
        return None

    parsed = ParsedPrecisionSpec()
    tc_suffix: bool | None = None
    if precision is not None:
        if isinstance(precision, AuroraInferencePrecision):
            parsed = ParsedPrecisionSpec(named_preset=precision)
        else:
            base, tc_suffix = _split_precision_modifiers(str(precision))
            parsed = parse_precision_spec(base)

    named = parsed.named_preset
    cell = _PRESET_GRID[named] if named is not None else None

    bb_level = cell.backbone_matmul_level if cell is not None else parsed.backbone_matmul_level
    if backbone_matmul_level is not None:
        bb_level = (
            backbone_matmul_level
            if isinstance(backbone_matmul_level, BackboneMatmulLevel)
            else BackboneMatmulLevel(str(backbone_matmul_level).strip().lower())
        )
    if bb_level is None:
        raise ValueError("backbone_matmul_level is required (via preset or explicit override).")

    ed_level = cell.encoder_decoder_matmul_level if cell is not None else parsed.encoder_decoder_matmul_level
    ed_from_tc = _encoder_decoder_level_from_tensor_core_flag(
        tc_suffix if tc_suffix is not None else encoder_decoder_use_tensor_core
    )
    if encoder_decoder_matmul_level is not None:
        ed_level = (
            encoder_decoder_matmul_level
            if isinstance(encoder_decoder_matmul_level, EncoderDecoderMatmulLevel)
            else EncoderDecoderMatmulLevel(str(encoder_decoder_matmul_level).strip().lower())
        )
    elif ed_from_tc is not None:
        ed_level = ed_from_tc
    if ed_level is None:
        raise ValueError("encoder_decoder_matmul_level is required (via preset or explicit override).")

    prof: KernelProfile
    if kernel_profile is not None:
        kp = str(kernel_profile).strip()
        if kp not in _KERNEL_PROFILES:
            raise ValueError(f"Unknown kernel_profile {kernel_profile!r}.")
        prof = kp  # type: ignore[assignment]
    elif cell is not None:
        prof = cell.kernel_profile
    else:
        prof = kernel_profile_for_backbone(bb_level, named_preset=named)

    autocast_bb = cell.autocast_backbone if cell is not None else None

    cfg = build_inference_config(
        precision=named,
        kernel_profile=prof,
        backbone_matmul_level=bb_level,
        encoder_decoder_matmul_level=ed_level,
        autocast_backbone=autocast_bb,
    )
    if enable_cuda_graph and cfg.cuda_graph_scope == "off":
        raise ValueError(
            f"CUDA graph capture is not supported for {cfg.config_label!r}. "
            "Use tf32, bf16_mixed, or bf16 (cuda_graph_scope='backbone')."
        )
    cfg.validate()
    return cfg


def expand_precision_combos(
    backbone_levels: list[str] | tuple[str, ...],
    encoder_decoder_levels: list[str] | tuple[str, ...],
) -> list[tuple[str, AuroraInferenceConfig]]:
    """Cartesian product of backbone × encoder/decoder matmul levels for benchmarking."""
    combos: list[tuple[str, AuroraInferenceConfig]] = []
    for bb in backbone_levels:
        for ed in encoder_decoder_levels:
            spec = f"{bb}@{ed}"
            cfg = resolve_inference_config(spec)
            assert cfg is not None
            combos.append((cfg.config_label, cfg))
    return combos


# Default 4×2 custom matmul grid (Triton/CuTe Swin + native Perceiver).
DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS: tuple[str, ...] = ("fp32", "tf32", "bf16_mixed", "bf16")
DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS: tuple[str, ...] = ("fp32", "tf32")


def apply_inference_config(
    precision: str | AuroraInferencePrecision | None = None,
    *,
    enable_cuda_graph: bool = False,
    backbone_matmul_level: BackboneMatmulLevel | str | None = None,
    encoder_decoder_matmul_level: EncoderDecoderMatmulLevel | str | None = None,
    encoder_decoder_use_tensor_core: bool | None = None,
    kernel_profile: KernelProfile | str | None = None,
) -> dict[str, bool | str]:
    """Expand resolved config into Aurora constructor kwargs."""
    cfg = resolve_inference_config(
        precision,
        enable_cuda_graph=enable_cuda_graph,
        backbone_matmul_level=backbone_matmul_level,
        encoder_decoder_matmul_level=encoder_decoder_matmul_level,
        encoder_decoder_use_tensor_core=encoder_decoder_use_tensor_core,
        kernel_profile=kernel_profile,
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
        "encoder_decoder_matmul_level": cfg.encoder_decoder_matmul_level.value,
        "inference_config_label": cfg.config_label,
    }


_PRESETS: dict[AuroraInferencePrecision, AuroraInferenceConfig] = {
    p: resolve_inference_config(p) for p in AuroraInferencePrecision
}
