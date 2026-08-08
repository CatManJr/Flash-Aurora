"""Copyright (c) Catman Jr. Licensed under the MIT license.

Compatibility shim. Prefer ``flash_aurora.models.inference_precision``.
"""

from __future__ import annotations

from flash_aurora.models.inference_precision import (  # noqa: F401
    AuroraInferenceConfig,
    AuroraInferencePrecision,
    BackboneComputeDtype,
    BackboneMatmulLevel,
    EncoderDecoderMatmulLevel,
    KernelProfile,
    ParsedPrecisionSpec,
    apply_inference_config,
    backbone_matmul_flags,
    describe_backbone_matmul_level,
    describe_encoder_decoder_matmul_level,
    describe_inference_config,
    encoder_decoder_matmul_flags,
    inference_config_label,
    parse_precision_spec,
    resolve_inference_config,
)

__all__ = [
    "AuroraInferenceConfig",
    "AuroraInferencePrecision",
    "BackboneComputeDtype",
    "BackboneMatmulLevel",
    "EncoderDecoderMatmulLevel",
    "KernelProfile",
    "ParsedPrecisionSpec",
    "apply_inference_config",
    "backbone_matmul_flags",
    "describe_backbone_matmul_level",
    "describe_encoder_decoder_matmul_level",
    "describe_inference_config",
    "encoder_decoder_matmul_flags",
    "inference_config_label",
    "parse_precision_spec",
    "resolve_inference_config",
]
