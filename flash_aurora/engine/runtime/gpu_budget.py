from __future__ import annotations

import math

from flash_aurora.engine.core.config import ModelVariantSpec

_GIB = 1024**3
_HISTORY_STEPS = 2
_LATENT_LEVELS = 4

# Default notebook / engine inference path used for calibration.
_DEFAULT_INFERENCE_PRECISION = "bf16_mixed@fp32"

# bf16 weights on GPU (~1.3B Aurora backbone).
_WEIGHT_GIB_BY_CLASS: dict[str, float] = {
    "AuroraSmallPretrained": 0.35,
}

# Patch size per model class (must match ``aurora.model.aurora`` constructors).
_PATCH_SIZE_BY_CLASS: dict[str, int] = {
    "Aurora": 4,
    "AuroraPretrained": 4,
    "Aurora12hPretrained": 4,
    "AuroraSmallPretrained": 4,
    "AuroraHighRes": 10,
    "AuroraAirPollution": 3,
    "AuroraWave": 4,
}

# Transformer activation scale vs the 0.25° reference grid (4 x 180 x 360 tokens).
_DEPTH_SCALE_BY_CLASS: dict[str, float] = {
    "AuroraSmallPretrained": 0.38,
}
_ACTIVATION_GIB_AT_REFERENCE = 7.0
_REFERENCE_PATCH_TOKENS = _LATENT_LEVELS * 180 * 360

# Full-resolution encoder/decoder paths dominate hres VRAM beyond patch-grid activations.
_HRES_FULLRES_OVERHEAD_GIB = 32.5
_WAVE_ACTIVATION_BONUS_GIB = 1.0

# Fallback when a variant has no reserved calibration entry.
_ALLOCATOR_HEADROOM_GIB = 2.0
_FALLBACK_ALLOCATED_TO_RESERVED = 2.5

# Second rollout step reserved increment (bf16_mixed@fp32, preds on CPU after each step).
_DEFAULT_EXTRA_ROLLOUT_STEP_GIB = 4.0
_EXTRA_ROLLOUT_STEP_GIB_BY_CLASS: dict[str, float] = {
    "AuroraHighRes": 0.0,
    "AuroraSmallPretrained": 1.0,
}

# Variants above this peak reservation require exclusive GPU placement.
_EXCLUSIVE_THRESHOLD_GIB = 28.0

# Peak ``torch.cuda.max_memory_reserved`` (GiB), bf16_mixed@fp32, batch=1, one forward.
# Measured on RTX PRO 6000; Guard budgets use *reserved* so driver-level OOM is avoided.
_CALIBRATED_RESERVED_1STEP_GIB: dict[str, float] = {
    "aurora-0.25-pretrained": 36.5,
    "aurora-0.25-finetuned": 36.5,
    "aurora-0.25-12h-pretrained": 36.5,
    "aurora-0.25-small-pretrained": 4.5,
    "aurora-0.1-finetuned": 82.5,
    # Same backbone as 0.25° finetuned; wave surf channels add a small reserved bump.
    "aurora-0.25-wave": 38.0,
    # Lower resolution than 0.25°; conservative estimate until profiled.
    "aurora-0.4-air-pollution": 22.0,
}

# Scale reserved peak relative to bf16_mixed@fp32 (same model, one forward, batch=1).
_PRECISION_RESERVED_SCALE: dict[str, float] = {
    "bf16_mixed@fp32": 1.0,
    "bf16_mixed": 1.0,
    "bf16_mixed@tf32": 1.0,
    "bf16": 0.95,
    "tf32": 1.23,
    "fast_fp32": 1.23,
    "fp32": 1.22,
    "pytorch_autocast": 1.05,
}

_WAVE_ANGLE_VARS = frozenset({"mwd", "mdww", "mdts", "mwd1", "mwd2"})
_WAVE_DENSITY_VARS = frozenset(
    {
        "swh",
        "mwd",
        "mwp",
        "pp1d",
        "shww",
        "mdww",
        "mpww",
        "shts",
        "mdts",
        "mpts",
        "swh1",
        "mwd1",
        "mwp1",
        "swh2",
        "mwd2",
        "mwp2",
        "wind",
        "10u_wave",
        "10v_wave",
    }
)


def _round_budget_gib(value: float) -> float:
    """Round up to the nearest 0.5 GiB for conservative guard leases."""
    return math.ceil(value * 2.0) / 2.0


def _normalize_precision(inference_precision: str | None) -> str:
    if inference_precision is None:
        return _DEFAULT_INFERENCE_PRECISION
    return inference_precision.strip().lower()


def _precision_reserved_scale(inference_precision: str | None) -> float:
    key = _normalize_precision(inference_precision)
    if key in _PRECISION_RESERVED_SCALE:
        return _PRECISION_RESERVED_SCALE[key]
    if key.startswith("bf16"):
        return 1.0
    if "tf32" in key or key.startswith("fp32"):
        return 1.23
    return 1.0


def _patch_size(model_class: str) -> int:
    return _PATCH_SIZE_BY_CLASS.get(model_class, 4)


def _crop_to_patch(n: int, patch_size: int) -> int:
    return n - n % patch_size


def _wave_surf_channels(surf_vars: tuple[str, ...]) -> int:
    """Match ``AuroraWave`` supplemented surface channel expansion."""
    channels = 0
    for name in surf_vars:
        channels += 2 if name in _WAVE_ANGLE_VARS else 1
        if name in _WAVE_DENSITY_VARS:
            channels += 1
    return channels


def _surf_channels(variant: ModelVariantSpec) -> int:
    if variant.model_class == "AuroraWave":
        return _wave_surf_channels(variant.surf_vars)
    return len(variant.surf_vars)


def _ic_elements(variant: ModelVariantSpec) -> int:
    height, width = variant.resolution
    surf = _surf_channels(variant)
    atmos = len(variant.atmos_vars) * len(variant.levels)
    static = len(variant.static_vars)
    per_pixel = surf * _HISTORY_STEPS + atmos * _HISTORY_STEPS + static
    return per_pixel * height * width


def _fp32_gib(num_elements: int) -> float:
    return num_elements * 4 / _GIB


def _patch_tokens(variant: ModelVariantSpec) -> int:
    patch_size = _patch_size(variant.model_class)
    height, width = variant.resolution
    height = _crop_to_patch(height, patch_size)
    width = _crop_to_patch(width, patch_size)
    return _LATENT_LEVELS * (height // patch_size) * (width // patch_size)


def _weight_gib(variant: ModelVariantSpec) -> float:
    return _WEIGHT_GIB_BY_CLASS.get(variant.model_class, 2.6)


def _activation_gib(variant: ModelVariantSpec) -> float:
    depth_scale = _DEPTH_SCALE_BY_CLASS.get(variant.model_class, 1.0)
    tokens = _patch_tokens(variant)
    activation = _ACTIVATION_GIB_AT_REFERENCE * (tokens / _REFERENCE_PATCH_TOKENS) * depth_scale
    if variant.model_class == "AuroraWave":
        activation += _WAVE_ACTIVATION_BONUS_GIB
    return activation


def _fullres_overhead_gib(variant: ModelVariantSpec) -> float:
    if variant.model_class == "AuroraHighRes":
        return _HRES_FULLRES_OVERHEAD_GIB
    return 0.0


def _extra_rollout_step_gib(variant: ModelVariantSpec) -> float:
    return _EXTRA_ROLLOUT_STEP_GIB_BY_CLASS.get(
        variant.model_class,
        _DEFAULT_EXTRA_ROLLOUT_STEP_GIB,
    )


def estimate_vram_allocated_gib(variant: ModelVariantSpec, *, rollout_steps: int = 1) -> float:
    """Analytic tensor/activation estimate (``max_memory_allocated`` lower bound).

    Prefer :func:`estimate_vram_gib` for GpuGuard; PyTorch ``reserved`` is higher.
    """
    steps = max(1, rollout_steps)
    base = (
        _fp32_gib(_ic_elements(variant))
        + _weight_gib(variant)
        + _activation_gib(variant)
        + _fullres_overhead_gib(variant)
        + _ALLOCATOR_HEADROOM_GIB
    )
    extra = _extra_rollout_step_gib(variant) * (steps - 1)
    return _round_budget_gib(base + extra)


def _calibrated_reserved_1step_gib(variant: ModelVariantSpec) -> float | None:
    return _CALIBRATED_RESERVED_1STEP_GIB.get(variant.name)


def _fallback_reserved_1step_gib(variant: ModelVariantSpec) -> float:
    return estimate_vram_allocated_gib(variant, rollout_steps=1) * _FALLBACK_ALLOCATED_TO_RESERVED


def estimate_vram_gib(
    variant: ModelVariantSpec,
    *,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
) -> float:
    """Estimate peak CUDA memory for GpuGuard (``max_memory_reserved``-style budget).

    Uses benchmark-calibrated reserved peaks at ``bf16_mixed@fp32`` by default, scaled
    for other inference precision tiers (``tf32`` / ``fp32`` reserve more).
    """
    steps = max(1, rollout_steps)
    base = _calibrated_reserved_1step_gib(variant)
    if base is None:
        base = _fallback_reserved_1step_gib(variant)

    extra = _extra_rollout_step_gib(variant) * (steps - 1)
    scaled = (base + extra) * _precision_reserved_scale(inference_precision)
    return _round_budget_gib(scaled)


def is_exclusive_variant(
    variant: ModelVariantSpec,
    *,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
) -> bool:
    """Return whether the workload should not share the GPU with other jobs."""
    return (
        estimate_vram_gib(
            variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )
        >= _EXCLUSIVE_THRESHOLD_GIB
    )


def is_shareable_variant(
    variant: ModelVariantSpec,
    *,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
) -> bool:
    return not is_exclusive_variant(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
