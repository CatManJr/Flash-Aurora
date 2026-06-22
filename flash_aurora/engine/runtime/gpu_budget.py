from __future__ import annotations

from flash_aurora.engine.core.config import ModelVariantSpec

# Heuristic peak VRAM (GiB) for load + rollout on a single CUDA device.
_BASE_VARIANT_GIB: dict[str, float] = {
    "aurora-0.1-finetuned": 48.0,
    "aurora-0.25-finetuned": 14.0,
    "aurora-0.25-pretrained": 14.0,
    "aurora-0.25-12h-pretrained": 14.0,
    "aurora-0.25-small-pretrained": 6.0,
    "aurora-0.4-air-pollution": 10.0,
    "aurora-0.25-wave": 16.0,
}

_EXTRA_ROLLOUT_STEP_GIB: dict[str, float] = {
    "aurora-0.1-finetuned": 0.0,
    "aurora-0.25-small-pretrained": 1.0,
}
_DEFAULT_EXTRA_STEP_GIB = 3.0

# Variants above this peak reservation require exclusive GPU placement.
_EXCLUSIVE_THRESHOLD_GIB = 28.0


def estimate_vram_gib(variant: ModelVariantSpec, *, rollout_steps: int = 1) -> float:
    """Estimate CUDA memory required for ``variant`` and ``rollout_steps``."""
    steps = max(1, rollout_steps)
    base = _BASE_VARIANT_GIB.get(variant.name)
    if base is None:
        height, width = variant.resolution
        megapixels = (height * width) / 1_000_000
        base = 4.0 + megapixels * 2.2
    extra = _EXTRA_ROLLOUT_STEP_GIB.get(variant.name, _DEFAULT_EXTRA_STEP_GIB)
    return base + extra * (steps - 1)


def is_exclusive_variant(variant: ModelVariantSpec, *, rollout_steps: int = 1) -> bool:
    """Return whether the workload should not share the GPU with other jobs."""
    return estimate_vram_gib(variant, rollout_steps=rollout_steps) >= _EXCLUSIVE_THRESHOLD_GIB


def is_shareable_variant(variant: ModelVariantSpec, *, rollout_steps: int = 1) -> bool:
    return not is_exclusive_variant(variant, rollout_steps=rollout_steps)
