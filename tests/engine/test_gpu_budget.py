from __future__ import annotations

import math

import pytest

from flash_aurora.engine.core.presets import DEFAULT_PRESETS, VARIANTS
from flash_aurora.engine.runtime.gpu_budget import (
    _fp32_gib,
    _ic_elements,
    _patch_tokens,
    _precision_reserved_scale,
    _surf_channels,
    _wave_surf_channels,
    estimate_vram_allocated_gib,
    estimate_vram_gib,
    is_exclusive_variant,
)


def test_hres_ic_elements_matches_full_resolution_grid() -> None:
    variant = VARIANTS["aurora-0.1-finetuned"]
    elements = _ic_elements(variant)
    # 4 surf * 2 + 5 atmos * 13 * 2 + 3 static = 141 per pixel at 1801x3600.
    assert elements == 141 * 1801 * 3600
    assert _fp32_gib(elements) == pytest.approx(3.41, abs=0.05)


def test_hres_patch_tokens_match_025_degree_encoder_grid() -> None:
    hres = VARIANTS["aurora-0.1-finetuned"]
    era5 = VARIANTS["aurora-0.25-finetuned"]
    assert _patch_tokens(hres) == _patch_tokens(era5) == 4 * 180 * 360


def test_wave_surf_channel_expansion() -> None:
    variant = VARIANTS["aurora-0.25-wave"]
    assert _surf_channels(variant) == _wave_surf_channels(variant.surf_vars)
    assert _surf_channels(variant) == 44


def test_reserved_budget_exceeds_allocated_formula_for_pretrained() -> None:
    variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    reserved = estimate_vram_gib(variant)
    allocated = estimate_vram_allocated_gib(variant)
    assert reserved >= 36.0
    assert reserved > allocated


def test_tf32_precision_scales_reserved_budget() -> None:
    variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    bf16 = estimate_vram_gib(variant, inference_precision="bf16_mixed@fp32")
    tf32 = estimate_vram_gib(variant, inference_precision="tf32")
    assert tf32 > bf16
    assert _precision_reserved_scale("tf32") == pytest.approx(1.23)


@pytest.mark.parametrize(
    ("preset", "steps", "minimum", "maximum"),
    [
        ("hres_0.1", 2, 82.5, 84.0),
        ("era5_pretrained", 1, 36.0, 37.0),
        ("era5_pretrained", 2, 40.0, 41.0),
        ("small_pretrained", 1, 4.5, 5.0),
        ("small_pretrained", 2, 5.5, 6.5),
        ("cams", 1, 21.5, 22.5),
        ("wave", 1, 37.5, 38.5),
        ("hres_t0_finetuned", 1, 36.0, 37.0),
    ],
)
def test_estimate_vram_gib_preset_ranges(
    preset: str,
    steps: int,
    minimum: float,
    maximum: float,
) -> None:
    variant = DEFAULT_PRESETS.get(preset).variant
    estimate = estimate_vram_gib(variant, rollout_steps=steps)
    assert minimum <= estimate <= maximum


def test_hres_is_exclusive_small_is_shareable() -> None:
    hres = DEFAULT_PRESETS.get("hres_0.1").variant
    small = DEFAULT_PRESETS.get("small_pretrained").variant
    assert is_exclusive_variant(hres, rollout_steps=2)
    assert estimate_vram_gib(hres, rollout_steps=2) >= 82.5
    assert not is_exclusive_variant(small)
    assert estimate_vram_gib(small) < 10.0


def test_estimates_round_up_to_half_gib() -> None:
    for variant in VARIANTS.values():
        for steps in (1, 2):
            estimate = estimate_vram_gib(variant, rollout_steps=steps)
            assert math.isclose(estimate * 2, round(estimate * 2), rel_tol=0, abs_tol=1e-9)
