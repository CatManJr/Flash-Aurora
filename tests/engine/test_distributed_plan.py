from __future__ import annotations

import pytest

from flash_aurora.engine.core.presets import DEFAULT_PRESETS, VARIANTS
from flash_aurora.engine.distributed import DistributedConfig, plan_parallelism, requires_parallelism


def test_requires_parallelism_for_025_on_32gib_budget() -> None:
    variant = VARIANTS["aurora-0.25-pretrained"]
    assert requires_parallelism(
        variant,
        max_vram_gib_per_device=32.0,
        rollout_steps=1,
        inference_precision="bf16_mixed@fp32",
    )


def test_small_pretrained_fits_single_32gib_device() -> None:
    variant = VARIANTS["aurora-0.25-small-pretrained"]
    assert not requires_parallelism(
        variant,
        max_vram_gib_per_device=32.0,
        rollout_steps=1,
    )


def test_two_device_plan_balances_vram_across_gpus() -> None:
    variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    plan = plan_parallelism(
        variant,
        DistributedConfig(
            devices=("cuda:0", "cuda:1"),
            max_vram_gib_per_device=32.0,
            rollout_steps=1,
            force=True,
        ),
        inference_precision="bf16_mixed@fp32",
    )
    assert plan.encoder_device == "cuda:0"
    assert plan.backbone_device == "cuda:1"
    assert plan.decoder_device == "cuda:1"
    assert plan.decoder_spatial_parallel is True
    assert plan.decoder_spatial_devices == ("cuda:0", "cuda:1")
    assert max(plan.estimated_per_device_gib) <= plan.estimated_peak_gib
    assert abs(plan.estimated_per_device_gib[0] - plan.estimated_per_device_gib[1]) < (
        plan.estimated_peak_gib * 0.5
    )


def test_three_device_plan_assigns_one_stage_per_gpu() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    plan = plan_parallelism(
        variant,
        DistributedConfig(
            devices=("cuda:0", "cuda:1", "cuda:2"),
            max_vram_gib_per_device=40.0,
            rollout_steps=1,
            force=True,
        ),
    )
    assert plan.encoder_device == "cuda:0"
    assert plan.backbone_device == "cuda:1"
    assert plan.decoder_device == "cuda:2"


def test_single_device_plan_rejects_tight_budget_without_force() -> None:
    variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    with pytest.raises(ValueError, match="needs ~"):
        plan_parallelism(
            variant,
            DistributedConfig(
                devices=("cuda:0",),
                max_vram_gib_per_device=32.0,
                rollout_steps=1,
            ),
        )


def test_two_device_plan_rejects_when_budget_too_small_without_force() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    with pytest.raises(ValueError, match="exceeds per-device budget"):
        plan_parallelism(
            variant,
            DistributedConfig(
                devices=("cuda:0", "cuda:1"),
                max_vram_gib_per_device=24.0,
                rollout_steps=1,
            ),
        )
