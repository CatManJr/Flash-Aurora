from __future__ import annotations

import math

from flash_aurora.engine.core.config import ModelVariantSpec
from flash_aurora.engine.distributed.config import DistributedConfig, ParallelPlan
from flash_aurora.engine.runtime.gpu_budget import estimate_vram_gib

# Heuristic stage fractions of peak reserved VRAM (weights + activations).
_STAGE_FRACTIONS: dict[str, tuple[float, float, float]] = {
    "Aurora": (0.22, 0.53, 0.25),
    "AuroraPretrained": (0.22, 0.53, 0.25),
    "Aurora12hPretrained": (0.22, 0.53, 0.25),
    "AuroraSmallPretrained": (0.25, 0.45, 0.30),
    "AuroraHighRes": (0.38, 0.34, 0.28),
    "AuroraAirPollution": (0.24, 0.50, 0.26),
    "AuroraWave": (0.23, 0.52, 0.25),
}

_PIPELINE_HEADROOM_GIB = 1.5


def _round_up_half(value: float) -> float:
    return math.ceil(value * 2.0) / 2.0


def _stage_fractions(variant: ModelVariantSpec) -> tuple[float, float, float]:
    return _STAGE_FRACTIONS.get(variant.model_class, (0.22, 0.53, 0.25))


def estimate_stage_vram_gib(
    variant: ModelVariantSpec,
    *,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
) -> tuple[float, float, float]:
    """Return estimated reserved GiB for encoder, backbone, and decoder."""
    total = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    enc_frac, bb_frac, dec_frac = _stage_fractions(variant)
    return (
        _round_up_half(total * enc_frac),
        _round_up_half(total * bb_frac),
        _round_up_half(total * dec_frac),
    )


def requires_parallelism(
    variant: ModelVariantSpec,
    *,
    max_vram_gib_per_device: float,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
) -> bool:
    peak = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    return peak + _PIPELINE_HEADROOM_GIB > max_vram_gib_per_device


def _single_device_plan(
    device: str,
    *,
    variant: ModelVariantSpec,
    rollout_steps: int,
    inference_precision: str | None,
) -> ParallelPlan:
    peak = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    enc, bb, dec = estimate_stage_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    return ParallelPlan(
        devices=(device,),
        encoder_device=device,
        backbone_device=device,
        decoder_device=device,
        estimated_peak_gib=peak,
        estimated_per_device_gib=(peak,),
    )


def _two_device_plan(
    devices: tuple[str, ...],
    *,
    variant: ModelVariantSpec,
    rollout_steps: int,
    inference_precision: str | None,
) -> ParallelPlan:
    enc_gib, bb_gib, dec_gib = estimate_stage_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    peak = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    host = devices[0]
    backbone = devices[1]
    host_load = enc_gib + dec_gib + _PIPELINE_HEADROOM_GIB
    backbone_load = bb_gib + _PIPELINE_HEADROOM_GIB
    return ParallelPlan(
        devices=devices[:2],
        encoder_device=host,
        backbone_device=backbone,
        decoder_device=host,
        estimated_peak_gib=peak,
        estimated_per_device_gib=(host_load, backbone_load),
    )


def _three_device_plan(
    devices: tuple[str, ...],
    *,
    variant: ModelVariantSpec,
    rollout_steps: int,
    inference_precision: str | None,
) -> ParallelPlan:
    enc_gib, bb_gib, dec_gib = estimate_stage_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    peak = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    return ParallelPlan(
        devices=devices[:3],
        encoder_device=devices[0],
        backbone_device=devices[1],
        decoder_device=devices[2],
        estimated_peak_gib=peak,
        estimated_per_device_gib=(
            enc_gib + _PIPELINE_HEADROOM_GIB,
            bb_gib + _PIPELINE_HEADROOM_GIB,
            dec_gib + _PIPELINE_HEADROOM_GIB,
        ),
    )


def plan_parallelism(
    variant: ModelVariantSpec,
    config: DistributedConfig,
    *,
    inference_precision: str | None = None,
) -> ParallelPlan:
    """Choose a pipeline placement that targets ``max_vram_gib_per_device``."""
    rollout_steps = config.rollout_steps
    peak = estimate_vram_gib(
        variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
    )
    fits_one = peak + _PIPELINE_HEADROOM_GIB <= config.max_vram_gib_per_device
    if len(config.devices) == 1:
        if not fits_one and not config.force:
            raise ValueError(
                f"Preset {variant.name!r} needs ~{peak:.1f} GiB reserved but "
                f"{config.devices[0]} budget is {config.max_vram_gib_per_device:.1f} GiB. "
                "Add GPUs or set distributed.force=True to attempt pipeline placement anyway."
            )
        return _single_device_plan(
            config.devices[0],
            variant=variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )

    if len(config.devices) == 2:
        plan = _two_device_plan(
            config.devices,
            variant=variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )
    elif len(config.devices) >= 3:
        plan = _three_device_plan(
            config.devices,
            variant=variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )
    else:
        raise ValueError("devices must contain at least one CUDA device id")

    if not config.force:
        overloaded = [
            (device, load)
            for device, load in zip(plan.devices, plan.estimated_per_device_gib, strict=True)
            if load > config.max_vram_gib_per_device
        ]
        if overloaded:
            details = ", ".join(f"{dev} ~{load:.1f} GiB" for dev, load in overloaded)
            raise ValueError(
                f"Pipeline plan exceeds per-device budget ({config.max_vram_gib_per_device:.1f} GiB): "
                f"{details}. Add another GPU or set distributed.force=True."
            )
    return plan
