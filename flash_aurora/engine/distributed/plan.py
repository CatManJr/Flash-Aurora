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

# Relative forward compute (era5 bf16 profile; sums to ~1.0).
_STAGE_COMPUTE_FRACTION: tuple[float, float, float] = (0.075, 0.627, 0.172)

# Backbone checkpoint weight footprint (GiB) when Swin lives on a dedicated GPU.
_BACKBONE_WEIGHT_GIB: dict[str, float] = {
    "AuroraSmallPretrained": 0.35,
}
_DEFAULT_BACKBONE_WEIGHT_GIB = 5.0

# Latent + batch retained on the encoder GPU while backbone runs elsewhere.
_CROSS_GPU_RETENTION_RATIO = 0.30

_PIPELINE_HEADROOM_GIB = 1.5

# Decoder activation halved per GPU when spatial columns run in parallel.
_DECODER_SPATIAL_VRAM_FRACTION = 0.50


def _round_up_half(value: float) -> float:
    return math.ceil(value * 2.0) / 2.0


def _stage_fractions(variant: ModelVariantSpec) -> tuple[float, float, float]:
    return _STAGE_FRACTIONS.get(variant.model_class, (0.22, 0.53, 0.25))


def _backbone_weight_gib(variant: ModelVariantSpec) -> float:
    return _BACKBONE_WEIGHT_GIB.get(variant.model_class, _DEFAULT_BACKBONE_WEIGHT_GIB)


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
    return ParallelPlan(
        devices=(device,),
        encoder_device=device,
        backbone_device=device,
        decoder_device=device,
        estimated_peak_gib=peak,
        estimated_per_device_gib=(peak,),
        decoder_spatial_parallel=False,
        decoder_spatial_devices=(),
    )


def _cross_gpu_retention_gib(bb_gib: float) -> float:
    return _round_up_half(bb_gib * _CROSS_GPU_RETENTION_RATIO)


def _spatial_decoder_vram_gib(dec_gib: float) -> float:
    return _round_up_half(dec_gib * _DECODER_SPATIAL_VRAM_FRACTION)


def _device_loads_two_gpu(
    *,
    encoder_idx: int,
    backbone_idx: int,
    decoder_idx: int,
    enc_gib: float,
    bb_gib: float,
    dec_gib: float,
    bb_weight_gib: float,
    decoder_spatial_parallel: bool,
) -> tuple[float, float]:
    """Estimate per-GPU peak GiB for a 2-GPU pipeline assignment."""
    loads = [0.0, 0.0]

    def add(idx: int, amount: float) -> None:
        loads[idx] += amount

    add(backbone_idx, bb_weight_gib)
    add(encoder_idx, enc_gib)

    spatial_dec = (
        decoder_spatial_parallel
        and decoder_idx == backbone_idx
        and encoder_idx != backbone_idx
    )
    if spatial_dec:
        dec_half = _spatial_decoder_vram_gib(dec_gib)
        add(encoder_idx, dec_half)
        add(backbone_idx, max(bb_gib, dec_half))
    elif decoder_idx == backbone_idx:
        add(backbone_idx, max(bb_gib, dec_gib))
    elif decoder_idx == encoder_idx and encoder_idx != backbone_idx:
        add(decoder_idx, dec_gib + _cross_gpu_retention_gib(bb_gib))
        add(backbone_idx, bb_gib)
    else:
        add(decoder_idx, dec_gib)
        add(backbone_idx, bb_gib)

    return (
        loads[0] + _PIPELINE_HEADROOM_GIB,
        loads[1] + _PIPELINE_HEADROOM_GIB,
    )


def _busy_fractions_two_gpu(
    *,
    encoder_idx: int,
    backbone_idx: int,
    decoder_idx: int,
) -> tuple[float, float]:
    enc_t, bb_t, dec_t = _STAGE_COMPUTE_FRACTION
    busy = [0.0, 0.0]
    busy[encoder_idx] += enc_t
    busy[backbone_idx] += bb_t
    busy[decoder_idx] += dec_t
    return busy[0], busy[1]


def _two_device_plan(
    devices: tuple[str, ...],
    *,
    variant: ModelVariantSpec,
    rollout_steps: int,
    inference_precision: str | None,
    decoder_spatial_parallel: bool,
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
    bb_weight = _backbone_weight_gib(variant)

    candidates: list[tuple[tuple[float, float], int, int, int, str]] = []
    for encoder_idx in (0, 1):
        for backbone_idx in (0, 1):
            if backbone_idx == encoder_idx:
                continue
            for decoder_idx in (0, 1):
                loads = _device_loads_two_gpu(
                    encoder_idx=encoder_idx,
                    backbone_idx=backbone_idx,
                    decoder_idx=decoder_idx,
                    enc_gib=enc_gib,
                    bb_gib=bb_gib,
                    dec_gib=dec_gib,
                    bb_weight_gib=bb_weight,
                    decoder_spatial_parallel=decoder_spatial_parallel,
                )
                # Sort key: min peak, then balance, prefer decoder colocated with backbone.
                colocate_dec_bb = decoder_idx == backbone_idx
                candidates.append(
                    (
                        loads,
                        encoder_idx,
                        backbone_idx,
                        decoder_idx,
                        "dec_with_backbone" if colocate_dec_bb else "dec_with_encoder",
                    )
                )

    def sort_key(item: tuple[tuple[float, float], int, int, int, str]) -> tuple[float, int, float]:
        loads, enc_i, _bb, _dec, layout = item
        peak_load = max(loads)
        prefer_dec_bb = 0 if layout == "dec_with_backbone" else 1
        return (peak_load, prefer_dec_bb, loads[enc_i])

    loads, encoder_idx, backbone_idx, decoder_idx, _layout = min(candidates, key=sort_key)

    spatial_split = (
        decoder_spatial_parallel
        and decoder_idx == backbone_idx
        and encoder_idx != backbone_idx
    )
    decoder_device = devices[decoder_idx]
    spatial_devices = (
        (devices[encoder_idx], decoder_device) if spatial_split else ()
    )

    return ParallelPlan(
        devices=devices[:2],
        encoder_device=devices[encoder_idx],
        backbone_device=devices[backbone_idx],
        decoder_device=decoder_device,
        estimated_peak_gib=peak,
        estimated_per_device_gib=loads,
        decoder_spatial_parallel=spatial_split,
        decoder_spatial_devices=spatial_devices,
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
    dec = devices[2]
    return ParallelPlan(
        devices=devices[:3],
        encoder_device=devices[0],
        backbone_device=devices[1],
        decoder_device=dec,
        estimated_peak_gib=peak,
        estimated_per_device_gib=(
            enc_gib + _PIPELINE_HEADROOM_GIB,
            bb_gib + _PIPELINE_HEADROOM_GIB,
            dec_gib + _PIPELINE_HEADROOM_GIB,
        ),
        decoder_spatial_parallel=False,
        decoder_spatial_devices=(),
    )


def plan_parallelism(
    variant: ModelVariantSpec,
    config: DistributedConfig,
    *,
    inference_precision: str | None = None,
) -> ParallelPlan:
    """Choose a 2-GPU pipeline placement that balances peak VRAM across devices."""
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
            decoder_spatial_parallel=config.decoder_spatial_parallel,
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


def estimate_device_busy_fraction(
    plan: ParallelPlan,
) -> tuple[tuple[str, float], ...]:
    """Rough per-device share of forward compute time (pipeline is sequential)."""
    if len(plan.devices) != 2:
        return tuple((device, 1.0) for device in plan.devices)

    device_to_idx = {name: idx for idx, name in enumerate(plan.devices)}
    enc_idx = device_to_idx[plan.encoder_device]
    bb_idx = device_to_idx[plan.backbone_device]
    dec_idx = device_to_idx[plan.decoder_device]
    busy = _busy_fractions_two_gpu(
        encoder_idx=enc_idx,
        backbone_idx=bb_idx,
        decoder_idx=dec_idx,
    )
    return tuple((plan.devices[i], busy[i]) for i in range(2))
