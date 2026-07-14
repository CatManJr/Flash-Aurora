from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from flash_aurora.engine.core.config import V1P5_SURF_OUTPUT_ONLY
from flash_aurora.engine.core.model_protocol import is_v1p5_model_class, model_uses_v1p5_rollout
from flash_aurora.engine.core.model_registry import MODEL_REGISTRY, ModelFactory
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.core.rollout_session import RolloutSession
from flash_aurora.engine.ingress.adapters.era5_v1p5 import CdsEra5V1p5Adapter
from flash_aurora.engine.ingress.adapters.registry import DEFAULT_ADAPTERS
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.validator import BatchValidator
from flash_aurora.models.aurora import Batch, Metadata
from flash_aurora.models.aurora_v1p5 import AuroraV1p5, AuroraV1p5Ensemble


def test_preset_aurora_v1p5_registered() -> None:
    config = DEFAULT_PRESETS.get("aurora_v1p5")
    assert config.variant.model_class == "AuroraV1p5"
    assert config.variant.hf_repo == "ikwessel/aurora-1.5"
    assert config.source.name == "cds_era5_v1p5"
    assert config.source.time_policy == "first_two"
    assert config.cuda_graph is False
    assert config.inference_precision is None
    assert "AuroraV1p5" in MODEL_REGISTRY
    assert "cds_era5_v1p5" in DEFAULT_ADAPTERS.names()


def test_model_factory_accepts_inference_precision_for_v1p5() -> None:
    model = ModelFactory.create(
        "AuroraV1p5",
        use_lora=False,
        lora_mode="single",
        inference_precision="bf16_mixed@fp32",
        use_lora_merged_inference=True,
        surf_vars=("2t", "10u", "10v", "msl", "insolation", "scaled_tp_1h"),
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
        output_only_surf_vars=("scaled_tp_1h",),
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        autocast=False,
        use_fp16_safe_attention=False,
    )
    assert isinstance(model, AuroraV1p5)
    assert model.inference_config is not None
    assert not any(
        getattr(m, "use_fp16_safe_attention", False)
        for m in model.modules()
        if hasattr(m, "use_fp16_safe_attention")
    )
    # use_lora_merged_inference remains optimized-only and is stripped.
    assert not any(
        getattr(m, "use_lora_merged_inference", False)
        for m in model.modules()
        if hasattr(m, "use_lora_merged_inference")
    )


def test_validator_skips_output_only_surf_vars() -> None:
    from dataclasses import replace

    config = DEFAULT_PRESETS.get("aurora_v1p5")
    variant = replace(config.variant, resolution=(8, 16), levels=(100, 250, 500, 850))
    validator = BatchValidator(variant)
    required = [
        name for name in variant.surf_vars if name not in variant.output_only_surf_vars
    ]
    h, w = 8, 16
    batch = Batch(
        surf_vars={name: torch.zeros(1, 2, h, w) for name in required},
        static_vars={name: torch.zeros(h, w) for name in variant.static_vars},
        atmos_vars={
            name: torch.zeros(1, 2, len(variant.levels), h, w) for name in variant.atmos_vars
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=variant.levels,
        ),
    )
    for name in V1P5_SURF_OUTPUT_ONLY:
        assert name not in batch.surf_vars
    validator.validate(batch)


def test_adapter_lists_missing_surface_fields(tmp_path: Path) -> None:
    day = "2023-01-01"
    cache = tmp_path / "era5_v1p5"
    cache.mkdir()
    times = np.array(
        ["2023-01-01T00:00:00", "2023-01-01T06:00:00", "2023-01-01T12:00:00", "2023-01-01T18:00:00"],
        dtype="datetime64[s]",
    )
    lat = np.linspace(90, -90, 17)
    lon = np.linspace(0, 360, 32, endpoint=False)
    # Incomplete CDS short-name surface set.
    surface = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.zeros((4, 17, 32))),
            "u10": (("time", "latitude", "longitude"), np.zeros((4, 17, 32))),
            "v10": (("time", "latitude", "longitude"), np.zeros((4, 17, 32))),
            "msl": (("time", "latitude", "longitude"), np.zeros((4, 17, 32))),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    levels = list(DEFAULT_PRESETS.get("aurora_v1p5").variant.levels)
    atmospheric = xr.Dataset(
        {
            "z": (("time", "pressure_level", "latitude", "longitude"), np.zeros((4, 13, 17, 32))),
            "u": (("time", "pressure_level", "latitude", "longitude"), np.zeros((4, 13, 17, 32))),
            "v": (("time", "pressure_level", "latitude", "longitude"), np.zeros((4, 13, 17, 32))),
            "t": (("time", "pressure_level", "latitude", "longitude"), np.zeros((4, 13, 17, 32))),
            "q": (("time", "pressure_level", "latitude", "longitude"), np.zeros((4, 13, 17, 32))),
        },
        coords={
            "time": times,
            "pressure_level": levels,
            "latitude": lat,
            "longitude": lon,
        },
    )
    surface.to_netcdf(cache / f"{day}-surface-level.nc")
    atmospheric.to_netcdf(cache / f"{day}-atmospheric.nc")

    config = DEFAULT_PRESETS.get("aurora_v1p5")
    config.asset_root = tmp_path
    config.user_cwd = tmp_path
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        time_index=1,
        cache_dir=cache,
    )
    with pytest.raises(FileNotFoundError, match="incomplete"):
        CdsEra5V1p5Adapter().build_initial_batch(request, config)


def test_rollout_session_dispatches_v1p5() -> None:
    model = AuroraV1p5(
        surf_vars=("2t", "10u", "10v", "msl", "insolation", "scaled_tp_1h"),
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
        output_only_surf_vars=("scaled_tp_1h",),
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        use_lora=False,
        autocast=False,
        use_fp16_safe_attention=False,
    )
    model.eval()
    assert is_v1p5_model_class("AuroraV1p5")
    assert model_uses_v1p5_rollout(model)
    h, w = 17, 32
    batch = Batch(
        surf_vars={
            k: torch.randn(1, 2, h, w)
            for k in ("2t", "10u", "10v", "msl", "insolation")
        },
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z")},
        atmos_vars={k: torch.randn(1, 2, 4, h, w) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    preds = list(RolloutSession(model).run(batch, steps=1))
    assert len(preds) == 1


def test_preset_aurora_v1p5_ensemble_registered() -> None:
    from flash_aurora.engine.core.config import V1P5_CHECKPOINT_REVISION

    config = DEFAULT_PRESETS.get("aurora_v1p5_ensemble")
    assert config.variant.model_class == "AuroraV1p5Ensemble"
    assert config.variant.checkpoint_filename == "aurora-0.25-v1.5-ensemble.ckpt"
    assert config.variant.hf_repo == "ikwessel/aurora-1.5"
    assert config.hf_revision == V1P5_CHECKPOINT_REVISION
    assert config.source.name == "cds_era5_v1p5"
    assert "AuroraV1p5Ensemble" in MODEL_REGISTRY


def test_model_factory_builds_ensemble() -> None:
    model = ModelFactory.create(
        "AuroraV1p5Ensemble",
        use_lora=False,
        lora_mode="single",
        surf_vars=("2t", "10u", "10v", "msl", "insolation", "scaled_tp_1h"),
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
        output_only_surf_vars=("scaled_tp_1h",),
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        autocast=False,
        use_fp16_safe_attention=False,
    )
    assert isinstance(model, AuroraV1p5Ensemble)
    assert model.backbone.stochastic is True


def _tiny_v1p5_batch() -> Batch:
    h, w = 17, 32
    return Batch(
        surf_vars={
            k: torch.randn(1, 2, h, w)
            for k in ("2t", "10u", "10v", "msl", "insolation")
        },
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z")},
        atmos_vars={k: torch.randn(1, 2, 4, h, w) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def _tiny_v1p5_model() -> AuroraV1p5:
    return AuroraV1p5(
        surf_vars=("2t", "10u", "10v", "msl", "insolation", "scaled_tp_1h"),
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
        output_only_surf_vars=("scaled_tp_1h",),
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        use_lora=False,
        autocast=False,
        use_fp16_safe_attention=False,
    )


def test_rollout_session_fine_lead_times() -> None:
    model = _tiny_v1p5_model()
    model.eval()
    batch = _tiny_v1p5_batch()
    preds = list(
        RolloutSession(model).run(batch, steps=2, fine_lead_times=[3.0, 6.0])
    )
    assert len(preds) == 4


def test_forecast_step_prefers_metadata_time() -> None:
    from flash_aurora.engine.egress.forecast_step import ForecastStep

    batch = _tiny_v1p5_batch()
    # Simulate a +3h fine-lead prediction valid time.
    batch.metadata.time = (datetime(2023, 1, 1, 9),)
    step = ForecastStep.from_batch(
        batch,
        step_index=0,
        base_time=datetime(2023, 1, 1, 6),
        timestep_hours=6,
    )
    assert step.valid_time == datetime(2023, 1, 1, 9)
