"""Unit tests for Aurora 1.5 side path."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest
import torch

from flash_aurora.models.aurora import Batch, Metadata
from flash_aurora.models.aurora_v1p5 import AuroraV1p5, AuroraV1p5Ensemble, insolation, rollout
from flash_aurora.models.aurora_v1p5.normalisation import log_transform, log_untransform

_SURF_VARS = ("2t", "10u", "10v", "msl", "scaled_tp_1h", "insolation")
_STATIC_VARS = ("lsm", "z")
_ATMOS_VARS = ("z", "u", "v", "t", "q")
_OUTPUT_ONLY_SURF = ("scaled_tp_1h",)

H, W = 17, 32
N_LEVELS = 4
BATCH = 1
HISTORY = 2


def _make_batch(
    surf_vars: tuple[str, ...] = _SURF_VARS,
    static_vars: tuple[str, ...] = _STATIC_VARS,
    atmos_vars: tuple[str, ...] = _ATMOS_VARS,
) -> Batch:
    return Batch(
        surf_vars={k: torch.randn(BATCH, HISTORY, H, W) for k in surf_vars},
        static_vars={k: torch.randn(H, W) for k in static_vars},
        atmos_vars={k: torch.randn(BATCH, HISTORY, N_LEVELS, H, W) for k in atmos_vars},
        metadata=Metadata(
            lat=torch.linspace(90, -90, H),
            lon=torch.linspace(0, 360, W + 1)[:-1],
            time=(datetime(2023, 6, 15, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def _make_small_v1p5(**overrides: Any) -> AuroraV1p5:
    defaults: dict[str, Any] = dict(
        surf_vars=_SURF_VARS,
        static_vars=_STATIC_VARS,
        atmos_vars=_ATMOS_VARS,
        output_only_surf_vars=_OUTPUT_ONLY_SURF,
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
    defaults.update(overrides)
    return AuroraV1p5(**defaults)


def test_insolation_shape_and_finite() -> None:
    lat = np.linspace(90, -90, H, dtype=np.float32)
    lon = np.linspace(0, 360, W, endpoint=False, dtype=np.float32)
    values = insolation([datetime(2023, 6, 15, 12, 0)], lat, lon, enforce_2d=True)
    assert values.shape == (1, H, W)
    assert np.isfinite(values).all()


def test_v1p5_forward_zero_pads_output_only() -> None:
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(surf_vars=("2t", "10u", "10v", "msl", "insolation"))
    assert "scaled_tp_1h" not in batch.surf_vars
    lead = torch.full((BATCH,), 6.0)
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=lead)
    assert "scaled_tp_1h" in pred.surf_vars
    assert pred.surf_vars["scaled_tp_1h"].shape == (BATCH, 1, H - 1, W)


def test_v1p5_rollout_one_step_cpu() -> None:
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(surf_vars=("2t", "10u", "10v", "msl", "insolation"))
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=1))
    assert len(preds) == 1
    assert "insolation" in preds[0].surf_vars


def test_v1p5_rollout_fine_lead_times_cpu() -> None:
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(surf_vars=("2t", "10u", "10v", "msl", "insolation"))
    fine = [3.0, 6.0]
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=2, fine_lead_times=fine))
    assert len(preds) == 4
    assert preds[0].metadata.time[-1].hour == 15  # +3h from 12:00
    assert preds[1].metadata.time[-1].hour == 18  # +6h advances AR
    assert preds[2].metadata.time[-1].hour == 21  # next main step +3h


def test_v1p5_ensemble_factory_and_reset_noise() -> None:
    model = AuroraV1p5Ensemble(
        surf_vars=_SURF_VARS,
        static_vars=_STATIC_VARS,
        atmos_vars=_ATMOS_VARS,
        output_only_surf_vars=_OUTPUT_ONLY_SURF,
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
    assert model.backbone.stochastic is True
    model.reset_noise()
    model.set_noise_accumulation(n=6)


def test_log_transform_roundtrip() -> None:
    raw = torch.tensor([0.0, 1.0, 10.0])
    assert torch.allclose(log_untransform(log_transform(raw)), raw, atol=1e-5)
