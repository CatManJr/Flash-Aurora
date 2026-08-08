"""Aurora 1.5 pipeline-parallel wiring (no hard reject; lead_times path)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.distributed import DistributedConfig
from flash_aurora.engine.distributed.lead_routing import (
    encoder_decoder_lead_kwargs,
    resolve_lead_times_tensor,
    uses_variable_lead_times,
)
from flash_aurora.engine.distributed.pipeline import pipeline_forward
from flash_aurora.models.aurora.batch import Batch, Metadata
from flash_aurora.models.aurora_v1p5.rollout import _advance_batch


def test_from_preset_accepts_distributed_for_aurora_v1p5(tmp_path: Path) -> None:
    from flash_aurora.engine.core.engine import AuroraEngine

    engine = AuroraEngine.from_preset(
        "aurora_v1p5",
        asset_root=tmp_path,
        distributed=DistributedConfig(
            devices=("cuda:0", "cuda:1"),
            max_vram_gib_per_device=32.0,
            force=True,
        ),
    )
    assert engine.config.distributed is not None
    assert engine.config.cuda_graph is False


def test_lead_routing_for_v1p5_factory_model() -> None:
    from flash_aurora.engine.core.model_registry import ModelFactory

    variant = DEFAULT_PRESETS.get("aurora_v1p5").variant
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
    )
    assert uses_variable_lead_times(model)

    batch = Batch(
        surf_vars={"2t": torch.zeros(1, 2, 4, 4)},
        static_vars={"lsm": torch.zeros(4, 4)},
        atmos_vars={"z": torch.zeros(1, 2, 1, 4, 4)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 4),
            lon=torch.linspace(0, 360, 4)[:-1],
            time=(datetime(2023, 1, 1, 0),),
            atmos_levels=(500,),
            rollout_step=0,
        ),
    )
    lead = resolve_lead_times_tensor(model, batch)
    assert lead is not None
    assert lead.shape == (1,)
    assert abs(float(lead.item()) - 6.0) < 1e-5
    assert encoder_decoder_lead_kwargs(model, lead) == {"lead_times": lead}


def test_pipeline_forward_signature_accepts_lead_times() -> None:
    import inspect

    params = inspect.signature(pipeline_forward).parameters
    assert "lead_times" in params


def test_v1p5_advance_drops_output_only_keys() -> None:
    ic = Batch(
        surf_vars={
            "2t": torch.zeros(1, 2, 2, 2),
            "10u": torch.zeros(1, 2, 2, 2),
        },
        static_vars={"lsm": torch.zeros(2, 2)},
        atmos_vars={"z": torch.zeros(1, 2, 1, 2, 2)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 2),
            lon=torch.linspace(0, 360, 2)[:-1],
            time=(datetime(2023, 1, 1, 0),),
            atmos_levels=(500,),
            rollout_step=0,
        ),
    )
    pred = Batch(
        surf_vars={
            "2t": torch.ones(1, 1, 2, 2),
            "10u": torch.ones(1, 1, 2, 2),
            "tp": torch.ones(1, 1, 2, 2),  # output-only in IC-absent sense
        },
        static_vars=ic.static_vars,
        atmos_vars={"z": torch.ones(1, 1, 1, 2, 2)},
        metadata=Metadata(
            lat=ic.metadata.lat,
            lon=ic.metadata.lon,
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=(500,),
            rollout_step=1,
        ),
    )
    nxt = _advance_batch(ic, pred)
    assert "tp" not in nxt.surf_vars
    assert set(nxt.surf_vars) == {"2t", "10u"}
    assert nxt.surf_vars["2t"].shape[1] == 2
