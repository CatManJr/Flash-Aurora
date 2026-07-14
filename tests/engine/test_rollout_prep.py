from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import torch

from flash_aurora.models.aurora import Batch, Metadata
from flash_aurora.engine.runtime.rollout_prep import (
    advance_rollout_batch,
    prepare_rollout_batch,
    warmup_forwards,
)


def _tiny_batch() -> Batch:
    lat = torch.from_numpy(np.linspace(90, -90, 5))
    lon = torch.from_numpy(np.linspace(0, 360, 8, endpoint=False))
    return Batch(
        surf_vars={"2t": torch.randn(1, 2, 5, 8)},
        static_vars={"lsm": torch.randn(5, 8)},
        atmos_vars={"t": torch.randn(1, 2, 2, 5, 8)},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 1, 1),),
            atmos_levels=(850, 925),
            rollout_step=0,
        ),
    )


def test_advance_rollout_batch_keeps_history_width() -> None:
    batch = _tiny_batch()
    pred = Batch(
        surf_vars={"2t": torch.randn(1, 1, 5, 8)},
        static_vars=batch.static_vars,
        atmos_vars={"t": torch.randn(1, 1, 2, 5, 8)},
        metadata=batch.metadata,
    )
    advanced = advance_rollout_batch(batch, pred)
    assert advanced.surf_vars["2t"].shape == (1, 2, 5, 8)


def test_warmup_forwards_zero_iters_is_noop() -> None:
    batch = _tiny_batch()
    model = MagicMock()
    out = warmup_forwards(model, batch, iters=0, device=torch.device("cpu"))
    assert out is batch
    model.forward.assert_not_called()


def test_warmup_forwards_passes_lead_times_for_v1p5() -> None:
    from datetime import timedelta

    batch = _tiny_batch()
    model = MagicMock()
    model.timestep = timedelta(hours=6)
    model.forward.return_value = batch
    # Structural marker used by model_uses_v1p5_rollout.
    model.variable_lead_time = True

    out = warmup_forwards(model, batch, iters=2, device=torch.device("cpu"))
    assert out is batch
    assert model.forward.call_count == 2
    for call in model.forward.call_args_list:
        assert call.args[0] is batch
        lead_times = call.kwargs["lead_times"]
        assert lead_times.shape == (1,)
        assert float(lead_times[0]) == 6.0
