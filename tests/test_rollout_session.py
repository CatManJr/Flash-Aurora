from __future__ import annotations

import pytest
import torch
from aurora import AuroraSmallPretrained, rollout

from engine.core.rollout_session import RolloutSession
from tests.helpers import assert_batches_close, smoke_batch


@pytest.mark.integration
@pytest.mark.gpu
def test_rollout_session_matches_aurora_rollout() -> None:
    model = AuroraSmallPretrained(use_lora=False)
    model.eval()
    batch = smoke_batch()
    steps = 1

    with torch.inference_mode():
        reference = list(rollout(model, batch, steps))
        session = list(RolloutSession(model).run(batch, steps))

    assert len(reference) == steps
    assert len(session) == steps
    for ref_step, session_step in zip(reference, session):
        assert_batches_close(ref_step, session_step)
