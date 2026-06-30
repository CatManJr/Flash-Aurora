from __future__ import annotations

import pytest
import torch

from flash_aurora.engine.distributed import DistributedConfig
from flash_aurora.engine.distributed.rollout_pipeline import distributed_rollout


@pytest.mark.gpu
def test_distributed_rollout_matches_rollout_stream(engine_config_offline) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    from flash_aurora.engine.core.engine import AuroraEngine
    from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder

    config = engine_config_offline
    config.distributed = DistributedConfig(
        devices=("cuda:0", "cuda:1"),
        max_vram_gib_per_device=32.0,
        force=True,
    )
    engine = AuroraEngine(config)
    builder = InitialConditionBuilder(config)
    try:
        batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")
    except FileNotFoundError:
        pytest.skip("small pretrained test pickle not available")

    try:
        engine.load()
        with torch.inference_mode():
            pipelined = list(
                distributed_rollout(engine.model, batch, 2)
            )
            streamed = list(engine.rollout_stream(batch, 2))

        assert len(pipelined) == len(streamed) == 2
        for pipe_pred, stream_pred in zip(pipelined, streamed):
            for key in pipe_pred.surf_vars:
                assert torch.allclose(
                    pipe_pred.surf_vars[key].float(),
                    stream_pred.surf_vars[key].float(),
                    rtol=1e-3,
                    atol=1e-3,
                )
            for key in pipe_pred.atmos_vars:
                assert torch.allclose(
                    pipe_pred.atmos_vars[key].float(),
                    stream_pred.atmos_vars[key].float(),
                    rtol=1e-3,
                    atol=1e-3,
                )
    finally:
        engine.release_gpu()
