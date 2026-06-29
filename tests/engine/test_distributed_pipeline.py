from __future__ import annotations

import pytest
import torch

from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.distributed import (
    DistributedConfig,
    apply_pipeline_parallel,
    is_pipeline_parallel,
    plan_parallelism,
)
from flash_aurora.engine.distributed.pipeline import restore_pipeline_parallel


@pytest.mark.gpu
def test_pipeline_parallel_forward_small_pretrained(engine_config_offline) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    from flash_aurora.engine.core.engine import AuroraEngine
    from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder

    config = engine_config_offline
    config.inference_precision = "bf16_mixed@fp32"
    config.distributed = DistributedConfig(
        devices=("cuda:0", "cuda:1"),
        max_vram_gib_per_device=32.0,
        rollout_steps=1,
        force=True,
    )
    engine = AuroraEngine(config)
    builder = InitialConditionBuilder(config)
    batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")

    try:
        engine.load()
        status = engine.distributed_status()
        assert status["enabled"] is True
        assert is_pipeline_parallel(engine.model)

        pred = engine.predict(batch)
        assert next(iter(pred.surf_vars.values())).device.type == "cuda"
        assert len(list(engine.rollout_stream(batch, steps=2))) == 2
    finally:
        engine.release_gpu()


def test_apply_pipeline_parallel_restores_forward() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    from flash_aurora.engine.core.model_registry import ModelFactory

    variant = DEFAULT_PRESETS.get("small_pretrained").variant
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        inference_precision="bf16_mixed@fp32",
    )
    plan = plan_parallelism(
        variant,
        DistributedConfig(
            devices=("cuda:0", "cuda:1"),
            max_vram_gib_per_device=32.0,
            force=True,
        ),
    )
    apply_pipeline_parallel(model, plan)
    assert is_pipeline_parallel(model)
    assert str(next(model.encoder.parameters()).device) == "cuda:0"
    assert str(next(model.backbone.parameters()).device) == "cuda:1"

    restore_pipeline_parallel(model)
    assert not is_pipeline_parallel(model)
    assert str(next(model.parameters()).device) == "cuda:0"
