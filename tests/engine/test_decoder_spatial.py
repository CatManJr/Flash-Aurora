from __future__ import annotations

import pytest
import torch

from flash_aurora.engine.distributed.decoder_spatial import (
    _split_backbone_tokens,
    forward_decoder_spatial_parallel,
)


@pytest.mark.gpu
def test_spatial_decoder_matches_unified_forward(engine_config_offline) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs at least two CUDA devices")

    from flash_aurora.engine.core.model_registry import ModelFactory
    from flash_aurora.engine.core.presets import DEFAULT_PRESETS
    from flash_aurora.engine.distributed.decoder_spatial import apply_decoder_spatial_placement
    from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder

    variant = DEFAULT_PRESETS.get("small_pretrained").variant
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        inference_precision="bf16_mixed@fp32",
    )
    model.eval()

    config = engine_config_offline
    builder = InitialConditionBuilder(config)
    try:
        batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")
    except FileNotFoundError:
        pytest.skip("small pretrained test pickle not available")

    batch = batch.to("cuda:0")
    with torch.inference_mode():
        prepared, transformed, patch_res = model._prepare_encoder_batch(batch)
        x = model.encoder(transformed, lead_time=model.timestep)
        x = model._run_backbone(
            x,
            lead_time=model.timestep,
            patch_res=patch_res,
            rollout_step=batch.metadata.rollout_step,
        )

    model.decoder.to("cuda:1")
    apply_decoder_spatial_placement(
        model,
        decoder_device="cuda:1",
        decoder_spatial_parallel=True,
        decoder_spatial_devices=("cuda:0", "cuda:1"),
    )
    x = x.to("cuda:1")

    with torch.inference_mode():
        unified = model.decoder.forward(x, batch, patch_res, model.timestep)
        spatial = forward_decoder_spatial_parallel(
            model,
            x,
            batch,
            patch_res=patch_res,
            lead_time=model.timestep,
            spatial_devices=(torch.device("cuda:0"), torch.device("cuda:1")),
            autocast_bf16=model.autocast_encoder_decoder,
            use_tensor_core=model.encoder_decoder_use_tensor_core,
        )

    for key in unified.surf_vars:
        assert torch.allclose(
            unified.surf_vars[key].float(),
            spatial.surf_vars[key].float(),
            rtol=1e-3,
            atol=1e-3,
        ), key
    for key in unified.atmos_vars:
        assert torch.allclose(
            unified.atmos_vars[key].float(),
            spatial.atmos_vars[key].float(),
            rtol=1e-3,
            atol=1e-3,
        ), key


def test_split_backbone_tokens_preserves_token_count() -> None:
    x = torch.randn(1, 4 * 3 * 5, 8)
    patch_res = (4, 3, 5)
    west, east, res_w, res_e, w_west, w_east = _split_backbone_tokens(x, patch_res)
    assert west.shape[1] == res_w[0] * res_w[1] * res_w[2]
    assert east.shape[1] == res_e[0] * res_e[1] * res_e[2]
    assert w_west + w_east == patch_res[2]
    assert west.shape[1] + east.shape[1] == x.shape[1]
