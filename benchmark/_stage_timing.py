"""CUDA-event stage timings for Aurora forward (encoder / backbone / decoder)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StageTiming:
    encoder_ms: float
    backbone_ms: float
    decoder_ms: float
    post_ms: float
    total_ms: float

    @property
    def backbone_pct(self) -> float:
        return 100.0 * self.backbone_ms / self.total_ms if self.total_ms > 0 else 0.0

    @property
    def encoder_decoder_ms(self) -> float:
        return self.encoder_ms + self.decoder_ms + self.post_ms


def _run_forward_stages_once(
    model: Any,
    batch: Any,
    *,
    device: torch.device,
) -> tuple[StageTiming, Any]:
    """One forward with CUDA events between encoder, backbone, decoder, post."""
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

    batch, transformed_batch, patch_res = model._prepare_encoder_batch(batch)
    rollout_step = batch.metadata.rollout_step

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    e4 = torch.cuda.Event(enable_timing=True)

    e0.record()
    x = run_with_encoder_decoder_routing(
        model.encoder,
        transformed_batch,
        autocast_bf16=model.autocast_encoder_decoder,
        use_tensor_core=model.encoder_decoder_use_tensor_core,
        lead_time=model.timestep,
    )
    e1.record()

    x = model._run_backbone(
        x,
        lead_time=model.timestep,
        patch_res=patch_res,
        rollout_step=rollout_step,
    )
    e2.record()

    pred = run_with_encoder_decoder_routing(
        model.decoder,
        x,
        batch,
        autocast_bf16=model.autocast_encoder_decoder,
        use_tensor_core=model.encoder_decoder_use_tensor_core,
        lead_time=model.timestep,
        patch_res=patch_res,
    )
    e3.record()

    pred = model._finish_prediction(batch, pred)
    e4.record()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    enc = e0.elapsed_time(e1)
    bb = e1.elapsed_time(e2)
    dec = e2.elapsed_time(e3)
    post = e3.elapsed_time(e4)
    total = e0.elapsed_time(e4)
    return StageTiming(enc, bb, dec, post, total), pred


def time_forward_stages(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[StageTiming, Any]:
    """Average per-stage CUDA times over ``repeat`` forwards (after ``warmup``)."""
    with torch.inference_mode():
        for _ in range(warmup):
            _run_forward_stages_once(model, batch, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        enc_sum = bb_sum = dec_sum = post_sum = total_sum = 0.0
        pred = None
        for _ in range(repeat):
            timing, pred = _run_forward_stages_once(model, batch, device=device)
            enc_sum += timing.encoder_ms
            bb_sum += timing.backbone_ms
            dec_sum += timing.decoder_ms
            post_sum += timing.post_ms
            total_sum += timing.total_ms

        n = float(repeat)
        avg = StageTiming(
            encoder_ms=enc_sum / n,
            backbone_ms=bb_sum / n,
            decoder_ms=dec_sum / n,
            post_ms=post_sum / n,
            total_ms=total_sum / n,
        )
    return avg, pred
