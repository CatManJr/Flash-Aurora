"""Shared helpers for end-to-end latency benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from _pretrained_era5 import (
    _PYTORCH_BASELINE_KEY,
    pytorch_reference_tiers,
    purge_gpu,
    tier_entry,
    time_forward,
)

PYTORCH_FP32_REF_TIER = _PYTORCH_BASELINE_KEY

DEFAULT_LATENCY_TIERS: tuple[str, ...] = (
    PYTORCH_FP32_REF_TIER,
    "bf16_mixed@fp32",
    "bf16_mixed@tf32",
    "tf32@fp32",
    "tf32@tf32",
    "fp32@fp32",
    "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
)


def resolve_tier_specs(names: list[str]) -> list[tuple[str, str]]:
    pytorch_map = {label: precision for label, precision, _desc in pytorch_reference_tiers()}
    resolved: list[tuple[str, str]] = []
    for name in names:
        if name in pytorch_map:
            resolved.append((name, pytorch_map[name]))
            continue
        try:
            label, precision, _desc = tier_entry(name)
            resolved.append((label, precision))
        except ValueError:
            resolved.append((name, name))
    return resolved


def order_tier_specs_for_timing(
    specs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split tiers so the PyTorch FP32 ref is timed before Triton/CuTe pollute cuDNN.

    Custom kernels (and even a finetuned ``bf16_mixed`` warmup) can leave cuDNN
    autotune state that makes a later ``fp32`` baseline appear ~1.9x faster than a
    cold true PyTorch run.  Returns ``(ref_specs, other_specs)``; table order should
    still follow the original ``specs`` list.
    """
    ref = [s for s in specs if s[0] == PYTORCH_FP32_REF_TIER]
    other = [s for s in specs if s[0] != PYTORCH_FP32_REF_TIER]
    return ref, other


def build_model(
    config,
    ckpt: Path,
    *,
    precision: str,
    use_lora_merged_inference: bool,
    device: torch.device,
) -> Any:
    from flash_aurora.engine.core.model_registry import ModelFactory

    variant = config.variant
    kwargs: dict[str, Any] = {"inference_precision": precision}
    if variant.use_lora:
        kwargs["use_lora_merged_inference"] = use_lora_merged_inference
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        **kwargs,
    )
    model.load_checkpoint_local(str(ckpt), strict=variant.strict_checkpoint)
    model.eval()
    return model.to(device)


def time_forward_ms(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[float, float, float]:
    _, ms, peak_alloc, peak_reserved = time_forward(
        model,
        batch,
        warmup=warmup,
        repeat=repeat,
        device=device,
    )
    return ms, peak_alloc, peak_reserved


def run_tier_lora_modes(
    *,
    config,
    ckpt: Path,
    precision: str,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, tuple[float, float, float]]:
    """Return ``lora_eager`` / ``lora_merged`` timings, or single ``forward`` for no-LoRA."""
    out: dict[str, tuple[float, float, float]] = {}
    if config.variant.use_lora:
        for key, merged in (("lora_eager", False), ("lora_merged", True)):
            model = build_model(
                config,
                ckpt,
                precision=precision,
                use_lora_merged_inference=merged,
                device=device,
            )
            dev_batch = batch.to(device)
            try:
                out[key] = time_forward_ms(
                    model, dev_batch, warmup=warmup, repeat=repeat, device=device
                )
            finally:
                purge_gpu(model, dev_batch)
    else:
        model = build_model(
            config,
            ckpt,
            precision=precision,
            use_lora_merged_inference=False,
            device=device,
        )
        dev_batch = batch.to(device)
        try:
            out["forward"] = time_forward_ms(
                model, dev_batch, warmup=warmup, repeat=repeat, device=device
            )
        finally:
            purge_gpu(model, dev_batch)
    return out
