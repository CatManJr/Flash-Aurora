from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.distributed import DistributedConfig
from flash_aurora.engine.runtime.vram_preflight import (
    InsufficientVramError,
    check_distributed_vram,
    check_single_device_vram,
    compute_inference_vram_budget,
)
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind


def _snapshot(*, total_gib: float, free_gib: float | None = None):
    return type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": free_gib if free_gib is not None else total_gib,
            "total_gib": total_gib,
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
            "other_processes_gib": 0.0,
        },
    )()


def test_compute_budget_uses_rollout_steps_and_precision() -> None:
    variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    with patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
        return_value=_snapshot(total_gib=95.0),
    ):
        one_step = compute_inference_vram_budget(
            variant,
            preset="era5_pretrained",
            rollout_steps=1,
            inference_precision="bf16_mixed@fp32",
        )
        two_step = compute_inference_vram_budget(
            variant,
            preset="era5_pretrained",
            rollout_steps=2,
            inference_precision="tf32",
        )
    assert two_step.needed_gib > one_step.needed_gib
    assert two_step.rollout_steps == 2


def test_single_device_fast_fail_on_24gib_card() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    with patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
        return_value=_snapshot(total_gib=24.0, free_gib=24.0),
    ):
        with pytest.raises(InsufficientVramError, match="DistributedConfig"):
            check_single_device_vram(
                variant,
                preset="hres_0.1",
                rollout_steps=2,
                inference_precision="bf16_mixed@fp32",
            )


def test_gpu_guard_fast_fail_without_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.runtime.gpu_guard import GpuGuardRegistry

    monkeypatch.setenv("FLASH_AURORA_GPU_GUARD", "1")
    registry = GpuGuardRegistry(tmp_path / "guard")
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    with patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
        return_value=_snapshot(total_gib=24.0, free_gib=24.0),
    ):
        with pytest.raises(InsufficientVramError):
            registry.acquire(
                device_index=0,
                preset="hres_0.1",
                variant=variant,
                rollout_steps=2,
                timeout=0.2,
            )


def test_distributed_preflight_unified_message() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    with pytest.raises(InsufficientVramError, match="cannot be placed"):
        check_distributed_vram(
            variant,
            DistributedConfig(
                devices=("cuda:0", "cuda:1"),
                max_vram_gib_per_device=24.0,
                rollout_steps=1,
            ),
            preset="hres_0.1",
            rollout_steps=1,
            inference_precision="bf16_mixed@fp32",
        )