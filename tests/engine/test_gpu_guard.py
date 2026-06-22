from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.runtime.gpu_budget import estimate_vram_gib, is_exclusive_variant
from flash_aurora.engine.runtime.gpu_guard import GpuGuardRegistry


def test_estimate_vram_hres_01_is_exclusive() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    assert estimate_vram_gib(variant, rollout_steps=2) >= 82.5
    assert is_exclusive_variant(variant, rollout_steps=2)


def test_estimate_vram_small_is_shareable() -> None:
    variant = DEFAULT_PRESETS.get("small_pretrained").variant
    assert estimate_vram_gib(variant) < 10.0
    assert not is_exclusive_variant(variant)


def test_gpu_guard_allows_two_small_leases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASH_AURORA_GPU_GUARD", "1")
    registry = GpuGuardRegistry(tmp_path / "guard")
    small = DEFAULT_PRESETS.get("small_pretrained").variant

    snapshot = type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 80.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 1.0,
            "torch_reserved_gib": 1.0,
            "other_processes_gib": 0.0,
        },
    )()

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=snapshot,
    ), patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", side_effect=[1001, 1002]):
        first = registry.acquire(
            device_index=0,
            preset="small_pretrained",
            variant=small,
            rollout_steps=1,
            timeout=1.0,
        )
        second = registry.acquire(
            device_index=0,
            preset="small_pretrained",
            variant=small,
            rollout_steps=1,
            timeout=1.0,
        )

    assert first.reserved_gib < 10.0
    assert second.reserved_gib < 10.0
    first.release()
    second.release()


def test_gpu_guard_queues_exclusive_when_memory_tight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASH_AURORA_GPU_GUARD", "1")
    registry = GpuGuardRegistry(tmp_path / "guard")
    large = DEFAULT_PRESETS.get("hres_0.1").variant
    small = DEFAULT_PRESETS.get("small_pretrained").variant

    roomy = type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 90.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 1.0,
            "torch_reserved_gib": 1.0,
            "other_processes_gib": 0.0,
        },
    )()
    blocked = type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 10.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 70.0,
            "torch_reserved_gib": 75.0,
            "other_processes_gib": 0.0,
        },
    )()

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=roomy,
    ), patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=2001):
        small_ticket = registry.acquire(
            device_index=0,
            preset="small_pretrained",
            variant=small,
            rollout_steps=1,
            timeout=1.0,
        )

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=blocked,
    ), patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=2002):
        with pytest.raises(TimeoutError, match="Timed out waiting for GPU"):
            registry.acquire(
                device_index=0,
                preset="hres_0.1",
                variant=large,
                rollout_steps=2,
                timeout=0.2,
            )

    with patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=2001):
        small_ticket.release()

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=roomy,
    ), patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=2002):
        large_ticket = registry.acquire(
            device_index=0,
            preset="hres_0.1",
            variant=large,
            rollout_steps=2,
            timeout=1.0,
        )
        assert large_ticket.exclusive
        large_ticket.release()


def test_engine_acquire_and_release_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.core.engine import AuroraEngine

    monkeypatch.setenv("FLASH_AURORA_GPU_GUARD", "1")
    engine = AuroraEngine.from_preset("small_pretrained", asset_root=tmp_path)
    engine.config.gpu_guard_timeout = 1.0

    snapshot = type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 80.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
            "other_processes_gib": 0.0,
        },
    )()

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=snapshot,
    ):
        ticket = engine.acquire_gpu(rollout_steps=1)
        assert ticket is not None
        engine.release_gpu(move_model_to_cpu=False)
        assert engine._gpu_ticket is None
