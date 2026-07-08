from __future__ import annotations

from unittest.mock import patch

import pytest

from flash_aurora.engine.distributed.config import DistributedConfig, resolve_distributed_config


def _snapshot(*, device_index: int, total_gib: float):
    return type(
        "Snap",
        (),
        {
            "device_index": device_index,
            "free_gib": total_gib,
            "total_gib": total_gib,
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
            "other_processes_gib": 0.0,
        },
    )()


def test_probe_max_vram_uses_minimum_across_devices() -> None:
    from flash_aurora.engine.runtime.gpu_memory import probe_max_vram_gib_per_device

    with patch(
        "flash_aurora.engine.runtime.gpu_memory.cuda_memory_snapshot",
        side_effect=[
            _snapshot(device_index=0, total_gib=32.0),
            _snapshot(device_index=1, total_gib=24.0),
        ],
    ):
        budget = probe_max_vram_gib_per_device(("cuda:0", "cuda:1"))
    assert budget == 24.0


def test_probe_max_vram_honors_override() -> None:
    from flash_aurora.engine.runtime.gpu_memory import probe_max_vram_gib_per_device

    budget = probe_max_vram_gib_per_device(
        ("cuda:0", "cuda:1"),
        override=40.0,
    )
    assert budget == 40.0


def test_resolve_distributed_config_probes_when_unset() -> None:
    config = DistributedConfig(devices=("cuda:0", "cuda:1"))
    with patch(
        "flash_aurora.engine.runtime.gpu_memory.probe_max_vram_gib_per_device",
        return_value=24.0,
    ):
        resolved = resolve_distributed_config(config)
    assert resolved.max_vram_gib_per_device == 24.0


def test_resolve_distributed_config_keeps_explicit_override() -> None:
    config = DistributedConfig(
        devices=("cuda:0", "cuda:1"),
        max_vram_gib_per_device=32.0,
    )
    resolved = resolve_distributed_config(config)
    assert resolved.max_vram_gib_per_device == 32.0
