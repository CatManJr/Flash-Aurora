from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.runtime.gpu_budget import estimate_vram_gib, is_exclusive_variant
from flash_aurora.engine.runtime.gpu_guard import GpuGuardRegistry, GpuLeaseRecord


def test_estimate_vram_hres_01_is_exclusive() -> None:
    variant = DEFAULT_PRESETS.get("hres_0.1").variant
    assert estimate_vram_gib(variant, rollout_steps=2) >= 70.0
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
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
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
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
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
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
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
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
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
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
        return_value=snapshot,
    ):
        ticket = engine.acquire_gpu(rollout_steps=1)
        assert ticket is not None
        engine.release_gpu()
        assert engine._gpu_ticket is None


_HEARTBEAT_WAIT_S = 3.0


def _roomy_snapshot():
    return type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 90.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 0.0,
            "torch_reserved_gib": 0.0,
            "other_processes_gib": 0.0,
        },
    )()


@contextmanager
def _roomy_gpu():
    snapshot = _roomy_snapshot()
    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot",
        return_value=snapshot,
    ), patch(
        "flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot",
        return_value=snapshot,
    ):
        yield


def _own_leases(registry: GpuGuardRegistry) -> list[GpuLeaseRecord]:
    return [lease for lease in registry.status(device_index=0).leases if lease.pid == os.getpid()]


def _acquire_one_step(registry: GpuGuardRegistry, *, preset: str = "small_pretrained"):
    return registry.acquire(
        device_index=0,
        preset=preset,
        variant=DEFAULT_PRESETS.get(preset).variant,
        rollout_steps=1,
        timeout=1.0,
    )


def test_lease_survives_until_every_ticket_of_the_process_is_released(tmp_path: Path) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    with _roomy_gpu():
        first = _acquire_one_step(registry)
        second = _acquire_one_step(registry)

        first.release()
        assert len(_own_leases(registry)) == 1

        second.release()
        assert _own_leases(registry) == []


def test_second_ticket_of_the_process_grows_the_reservation(tmp_path: Path) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    larger_variant = DEFAULT_PRESETS.get("era5_pretrained").variant
    with _roomy_gpu():
        small_ticket = _acquire_one_step(registry)
        larger_ticket = _acquire_one_step(registry, preset="era5_pretrained")

        [lease] = _own_leases(registry)
        assert lease.reserved_gib == estimate_vram_gib(larger_variant, rollout_steps=1)
        assert lease.reserved_gib > small_ticket.reserved_gib

        small_ticket.release()
        larger_ticket.release()


def test_background_heartbeat_keeps_an_idle_lease_fresh(tmp_path: Path) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard", heartbeat_seconds=0.05)
    with _roomy_gpu():
        ticket = _acquire_one_step(registry)
        [lease] = _own_leases(registry)
        acquired_heartbeat = lease.heartbeat

        deadline = time.monotonic() + _HEARTBEAT_WAIT_S
        while time.monotonic() < deadline:
            [lease] = _own_leases(registry)
            if lease.heartbeat > acquired_heartbeat:
                break
            time.sleep(0.02)

        assert lease.heartbeat > acquired_heartbeat
        ticket.release()


def test_registry_without_fcntl_explains_how_to_disable_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    monkeypatch.setitem(sys.modules, "fcntl", None)

    with pytest.raises(RuntimeError, match="FLASH_AURORA_GPU_GUARD=0"):
        registry.status(device_index=0)


def _starved_snapshot():
    return type(
        "Snap",
        (),
        {
            "device_index": 0,
            "free_gib": 10.0,
            "total_gib": 95.0,
            "torch_allocated_gib": 80.0,
            "torch_reserved_gib": 85.0,
            "other_processes_gib": 0.0,
        },
    )()


@pytest.fixture
def other_process_pid() -> Iterator[int]:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process.pid
    process.terminate()
    process.wait(timeout=10)


def _acquire_as(registry: GpuGuardRegistry, pid: int, preset: str):
    with patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=pid):
        return _acquire_one_step(registry, preset=preset)


def test_growth_is_refused_when_the_device_lacks_the_extra_memory(tmp_path: Path) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    with _roomy_gpu():
        held = _acquire_one_step(registry, preset="era5_pretrained")
    starved = _starved_snapshot()

    with patch(
        "flash_aurora.engine.runtime.gpu_guard.cuda_memory_snapshot", return_value=starved
    ), patch("flash_aurora.engine.runtime.vram_preflight.cuda_memory_snapshot", return_value=starved):
        with pytest.raises(TimeoutError, match="only 10.0 GiB free"):
            _acquire_one_step(registry, preset="hres_0.1")

    held.release()


def test_process_cannot_become_exclusive_while_another_process_shares_the_gpu(
    tmp_path: Path, other_process_pid: int
) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    with _roomy_gpu():
        neighbour = _acquire_as(registry, other_process_pid, "small_pretrained")
        shared = _acquire_one_step(registry)

        with pytest.raises(TimeoutError, match="cannot become exclusive"):
            _acquire_one_step(registry, preset="era5_pretrained")

        shared.release()
        with patch("flash_aurora.engine.runtime.gpu_guard.os.getpid", return_value=other_process_pid):
            neighbour.release()


def test_lease_becomes_exclusive_when_a_lone_process_grows_into_an_exclusive_preset(
    tmp_path: Path,
) -> None:
    registry = GpuGuardRegistry(tmp_path / "guard")
    with _roomy_gpu():
        shared = _acquire_one_step(registry)
        exclusive = _acquire_one_step(registry, preset="era5_pretrained")

        [lease] = _own_leases(registry)
        assert lease.exclusive

        shared.release()
        exclusive.release()
