import pytest

from flash_aurora.engine.runtime.resource_monitor import (
    GpuResourceSample,
    device_index_from_name,
)


def test_device_index_from_name_parses_cuda_device() -> None:
    assert device_index_from_name("cuda:3") == 3


def test_device_index_from_name_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="expected CUDA device name"):
        device_index_from_name("cpu")


def test_gpu_resource_sample_reports_vram_utilization() -> None:
    sample = GpuResourceSample(
        index=0,
        name="NVIDIA RTX PRO 6000",
        utilization_percent=75.0,
        memory_used_mib=48_000.0,
        memory_total_mib=96_000.0,
    )

    assert sample.memory_used_gib == pytest.approx(46.875)
    assert sample.memory_utilization_percent == pytest.approx(50.0)
