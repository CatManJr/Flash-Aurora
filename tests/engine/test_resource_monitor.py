import pytest

from flash_aurora.engine.runtime.resource_monitor import (
    GpuResourceSample,
    ResourceSample,
    device_index_from_name,
    plot_distributed_rollout_utilization,
    resource_samples_to_dict,
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


def test_resource_samples_to_dict_roundtrip_fields() -> None:
    gpu = GpuResourceSample(
        index=0,
        name="GPU",
        utilization_percent=80.0,
        memory_used_mib=1024.0,
        memory_total_mib=2048.0,
    )
    samples = [
        ResourceSample(
            time_s=0.0,
            cpu_percent=12.5,
            dram_used_gib=8.0,
            dram_utilization_percent=25.0,
            gpus={0: gpu},
        )
    ]
    payload = resource_samples_to_dict(samples)
    assert payload[0]["cpu_percent"] == pytest.approx(12.5)
    assert payload[0]["gpus"]["0"]["utilization_percent"] == pytest.approx(80.0)


def test_plot_distributed_rollout_utilization_writes_file(tmp_path) -> None:
    gpu0 = GpuResourceSample(0, "A", 70.0, 1000.0, 2000.0)
    gpu1 = GpuResourceSample(1, "B", 40.0, 1500.0, 2000.0)
    samples = [
        ResourceSample(
            time_s=float(step) * 0.1,
            cpu_percent=10.0 + step,
            dram_used_gib=4.0,
            dram_utilization_percent=20.0 + step,
            gpus={0: gpu0, 1: gpu1},
        )
        for step in range(5)
    ]
    paths = plot_distributed_rollout_utilization(
        {"staged": samples, "overlap": samples},
        device_indices=[0, 1],
        output_path=tmp_path / "util.png",
    )
    assert len(paths) == 2
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
