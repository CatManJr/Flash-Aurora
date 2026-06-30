"""JSON payload roundtrip for distributed rollout benchmark subprocess IPC."""

from __future__ import annotations

from _distributed_rollout_ipc import (
    RolloutCaseResult,
    case_from_payload,
    case_to_payload,
)
from flash_aurora.engine.runtime.resource_monitor import (
    GpuResourceSample,
    ResourceSample,
)


def test_case_payload_roundtrip_preserves_resource_samples() -> None:
    samples = [
        ResourceSample(
            time_s=0.0,
            cpu_percent=12.5,
            dram_used_gib=8.0,
            dram_utilization_percent=25.0,
            gpus={
                0: GpuResourceSample(
                    index=0,
                    name="GPU0",
                    utilization_percent=40.0,
                    memory_used_mib=1024.0,
                    memory_total_mib=32768.0,
                ),
                1: GpuResourceSample(
                    index=1,
                    name="GPU1",
                    utilization_percent=55.0,
                    memory_used_mib=2048.0,
                    memory_total_mib=32768.0,
                ),
            },
        )
    ]
    case = RolloutCaseResult(
        mode="2gpu",
        total_ms=4816.0,
        per_step_ms=1204.0,
        peak_allocated_gib={"cuda:0": 13.3, "cuda:1": 18.1},
        distributed_status={"enabled": True, "decoder_spatial_parallel": True},
        resource_samples=samples,
    )

    payload = case_to_payload(case, preset="era5_pretrained")
    restored = case_from_payload(payload)

    assert payload["preset"] == "era5_pretrained"
    assert restored.mode == case.mode
    assert restored.total_ms == case.total_ms
    assert restored.peak_allocated_gib == case.peak_allocated_gib
    assert restored.distributed_status == case.distributed_status
    assert restored.resource_samples is not None
    assert len(restored.resource_samples) == 1
    assert restored.resource_samples[0].gpus[1].utilization_percent == 55.0
