"""Subprocess IPC helpers for distributed rollout benchmark results."""

from __future__ import annotations

from dataclasses import dataclass

from flash_aurora.engine.runtime.resource_monitor import (
    GpuResourceSample,
    ResourceSample,
    resource_samples_to_dict,
)

RESULT_PREFIX = "FLASH_AURORA_ROLLOUT_JSON "


@dataclass
class RolloutCaseResult:
    mode: str
    total_ms: float
    per_step_ms: float
    peak_allocated_gib: dict[str, float]
    distributed_status: dict[str, object]
    resource_samples: list[ResourceSample] | None = None


def case_to_payload(case: RolloutCaseResult, *, preset: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": case.mode,
        "total_ms": case.total_ms,
        "per_step_ms": case.per_step_ms,
        "peak_allocated_gib": case.peak_allocated_gib,
        "distributed_status": case.distributed_status,
        "resource_samples": (
            resource_samples_to_dict(case.resource_samples)
            if case.resource_samples is not None
            else None
        ),
    }
    if preset is not None:
        payload["preset"] = preset
    return payload


def case_from_payload(payload: dict[str, object]) -> RolloutCaseResult:
    samples_payload = payload.get("resource_samples")
    resource_samples = None
    if samples_payload is not None:
        resource_samples = [
            ResourceSample(
                time_s=float(row["time_s"]),
                cpu_percent=float(row["cpu_percent"]),
                dram_used_gib=float(row["dram_used_gib"]),
                dram_utilization_percent=float(row["dram_utilization_percent"]),
                gpus={
                    int(index): GpuResourceSample(
                        index=int(gpu["index"]),
                        name=gpu.get("name"),
                        utilization_percent=gpu.get("utilization_percent"),
                        memory_used_mib=gpu.get("memory_used_mib"),
                        memory_total_mib=gpu.get("memory_total_mib"),
                    )
                    for index, gpu in row["gpus"].items()
                },
            )
            for row in samples_payload
        ]
    peak = payload["peak_allocated_gib"]
    return RolloutCaseResult(
        mode=str(payload["mode"]),
        total_ms=float(payload["total_ms"]),
        per_step_ms=float(payload["per_step_ms"]),
        peak_allocated_gib={str(k): float(v) for k, v in peak.items()},
        distributed_status=dict(payload["distributed_status"]),
        resource_samples=resource_samples,
    )
