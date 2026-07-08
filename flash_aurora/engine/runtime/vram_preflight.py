from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from flash_aurora.engine.core.config import ModelVariantSpec
from flash_aurora.engine.distributed.config import DistributedConfig, resolve_distributed_config
from flash_aurora.engine.distributed.plan import plan_parallelism
from flash_aurora.engine.runtime.gpu_budget import (
    GPU_GUARD_RESERVED_FRACTION,
    estimate_vram_gib,
    is_exclusive_variant,
)
from flash_aurora.engine.runtime.gpu_memory import cuda_memory_snapshot, format_cuda_memory_snapshot

_DISTRIBUTED_HINT = (
    "Configure pipeline-parallel inference with "
    "AuroraEngine.from_preset(..., distributed=DistributedConfig("
    "devices=('cuda:0', 'cuda:1'))). "
    "Per-device VRAM is auto-detected from CUDA; override with "
    "max_vram_gib_per_device when needed."
)


class InsufficientVramError(RuntimeError):
    """Raised when peak VRAM cannot fit on the available hardware."""


@dataclass(frozen=True)
class InferenceVramBudget:
    """Peak VRAM reservation derived from the active inference settings."""

    preset: str
    variant_name: str
    rollout_steps: int
    inference_precision: str | None
    needed_gib: float
    exclusive: bool
    device_index: int
    device_total_gib: float
    device_budget_gib: float

    @property
    def physically_impossible(self) -> bool:
        return self.needed_gib > self.device_budget_gib


def compute_inference_vram_budget(
    variant: ModelVariantSpec,
    *,
    preset: str,
    rollout_steps: int,
    inference_precision: str | None,
    device_index: int = 0,
) -> InferenceVramBudget:
    """Estimate peak reserved VRAM from variant geometry and inference settings."""
    steps = max(1, rollout_steps)
    snapshot = cuda_memory_snapshot(device_index=device_index)
    needed = estimate_vram_gib(
        variant,
        rollout_steps=steps,
        inference_precision=inference_precision,
    )
    budget = snapshot.total_gib * GPU_GUARD_RESERVED_FRACTION
    return InferenceVramBudget(
        preset=preset,
        variant_name=variant.name,
        rollout_steps=steps,
        inference_precision=inference_precision,
        needed_gib=needed,
        exclusive=is_exclusive_variant(
            variant,
            rollout_steps=steps,
            inference_precision=inference_precision,
        ),
        device_index=device_index,
        device_total_gib=snapshot.total_gib,
        device_budget_gib=budget,
    )


def format_insufficient_vram_message(
    budget: InferenceVramBudget,
    *,
    reason: str | None = None,
) -> str:
    precision = budget.inference_precision or "default"
    lines = [
        (
            f"Preset {budget.preset!r} ({budget.variant_name}) needs ~{budget.needed_gib:.1f} GiB "
            f"peak VRAM for {budget.rollout_steps} rollout step(s) "
            f"with inference_precision={precision!r}, but GPU {budget.device_index} "
            f"has {budget.device_total_gib:.1f} GiB total "
            f"({budget.device_budget_gib:.1f} GiB usable budget)."
        ),
    ]
    if reason:
        lines.append(reason)
    lines.extend(
        [
            "This workload cannot run on a single GPU with this hardware.",
            "Options:",
            f"  - Use a GPU with at least ~{budget.needed_gib:.0f} GiB VRAM, or",
            f"  - {_DISTRIBUTED_HINT}",
        ]
    )
    return "\n".join(lines)


def format_distributed_insufficient_vram_message(
    *,
    preset: str,
    variant_name: str,
    rollout_steps: int,
    inference_precision: str | None,
    devices: tuple[str, ...],
    max_vram_gib_per_device: float,
    detail: str,
) -> str:
    precision = inference_precision or "default"
    device_list = ", ".join(devices)
    return "\n".join(
        [
            (
                f"Preset {preset!r} ({variant_name}) cannot be placed on "
                f"{len(devices)} GPU(s) [{device_list}] with "
                f"max_vram_gib_per_device={max_vram_gib_per_device:.1f} "
                f"for {rollout_steps} rollout step(s) "
                f"with inference_precision={precision!r}."
            ),
            detail,
            "Options:",
            "  - Try a GPU with more VRAM, or",
            f"  - Set distributed.force=True to attempt placement anyway, or",
            "  - Choose a smaller preset / fewer rollout steps.",
        ]
    )


def check_single_device_vram(
    variant: ModelVariantSpec,
    *,
    preset: str,
    rollout_steps: int,
    inference_precision: str | None,
    device_index: int = 0,
) -> InferenceVramBudget:
    """Fail fast when peak VRAM exceeds physical GPU capacity."""
    budget = compute_inference_vram_budget(
        variant,
        preset=preset,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
        device_index=device_index,
    )
    if budget.physically_impossible:
        snapshot = cuda_memory_snapshot(device_index=device_index)
        raise InsufficientVramError(
            format_insufficient_vram_message(budget)
            + "\n"
            + format_cuda_memory_snapshot(snapshot)
        )
    return budget


def check_distributed_vram(
    variant: ModelVariantSpec,
    config: DistributedConfig,
    *,
    preset: str,
    rollout_steps: int | None = None,
    inference_precision: str | None = None,
) -> None:
    """Validate pipeline placement before IC construction or model load."""
    steps = config.rollout_steps if rollout_steps is None else rollout_steps
    dist_config = resolve_distributed_config(replace(config, rollout_steps=max(1, steps)))
    try:
        plan_parallelism(
            variant,
            dist_config,
            inference_precision=inference_precision,
        )
    except ValueError as exc:
        raise InsufficientVramError(
            format_distributed_insufficient_vram_message(
                preset=preset,
                variant_name=variant.name,
                rollout_steps=dist_config.rollout_steps,
                inference_precision=inference_precision,
                devices=dist_config.devices,
                max_vram_gib_per_device=dist_config.max_vram_gib_per_device,
                detail=str(exc),
            )
        ) from exc
