from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelStage(str, Enum):
    ENCODER = "encoder"
    BACKBONE = "backbone"
    DECODER = "decoder"


@dataclass(frozen=True)
class ParallelPlan:
    """Device placement for Aurora encoder / backbone / decoder pipeline stages."""

    devices: tuple[str, ...]
    encoder_device: str
    backbone_device: str
    decoder_device: str
    estimated_peak_gib: float
    estimated_per_device_gib: tuple[float, ...]
    decoder_spatial_parallel: bool = False
    decoder_spatial_devices: tuple[str, ...] = ()

    @property
    def input_device(self) -> str:
        return self.encoder_device

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    def device_for(self, stage: ParallelStage | str) -> str:
        key = stage.value if isinstance(stage, ParallelStage) else stage
        if key == ParallelStage.ENCODER.value:
            return self.encoder_device
        if key == ParallelStage.BACKBONE.value:
            return self.backbone_device
        if key == ParallelStage.DECODER.value:
            return self.decoder_device
        raise KeyError(f"Unknown pipeline stage: {stage!r}")


@dataclass(frozen=True)
class DistributedConfig:
    """Pipeline-parallel multi-GPU inference settings for :class:`AuroraEngine`."""

    devices: tuple[str, ...]
    max_vram_gib_per_device: float = 40.0
    rollout_steps: int = 1
    force: bool = False
    decoder_spatial_parallel: bool = True

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("DistributedConfig.devices must not be empty")
        if self.max_vram_gib_per_device <= 0:
            raise ValueError("max_vram_gib_per_device must be positive")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")
