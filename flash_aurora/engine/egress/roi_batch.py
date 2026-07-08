from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from flash_aurora.engine.egress.mask import Mask


@dataclass(frozen=True)
class RoiBatch:
    """Named regional masks exported together from one rollout step.

  Each step performs a single GPU-to-CPU copy, then clips and writes every
  region without re-running inference.
    """

    regions: tuple[tuple[str, Mask], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.regions]
        if not names:
            raise ValueError("RoiBatch requires at least one region")
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate region names in RoiBatch: {names!r}")

    @classmethod
    def from_mapping(cls, regions: Mapping[str, Mask]) -> RoiBatch:
        return cls(regions=tuple(regions.items()))

    @classmethod
    def example_batch(cls) -> RoiBatch:
        """China, USA, Western Europe, and North Africa (0.25° axis-aligned envelopes)."""
        return cls.from_mapping(
            {
                "china": Mask.china(),
                "usa": Mask.usa(),
                "western_europe": Mask.western_europe(),
                "north_africa": Mask.north_africa(),
            }
        )
