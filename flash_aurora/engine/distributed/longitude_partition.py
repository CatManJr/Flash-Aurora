"""Longitude ranges that each spatial rank stores at every backbone level.

Patch merging pairs global columns ``(2k, 2k + 1)``. A shard boundary must
therefore fall on an even column at every level that merges, or one pair would
straddle two ranks. Equal shards do not guarantee that: on the 0.4 degree air
pollution grid the latent widths are 300, 150, and 75, and two equal shards of the
150-column level hold 75 columns each.

The partition is therefore built from the coarsest level up. The coarsest width
is split as evenly as possible; each finer level doubles every boundary. When a
finer width is odd, single-device patch merging pads one column on the east, so
the east-most rank holds the odd column and pads it locally, exactly where the
single-device grid pads.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class LongitudeShards:
    """Contiguous longitude ranges ``[boundaries[r], boundaries[r + 1])``, one per rank."""

    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.boundaries[0] != 0 or any(
            left >= right for left, right in zip(self.boundaries, self.boundaries[1:])
        ):
            raise ValueError(f"longitude boundaries {self.boundaries} must rise strictly from 0")

    @property
    def width(self) -> int:
        return self.boundaries[-1]

    @property
    def num_shards(self) -> int:
        return len(self.boundaries) - 1

    def owner(self, column: int) -> int:
        return bisect_right(self.boundaries, column) - 1

    def first_column(self, rank: int) -> int:
        return self.boundaries[rank]

    def shard_width(self, rank: int) -> int:
        return self.boundaries[rank + 1] - self.boundaries[rank]

    def columns(self, rank: int) -> slice:
        return slice(self.boundaries[rank], self.boundaries[rank + 1])

    def is_east_most(self, rank: int) -> bool:
        return rank == self.num_shards - 1


def level_widths(width: int, num_levels: int) -> tuple[int, ...]:
    """Latent widths of the backbone levels; merging pads an odd width by one column."""
    widths = [width]
    for _ in range(1, num_levels):
        widths.append((widths[-1] + widths[-1] % 2) // 2)
    return tuple(widths)


@lru_cache(maxsize=32)
def partition_levels(width: int, num_levels: int, num_shards: int) -> tuple[LongitudeShards, ...]:
    """Merge-aligned longitude shards for every backbone level, finest level first."""
    widths = level_widths(width, num_levels)
    coarsest = widths[-1]
    if coarsest < num_shards:
        raise ValueError(
            f"the coarsest level has {coarsest} longitude patches, fewer than {num_shards} spatial ranks"
        )
    boundaries = tuple(rank * coarsest // num_shards for rank in range(num_shards + 1))
    partitions = [LongitudeShards(boundaries)]
    for finer_width in reversed(widths[:-1]):
        boundaries = tuple(min(2 * b, finer_width) for b in partitions[0].boundaries)
        partitions.insert(0, LongitudeShards(boundaries))
    return tuple(partitions)
