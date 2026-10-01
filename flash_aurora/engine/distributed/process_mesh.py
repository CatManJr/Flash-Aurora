"""Process mesh for BEAST-style 4D parallel inference.

Kieckhefen et al. (arXiv:2609.12815) arrange GPUs on four orthogonal axes: channel
ranks ``c`` and spatial ranks ``s`` form one domain-tensor-parallel (DTP) instance,
uncertainty ranks ``u`` hold independent ensemble members, and data ranks ``d`` hold
independent forecasts. The total ``c * s * u * d`` is our derived GPU count; the
paper states the factors but does not display the product.

Rank order puts the channel axis innermost, because channel ranks exchange the most
data per block (an all-gather and a reduce-scatter per linear pair).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import product

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MeshShape:
    """Number of ranks along each of the four parallel axes."""

    channel: int = 1
    spatial: int = 1
    uncertainty: int = 1
    data: int = 1

    def __post_init__(self) -> None:
        for axis, size in self.axis_sizes().items():
            if size < 1:
                raise ValueError(f"mesh axis {axis!r} must be >= 1, got {size}")

    def axis_sizes(self) -> dict[str, int]:
        return {
            "channel": self.channel,
            "spatial": self.spatial,
            "uncertainty": self.uncertainty,
            "data": self.data,
        }

    @property
    def world_size(self) -> int:
        return self.channel * self.spatial * self.uncertainty * self.data


@dataclass(frozen=True)
class MeshCoordinate:
    """Position of one rank on the four axes."""

    channel: int
    spatial: int
    uncertainty: int
    data: int


def rank_of(shape: MeshShape, coordinate: MeshCoordinate) -> int:
    return coordinate.channel + shape.channel * (
        coordinate.spatial
        + shape.spatial * (coordinate.uncertainty + shape.uncertainty * coordinate.data)
    )


def coordinate_of(shape: MeshShape, rank: int) -> MeshCoordinate:
    channel = rank % shape.channel
    rest = rank // shape.channel
    spatial = rest % shape.spatial
    rest //= shape.spatial
    uncertainty = rest % shape.uncertainty
    data = rest // shape.uncertainty
    return MeshCoordinate(channel=channel, spatial=spatial, uncertainty=uncertainty, data=data)


@dataclass(frozen=True)
class ProcessMesh:
    """This rank's coordinate and the process groups along each axis."""

    shape: MeshShape
    coordinate: MeshCoordinate
    channel_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    uncertainty_group: dist.ProcessGroup
    data_group: dist.ProcessGroup


def _axis_rank_lists(shape: MeshShape, axis: str) -> list[list[int]]:
    """All rank lists that vary along ``axis`` with the other three coordinates fixed."""
    sizes = shape.axis_sizes()
    fixed_axes = [name for name in sizes if name != axis]
    groups = []
    for fixed in product(*(range(sizes[name]) for name in fixed_axes)):
        coordinate = dict(zip(fixed_axes, fixed, strict=True))
        groups.append(
            [
                rank_of(shape, MeshCoordinate(**{**coordinate, axis: index}))
                for index in range(sizes[axis])
            ]
        )
    return groups


def _group_containing(shape: MeshShape, axis: str, rank: int) -> dist.ProcessGroup:
    """Create every group along ``axis`` (all ranks must call) and return this rank's."""
    own_group = None
    for ranks in _axis_rank_lists(shape, axis):
        group = dist.new_group(ranks)
        if rank in ranks:
            own_group = group
    assert own_group is not None, f"rank {rank} is missing from every {axis} group"
    return own_group


def build_process_mesh(shape: MeshShape) -> ProcessMesh:
    """Build the four axis groups. Collective: every rank of the world must call this."""
    world_size = dist.get_world_size()
    if shape.world_size != world_size:
        raise ValueError(
            f"mesh {shape} needs {shape.world_size} ranks, but the world has {world_size}"
        )
    rank = dist.get_rank()
    return ProcessMesh(
        shape=shape,
        coordinate=coordinate_of(shape, rank),
        channel_group=_group_containing(shape, "channel", rank),
        spatial_group=_group_containing(shape, "spatial", rank),
        uncertainty_group=_group_containing(shape, "uncertainty", rank),
        data_group=_group_containing(shape, "data", rank),
    )


def init_distributed_from_env() -> torch.device:
    """Join the process group described by torchrun's environment; return this rank's device.

    NCCL on CUDA, gloo otherwise (CPU equivalence tests).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device
