from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.process_mesh import (
    MeshShape,
    build_process_mesh,
    coordinate_of,
    rank_of,
)
from gloo_spawn import run_on_gloo_ranks

_SHAPE = MeshShape(channel=2, spatial=2, uncertainty=2, data=1)


def test_rank_and_coordinate_are_inverse() -> None:
    shape = MeshShape(channel=2, spatial=3, uncertainty=2, data=2)
    ranks = [rank_of(shape, coordinate_of(shape, rank)) for rank in range(shape.world_size)]
    assert ranks == list(range(shape.world_size))


def test_channel_axis_is_innermost() -> None:
    assert [coordinate_of(_SHAPE, rank).channel for rank in range(4)] == [0, 1, 0, 1]
    assert [coordinate_of(_SHAPE, rank).spatial for rank in range(4)] == [0, 0, 1, 1]


def test_mesh_axis_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MeshShape(channel=0)


def _check_axis_groups() -> None:
    mesh = build_process_mesh(_SHAPE)
    rank = dist.get_rank()
    for group, axis in (
        (mesh.channel_group, "channel"),
        (mesh.spatial_group, "spatial"),
        (mesh.uncertainty_group, "uncertainty"),
    ):
        coordinates = [None] * dist.get_world_size(group)
        dist.all_gather_object(coordinates, coordinate_of(_SHAPE, rank), group=group)
        own = coordinate_of(_SHAPE, rank)
        for other in coordinates:
            for fixed in ("channel", "spatial", "uncertainty", "data"):
                if fixed != axis:
                    assert getattr(other, fixed) == getattr(own, fixed), (axis, fixed)
        assert sorted(getattr(c, axis) for c in coordinates) == list(range(_SHAPE.axis_sizes()[axis]))
    total = torch.ones(1)
    dist.all_reduce(total, group=mesh.data_group)
    assert total.item() == _SHAPE.data


def test_axis_groups_vary_one_coordinate(tmp_path: Path) -> None:
    run_on_gloo_ranks(_SHAPE.world_size, _check_axis_groups, tmp_path)
