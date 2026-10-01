from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.longitude_partition import LongitudeShards, partition_levels
from flash_aurora.engine.distributed.swin_halo import (
    gather_window_columns,
    plan_window_columns,
    scatter_window_columns,
)
from flash_aurora.models.aurora.model.swin3d import get_two_sidded_padding
from gloo_spawn import run_on_gloo_ranks

# (boundaries, window, shift): the Aurora 0.25 and 0.1 degree latent levels (360, 180,
# and 90 columns, window 12, shift 6), the odd 0.4 degree level (75 columns on merge-
# aligned shards 37 | 38), and a small unequal case.
_GRID_CASES = [
    ((0, 180, 360), 12, 0),
    ((0, 180, 360), 12, 6),
    ((0, 90, 180), 12, 6),
    ((0, 45, 90), 12, 0),
    ((0, 45, 90), 12, 6),
    ((0, 37, 75), 12, 6),
    ((0, 12, 22), 4, 2),
]


def _plan(boundaries: tuple[int, ...], window: int, shift: int):
    shards = LongitudeShards(boundaries)
    pad_left, pad_right, _, _ = get_two_sidded_padding(0, (-shards.width) % window)
    return plan_window_columns(shards, window, shift, pad_left, pad_right)


@pytest.mark.parametrize(("boundaries", "window", "shift"), _GRID_CASES)
def test_every_column_is_attended_exactly_once(boundaries, window: int, shift: int) -> None:
    plan = _plan(boundaries, window, shift)
    columns = [c for rank_columns in plan.attention_columns for c in rank_columns if c is not None]
    assert sorted(columns) == list(range(boundaries[-1]))


@pytest.mark.parametrize(("boundaries", "window", "shift"), _GRID_CASES)
def test_each_rank_holds_complete_windows(boundaries, window: int, shift: int) -> None:
    plan = _plan(boundaries, window, shift)
    for rank_columns in plan.attention_columns:
        assert len(rank_columns) % window == 0


def test_level_two_windows_straddle_shards() -> None:
    plan = _plan((0, 90, 180), 12, 0)
    owners = {plan.shards.owner(c) for c in plan.attention_columns[0] if c is not None}
    assert owners == {0, 1}


def test_air_pollution_grid_gets_merge_aligned_shards() -> None:
    levels = partition_levels(300, 3, 2)
    assert [shards.boundaries for shards in levels] == [(0, 148, 300), (0, 74, 150), (0, 37, 75)]
    assert levels == partition_levels(300, 3, 2)


def test_odd_width_level_puts_the_padding_on_the_east_most_rank() -> None:
    finest, middle, coarsest = partition_levels(22, 3, 2)
    assert middle.boundaries == (0, 6, 11)
    assert all(b % 2 == 0 for b in finest.boundaries[:-1] + middle.boundaries[:-1])
    assert coarsest.shard_width(1) == 3


def test_too_many_spatial_ranks_for_the_coarsest_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="coarsest"):
        partition_levels(8, 3, 4)


def _single_device_layout(x: torch.Tensor, window: int, shift: int) -> torch.Tensor:
    """Longitude roll and two-sided zero padding, as in ``Swin3DTransformerBlock``."""
    width = x.shape[3]
    pad_left, pad_right, _, _ = get_two_sidded_padding(0, (-width) % window)
    rolled = torch.roll(x, shifts=-shift, dims=3)
    return torch.nn.functional.pad(rolled, (0, 0, pad_left, pad_right))


def _check_round_trip(boundaries: tuple[int, ...], window: int, shift: int) -> None:
    rank = dist.get_rank()
    torch.manual_seed(0)
    full = torch.randn(1, 2, 3, boundaries[-1], 5)
    plan = _plan(boundaries, window, shift)
    stored = full[..., plan.shards.columns(rank), :].contiguous()

    gathered = gather_window_columns(stored, plan, dist.group.WORLD)
    first, end = plan.window_range(rank)
    expected = _single_device_layout(full, window, shift)[..., first * window : end * window, :]
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

    restored = scatter_window_columns(gathered, plan, dist.group.WORLD)
    torch.testing.assert_close(restored, stored, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("boundaries", "window", "shift"),
    [((0, 90, 180), 12, 6), ((0, 37, 75), 12, 6), ((0, 12, 22), 4, 2)],
)
def test_exchange_matches_single_device_layout(tmp_path: Path, boundaries, window: int, shift: int) -> None:
    run_on_gloo_ranks(2, _check_round_trip, tmp_path, boundaries, window, shift)
