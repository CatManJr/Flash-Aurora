from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.ensemble_moments import (
    all_reduce_moments,
    local_moments,
    merge_moments,
)
from gloo_spawn import run_on_gloo_ranks

# Geopotential-like members: mean near 5e4, spread of a few tens. Q/m - mu^2 in FP32
# loses the spread entirely at this magnitude; the moment merge must not.
_GEOPOTENTIAL_MEAN = 5.0e4
_GEOPOTENTIAL_SPREAD = 30.0
_NUM_MEMBERS = 8


def _members(seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (_GEOPOTENTIAL_MEAN + _GEOPOTENTIAL_SPREAD * torch.randn(16, generator=generator)).float()
        for _ in range(_NUM_MEMBERS)
    ]


def _reference(members: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(members).double()
    return stacked.mean(dim=0), stacked.std(dim=0, correction=1)


def test_local_moments_match_two_pass_statistics() -> None:
    members = _members()
    moments = local_moments(members)
    mean, spread = _reference(members)
    torch.testing.assert_close(moments.mean, mean)
    torch.testing.assert_close(moments.spread(ddof=1), spread)


def test_merging_split_member_sets_equals_one_set() -> None:
    members = _members()
    merged = merge_moments(local_moments(members[:3]), local_moments(members[3:]))
    mean, spread = _reference(members)
    assert merged.count == _NUM_MEMBERS
    torch.testing.assert_close(merged.mean, mean)
    torch.testing.assert_close(merged.spread(ddof=1), spread)


def test_spread_needs_more_members_than_ddof() -> None:
    with pytest.raises(ValueError):
        local_moments(_members()[:1]).spread(ddof=1)


def _check_uncertainty_ranks_reduce_to_the_full_ensemble() -> None:
    rank, size = dist.get_rank(), dist.get_world_size()
    members = _members()
    per_rank = _NUM_MEMBERS // size
    merged = all_reduce_moments(local_moments(members[rank * per_rank : (rank + 1) * per_rank]), dist.group.WORLD)
    mean, spread = _reference(members)
    torch.testing.assert_close(merged.mean, mean)
    torch.testing.assert_close(merged.spread(ddof=1), spread)


def test_uncertainty_ranks_reduce_to_the_full_ensemble(tmp_path: Path) -> None:
    run_on_gloo_ranks(2, _check_uncertainty_ranks_reduce_to_the_full_ensemble, tmp_path)
