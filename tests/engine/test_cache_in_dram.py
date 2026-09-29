"""DRAM caching of rollout predictions is opt-in."""

from __future__ import annotations

from dataclasses import replace

import torch

from flash_aurora.engine.core.engine import collect_rollout_batches
from flash_aurora.engine.core.presets import DEFAULT_PRESETS


def test_cache_in_dram_defaults_on() -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    assert config.cache_in_dram is True
    assert replace(config, cache_in_dram=False).cache_in_dram is False


def test_collect_rollout_batches_keeps_history_only_when_cached() -> None:
    steps = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    cached = collect_rollout_batches(iter(steps), cache_in_dram=True)
    assert len(cached) == 3
    assert all(torch.equal(got, ref) for got, ref in zip(cached, steps))
    kept = collect_rollout_batches(iter(steps), cache_in_dram=False)
    assert len(kept) == 1
    assert torch.equal(kept[0], steps[-1])


def test_collect_rollout_batches_empty_stream() -> None:
    assert collect_rollout_batches(iter(()), cache_in_dram=False) == []
