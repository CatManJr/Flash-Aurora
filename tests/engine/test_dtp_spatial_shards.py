"""DTP equivalence with three and four spatial ranks and no channel split.

``tiny_aurora``'s default grid has only two windows at the coarsest level, which cannot
host more than two spatial ranks. This grid is wide enough for four: latent widths
``50, 25, 13`` pad to ``52, 28, 16``, giving ``13, 7, 4`` windows of width four.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.dtp import apply_domain_tensor_parallel
from flash_aurora.engine.distributed.process_mesh import MeshShape, build_process_mesh
from gloo_spawn import run_on_gloo_ranks
from test_dtp_backbone import _assert_batches_close
from tiny_aurora import tiny_aurora, tiny_aurora_ensemble, tiny_batch

_WIDE_LATENT_WIDTH = 50
_NOISE_SEED = 7
_ENSEMBLE_LEAD_HOURS = 6.0


def _spatial_only_mesh() -> MeshShape:
    return MeshShape(channel=1, spatial=dist.get_world_size())


def _check_model_matches_single_device() -> None:
    mesh = build_process_mesh(_spatial_only_mesh())
    single = tiny_aurora()
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    batch = tiny_batch(latent_width=_WIDE_LATENT_WIDTH)
    with torch.inference_mode():
        expected = single(batch)
        actual = parallel(tiny_batch(latent_width=_WIDE_LATENT_WIDTH))
    _assert_batches_close(actual, expected)


def _check_ensemble_model_matches_single_device() -> None:
    mesh = build_process_mesh(_spatial_only_mesh())
    single = tiny_aurora_ensemble()
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    lead_times = torch.full((1,), _ENSEMBLE_LEAD_HOURS)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        expected = single(tiny_batch(latent_width=_WIDE_LATENT_WIDTH), lead_times=lead_times)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        actual = parallel(tiny_batch(latent_width=_WIDE_LATENT_WIDTH), lead_times=lead_times)
    _assert_batches_close(actual, expected)


@pytest.mark.parametrize("num_spatial_ranks", [3, 4])
def test_spatial_only_model_forward_matches_single_device(tmp_path: Path, num_spatial_ranks: int) -> None:
    run_on_gloo_ranks(num_spatial_ranks, _check_model_matches_single_device, tmp_path)


@pytest.mark.parametrize("num_spatial_ranks", [3, 4])
def test_spatial_only_ensemble_forward_is_the_same_member(tmp_path: Path, num_spatial_ranks: int) -> None:
    run_on_gloo_ranks(num_spatial_ranks, _check_ensemble_model_matches_single_device, tmp_path)
