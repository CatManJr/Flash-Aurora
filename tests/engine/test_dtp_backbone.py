from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.dtp import apply_domain_tensor_parallel
from flash_aurora.engine.distributed.dtp_backbone import (
    DomainTensorParallelBackbone,
    distributed_layer_norm,
)
from flash_aurora.engine.distributed.longitude_partition import partition_levels
from flash_aurora.engine.distributed.process_mesh import MeshShape, build_process_mesh
from gloo_spawn import run_on_gloo_ranks
from tiny_aurora import (
    LEAD_TIME,
    PATCH_RES,
    backbone_tokens,
    longitude_shard_tokens,
    tiny_aurora,
    tiny_aurora_ensemble,
    tiny_backbone,
    tiny_batch,
    tiny_stochastic_backbone,
)

# Reassociated FP32 sums differ from one device at the level of FP32 roundoff.
_RTOL = 1e-4
_ATOL = 1e-5
_MESH = MeshShape(channel=2, spatial=2)
_NOISE_SEED = 7
_NUM_LEVELS = 3


def _finest_columns(rank: int, size: int) -> slice:
    return partition_levels(PATCH_RES[2], _NUM_LEVELS, size)[0].columns(rank)


def _assert_batches_close(actual, expected) -> None:
    for group in ("surf_vars", "atmos_vars"):
        for name, field in getattr(expected, group).items():
            torch.testing.assert_close(
                getattr(actual, group)[name], field, rtol=_RTOL, atol=_ATOL, msg=f"{group}.{name}"
            )
    torch.testing.assert_close(actual.metadata.lon, expected.metadata.lon)


def _check_layer_norm_over_channel_shards() -> None:
    rank = dist.get_rank()
    torch.manual_seed(0)
    full = torch.randn(3, 5, 8) * 4.0 + 2.0
    local = full[..., rank * 4 : (rank + 1) * 4]
    normalized = distributed_layer_norm(local, full_width=8, eps=1e-5, group=dist.group.WORLD)
    expected = torch.nn.functional.layer_norm(full, (8,), eps=1e-5)[..., rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(normalized, expected, rtol=_RTOL, atol=_ATOL)


def test_distributed_layer_norm_matches_full_width(tmp_path: Path) -> None:
    run_on_gloo_ranks(2, _check_layer_norm_over_channel_shards, tmp_path)


def _check_backbone_matches_single_device() -> None:
    mesh = build_process_mesh(_MESH)
    single = tiny_backbone(use_lora=True)
    parallel = DomainTensorParallelBackbone(copy.deepcopy(single), mesh)
    tokens = backbone_tokens()
    columns = _finest_columns(mesh.coordinate.spatial, _MESH.spatial)
    with torch.inference_mode():
        expected = single(tokens, LEAD_TIME, rollout_step=0, patch_res=PATCH_RES)
        actual = parallel(
            longitude_shard_tokens(tokens, columns),
            lead_time=LEAD_TIME,
            rollout_step=0,
            patch_res=PATCH_RES,
        )
    torch.testing.assert_close(
        actual, longitude_shard_tokens(expected, columns), rtol=_RTOL, atol=_ATOL
    )


def test_dtp_backbone_matches_single_device(tmp_path: Path) -> None:
    run_on_gloo_ranks(_MESH.world_size, _check_backbone_matches_single_device, tmp_path)


def _check_stochastic_backbone_draws_the_single_device_noise(accumulation: int) -> None:
    mesh = build_process_mesh(_MESH)
    single = tiny_stochastic_backbone()
    parallel = DomainTensorParallelBackbone(copy.deepcopy(single), mesh)
    single.set_noise_accumulation(accumulation)
    parallel.set_noise_accumulation(accumulation)
    tokens = backbone_tokens()
    lead_times = torch.full((1,), 6.0)
    columns = _finest_columns(mesh.coordinate.spatial, _MESH.spatial)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        expected = [single(tokens, lead_times, 0, PATCH_RES) for _ in range(2)]
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        actual = [
            parallel(
                longitude_shard_tokens(tokens, columns),
                lead_times=lead_times,
                rollout_step=0,
                patch_res=PATCH_RES,
            )
            for _ in range(2)
        ]
    for step_actual, step_expected in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            step_actual, longitude_shard_tokens(step_expected, columns), rtol=_RTOL, atol=_ATOL
        )


def test_dtp_ensemble_backbone_is_the_same_member(tmp_path: Path) -> None:
    run_on_gloo_ranks(_MESH.world_size, _check_stochastic_backbone_draws_the_single_device_noise, tmp_path, 0)


def test_dtp_ensemble_backbone_keeps_noise_accumulation(tmp_path: Path) -> None:
    run_on_gloo_ranks(_MESH.world_size, _check_stochastic_backbone_draws_the_single_device_noise, tmp_path, 2)


def _check_model_matches_single_device() -> None:
    mesh = build_process_mesh(_MESH)
    single = tiny_aurora()
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    with torch.inference_mode():
        expected = single(tiny_batch())
        actual = parallel(tiny_batch())
    _assert_batches_close(actual, expected)


def test_dtp_model_forward_matches_single_device(tmp_path: Path) -> None:
    run_on_gloo_ranks(_MESH.world_size, _check_model_matches_single_device, tmp_path)


def _check_ensemble_model_matches_single_device() -> None:
    mesh = build_process_mesh(_MESH)
    single = tiny_aurora_ensemble()
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    lead_times = torch.full((1,), 6.0)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        expected = single(tiny_batch(), lead_times=lead_times)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        actual = parallel(tiny_batch(), lead_times=lead_times)
    _assert_batches_close(actual, expected)


def test_dtp_ensemble_model_forward_is_the_same_member(tmp_path: Path) -> None:
    run_on_gloo_ranks(_MESH.world_size, _check_ensemble_model_matches_single_device, tmp_path)
