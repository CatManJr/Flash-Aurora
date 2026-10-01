"""The DTP equivalence checks of ``test_dtp_backbone`` on NCCL, one GPU per rank."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.dtp import apply_domain_tensor_parallel
from flash_aurora.engine.distributed.dtp_backbone import DomainTensorParallelBackbone
from flash_aurora.engine.distributed.process_mesh import MeshShape, build_process_mesh
from gloo_spawn import run_on_nccl_ranks
from test_dtp_backbone import _assert_batches_close, _finest_columns
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

_RTOL = 1e-4
_ATOL = 1e-5
_MESH = MeshShape(channel=2, spatial=2)
_NOISE_SEED = 7

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < _MESH.world_size,
        reason=f"needs {_MESH.world_size} CUDA devices",
    ),
]


def _device() -> torch.device:
    return torch.device("cuda", dist.get_rank())


def _check_backbone_matches_single_device() -> None:
    device = _device()
    mesh = build_process_mesh(_MESH)
    single = tiny_backbone(use_lora=True).to(device)
    parallel = DomainTensorParallelBackbone(copy.deepcopy(single), mesh)
    tokens = backbone_tokens().to(device)
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


def _check_stochastic_backbone_draws_the_single_device_noise(accumulation: int) -> None:
    device = _device()
    mesh = build_process_mesh(_MESH)
    single = tiny_stochastic_backbone().to(device)
    parallel = DomainTensorParallelBackbone(copy.deepcopy(single), mesh)
    single.set_noise_accumulation(accumulation)
    parallel.set_noise_accumulation(accumulation)
    tokens = backbone_tokens().to(device)
    lead_times = torch.full((1,), 6.0, device=device)
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


def _check_model_matches_single_device() -> None:
    device = _device()
    mesh = build_process_mesh(_MESH)
    single = tiny_aurora().to(device)
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    with torch.inference_mode():
        expected = single(tiny_batch().to(device))
        actual = parallel(tiny_batch().to(device))
    _assert_batches_close(actual, expected)


def _check_ensemble_model_matches_single_device() -> None:
    device = _device()
    mesh = build_process_mesh(_MESH)
    single = tiny_aurora_ensemble().to(device)
    parallel = apply_domain_tensor_parallel(copy.deepcopy(single), mesh)
    lead_times = torch.full((1,), 6.0, device=device)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        expected = single(tiny_batch().to(device), lead_times=lead_times)
    torch.manual_seed(_NOISE_SEED)
    with torch.inference_mode():
        actual = parallel(tiny_batch().to(device), lead_times=lead_times)
    _assert_batches_close(actual, expected)


def test_dtp_backbone_matches_single_device_nccl(tmp_path: Path) -> None:
    run_on_nccl_ranks(_MESH.world_size, _check_backbone_matches_single_device, tmp_path)


@pytest.mark.parametrize("accumulation", [0, 2])
def test_dtp_ensemble_backbone_is_the_same_member_nccl(tmp_path: Path, accumulation: int) -> None:
    run_on_nccl_ranks(
        _MESH.world_size, _check_stochastic_backbone_draws_the_single_device_noise, tmp_path, accumulation
    )


def test_dtp_model_forward_matches_single_device_nccl(tmp_path: Path) -> None:
    run_on_nccl_ranks(_MESH.world_size, _check_model_matches_single_device, tmp_path)


def test_dtp_ensemble_model_forward_is_the_same_member_nccl(tmp_path: Path) -> None:
    run_on_nccl_ranks(_MESH.world_size, _check_ensemble_model_matches_single_device, tmp_path)
