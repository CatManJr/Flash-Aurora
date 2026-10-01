from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.sharding import shard_attention_heads
from flash_aurora.engine.distributed.tensor_parallel import RowParallelMLP, RowParallelWindowAttention
from gloo_spawn import run_on_gloo_ranks
from tiny_aurora import LEAD_TIME, PATCH_RES, backbone_tokens, tiny_backbone

# Reassociated FP32 sums differ from one device at the level of FP32 roundoff.
_RTOL = 1e-4
_ATOL = 1e-5


def _shard_backbone(backbone: torch.nn.Module) -> None:
    for layer in (*backbone.encoder_layers, *backbone.decoder_layers):
        for block in layer.blocks:
            block.attn = RowParallelWindowAttention(block.attn, dist.group.WORLD)
            block.mlp = RowParallelMLP(block.mlp, dist.group.WORLD)


def _check_backbone_matches_single_device(use_lora: bool) -> None:
    single = tiny_backbone(use_lora=use_lora)
    parallel = copy.deepcopy(single)
    _shard_backbone(parallel)
    tokens = backbone_tokens()
    with torch.inference_mode():
        expected = single(tokens, LEAD_TIME, rollout_step=0, patch_res=PATCH_RES)
        actual = parallel(tokens, LEAD_TIME, rollout_step=0, patch_res=PATCH_RES)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("use_lora", [False, True])
def test_tensor_parallel_backbone_matches_single_device(tmp_path: Path, use_lora: bool) -> None:
    run_on_gloo_ranks(2, _check_backbone_matches_single_device, tmp_path, use_lora)


def test_head_sharding_keeps_matching_qkv_rows_and_proj_columns() -> None:
    attention = tiny_backbone(use_lora=True).encoder_layers[0].blocks[0].attn
    full_qkv = attention.qkv.weight.detach().clone()
    full_proj = attention.proj.weight.detach().clone()
    dim, head_dim = attention.dim, attention.head_dim

    bias = shard_attention_heads(attention, rank=1, size=2)

    local = slice(dim // 2, dim)
    assert attention.num_heads == dim // head_dim // 2
    torch.testing.assert_close(attention.qkv.weight[: dim // 2], full_qkv[local])
    torch.testing.assert_close(attention.qkv.weight[dim // 2 : dim], full_qkv[dim:][local])
    torch.testing.assert_close(attention.proj.weight, full_proj[:, local])
    assert attention.proj.bias is None and bias.shape == (dim,)
    assert attention.lora_qkv.loras[0].lora_B.shape[0] == 3 * dim // 2
    assert attention.lora_proj.loras[0].lora_A.shape[1] == dim // 2
