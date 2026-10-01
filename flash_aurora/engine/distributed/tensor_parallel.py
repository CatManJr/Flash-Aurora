"""Megatron-style tensor parallelism for the Aurora 3D Swin backbone.

Each block is split over ``t`` ranks: column-parallel ``qkv`` and ``fc1``, row-parallel
``proj`` and ``fc2``. Activations stay replicated between blocks, so every
``AdaptiveLayerNorm`` sees the full channel width and needs no communication. One FP32
all-reduce follows each row-parallel layer, two per block. That all-reduce splits one
``K``-term dot product into ``t`` partial sums; it is the only reassociation this
scheme introduces.

Window partition, the shifted-window roll, and the cyclic longitude wrap are untouched,
because every rank holds the whole grid. The Perceiver encoder and decoder, the patch
merging and splitting layers, and the time embedding stay replicated on every rank, so
all ranks finish each step with the same prediction and roll out in lockstep.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from flash_aurora.engine.distributed.collectives import all_reduce_sum_fp32
from flash_aurora.engine.distributed.sharding import shard_attention_heads, shard_mlp_hidden
from flash_aurora.models.aurora.model.swin3d import (
    MLP,
    Swin3DTransformerBackbone,
    Swin3DTransformerBlock,
    WindowAttention,
)

_TENSOR_PARALLEL_SIZE_ATTR = "_flash_aurora_tensor_parallel_size"


class RowParallelWindowAttention(nn.Module):
    """Window attention over local heads, all-reduced into the full output projection."""

    def __init__(self, attention: WindowAttention, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group
        self.output_bias = nn.Buffer(
            shard_attention_heads(attention, dist.get_rank(group), dist.get_world_size(group))
        )
        self.attention = attention

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rollout_step: int = 0,
    ) -> torch.Tensor:
        partial = self.attention(x, mask=mask, rollout_step=rollout_step)
        return all_reduce_sum_fp32(partial, self.group) + self.output_bias


class RowParallelMLP(nn.Module):
    """MLP over local hidden units, all-reduced into the full ``fc2`` output."""

    def __init__(self, mlp: MLP, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group
        self.output_bias = nn.Buffer(
            shard_mlp_hidden(mlp, dist.get_rank(group), dist.get_world_size(group))
        )
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return all_reduce_sum_fp32(self.mlp(x), self.group) + self.output_bias


def iter_swin_blocks(backbone: Swin3DTransformerBackbone) -> Iterator[Swin3DTransformerBlock]:
    for layer in (*backbone.encoder_layers, *backbone.decoder_layers):
        yield from layer.blocks


def apply_tensor_parallel(model: Any, group: dist.ProcessGroup) -> Any:
    """Shard every backbone block of ``model`` over ``group`` in place.

    Call after the checkpoint is loaded and before the model is moved to its device.
    """
    if is_tensor_parallel(model):
        raise RuntimeError("model is already tensor-parallel")
    for block in iter_swin_blocks(model.backbone):
        block.attn = RowParallelWindowAttention(block.attn, group)
        block.mlp = RowParallelMLP(block.mlp, group)
    setattr(model, _TENSOR_PARALLEL_SIZE_ATTR, dist.get_world_size(group))
    return model


def is_tensor_parallel(model: Any) -> bool:
    return getattr(model, _TENSOR_PARALLEL_SIZE_ATTR, None) is not None
