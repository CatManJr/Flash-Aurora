"""In-place weight sharding of 3D Swin attention and MLP layers.

Both tensor parallelism and the channel axis of domain-tensor parallelism split a
block the Megatron way: the input-side linear (``qkv``, ``fc1``) keeps a slice of its
output rows, and the output-side linear (``proj``, ``fc2``) keeps the matching slice
of its input columns. The output-side linear then produces a partial sum over the
local slice. Its bias is detached and returned, because the bias must be added once,
after the partial sums are reduced across ranks.

Attention is split by heads. Head ``h`` owns channels ``[h * d_h, (h + 1) * d_h)``,
so a contiguous head range is also a contiguous channel range.
"""

from __future__ import annotations

import torch
from torch import nn

from flash_aurora.models.aurora.model.swin3d import MLP, WindowAttention


def shard_slice(total: int, rank: int, size: int) -> slice:
    """Contiguous slice ``rank`` of ``size`` equal parts of ``range(total)``."""
    if total % size != 0:
        raise ValueError(f"{total} does not split into {size} equal shards")
    width = total // size
    return slice(rank * width, (rank + 1) * width)


def _qkv_rows_for_channels(dim: int, channels: slice) -> torch.Tensor:
    """Rows of the packed ``(qkv H D)`` projection that produce ``channels`` of q, k, and v."""
    local = torch.arange(channels.start, channels.stop)
    return torch.cat([local, local + dim, local + 2 * dim])


def _keep_rows(linear: nn.Linear, rows: torch.Tensor | slice) -> None:
    linear.weight = nn.Parameter(linear.weight.detach()[rows].contiguous(), requires_grad=False)
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.detach()[rows].contiguous(), requires_grad=False)
    linear.out_features = linear.weight.shape[0]


def _keep_columns(linear: nn.Linear, columns: slice) -> torch.Tensor | None:
    """Keep input ``columns``; detach and return the full bias."""
    linear.weight = nn.Parameter(
        linear.weight.detach()[:, columns].contiguous(), requires_grad=False
    )
    linear.in_features = linear.weight.shape[1]
    bias = None if linear.bias is None else linear.bias.detach().clone()
    linear.bias = None
    return bias


def _lora_layers(lora: object) -> list[nn.Module]:
    """LoRA factors of a ``LoRARollout`` (either model family), or none when LoRA is off."""
    if isinstance(lora, nn.Module) and hasattr(lora, "loras"):
        return list(lora.loras)
    return []


def _keep_lora_output_rows(lora: object, rows: torch.Tensor) -> None:
    for layer in _lora_layers(lora):
        layer.lora_B = nn.Parameter(layer.lora_B.detach()[rows].contiguous(), requires_grad=False)


def _keep_lora_input_columns(lora: object, columns: slice) -> None:
    for layer in _lora_layers(lora):
        layer.lora_A = nn.Parameter(
            layer.lora_A.detach()[:, columns].contiguous(), requires_grad=False
        )


def shard_attention_heads(attention: WindowAttention, rank: int, size: int) -> torch.Tensor:
    """Keep heads ``rank`` of ``size`` in place; return the full ``proj`` bias.

    After sharding, ``attention(x)`` returns this rank's partial sum of the output
    projection, without bias. LoRA factors are cut with the same index, so both the
    merged and the unmerged LoRA paths stay consistent with the sliced base weights.
    """
    heads = shard_slice(attention.num_heads, rank, size)
    channels = slice(heads.start * attention.head_dim, heads.stop * attention.head_dim)
    qkv_rows = _qkv_rows_for_channels(attention.dim, channels)

    _keep_rows(attention.qkv, qkv_rows)
    _keep_lora_output_rows(attention.lora_qkv, qkv_rows)
    bias = _keep_columns(attention.proj, channels)
    _keep_lora_input_columns(attention.lora_proj, channels)
    attention.num_heads = heads.stop - heads.start
    attention._merged_linear_cache.clear()
    if bias is None:
        bias = torch.zeros(attention.proj.out_features, dtype=attention.proj.weight.dtype)
    return bias


def shard_mlp_hidden(mlp: MLP, rank: int, size: int) -> torch.Tensor:
    """Keep hidden units ``rank`` of ``size`` in place; return the full ``fc2`` bias."""
    hidden = shard_slice(mlp.fc1.out_features, rank, size)
    _keep_rows(mlp.fc1, hidden)
    bias = _keep_columns(mlp.fc2, hidden)
    if bias is None:
        bias = torch.zeros(mlp.fc2.out_features, dtype=mlp.fc2.weight.dtype)
    return bias


def shard_adaptive_modulation(modulation: nn.Linear, channels: slice) -> None:
    """Keep the shift and scale rows of ``channels`` in an AdaLN ``(shift | scale)`` projection."""
    width = modulation.out_features // 2
    local = torch.arange(channels.start, channels.stop)
    _keep_rows(modulation, torch.cat([local, local + width]))


def keep_output_rows(linear: nn.Linear, rank: int, size: int) -> None:
    """Column-parallel linear: keep output rows ``rank`` of ``size`` in place."""
    _keep_rows(linear, shard_slice(linear.out_features, rank, size))
