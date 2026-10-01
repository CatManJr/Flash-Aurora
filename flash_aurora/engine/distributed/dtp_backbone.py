"""Domain-tensor-parallel (DTP) 3D Swin backbone for Aurora inference.

A DTP instance shards tokens over ``s`` spatial ranks along longitude and channels
over ``c`` channel ranks, after Kieckhefen et al. (arXiv:2609.12815, III-A). Each rank
stores a ``1/c x 1/s`` slice of the activations and a ``1/c`` slice of the attention,
MLP, and AdaLN modulation weights.

Per block, the linear pairs follow the sequence-parallel Megatron pattern inside a
channel group: all-gather the channel shards, run the column-parallel linear
(``qkv``, ``fc1``) on the full width, and reduce-scatter the row-parallel output
(``proj``, ``fc2``) back to channel shards in FP32. Only the reduce-scatter reorders
a sum, so the reassociated reductions match tensor parallelism.

Deviations from BEAST that Aurora forces, implemented here:

* D1. BEAST trained with grouped normalization. Aurora's checkpoints normalize over
  the full channel width, so each block norm all-reduces its per-token mean and
  variance over the channel group. Patch merging and splitting all-gather channels
  and normalize locally, which is exact.
* D2. Windows straddle longitude shards at every level; see :mod:`swin_halo`. The
  block therefore uses the PyTorch roll/pad/partition layout at every precision tier
  instead of the fused Triton layout, which needs the whole grid on one device.
  Shards are merge-aligned per level (:mod:`longitude_partition`), so the east-most
  rank pads and crops an odd width locally, where the single-device grid does.
* D3. The Aurora 1.5 ensemble adds a per-token noise field to the conditioning. The
  conditioning is then sharded on longitude like the tokens, down-sampled per level
  with the shards, and drawn from the single-device global field (:mod:`ensemble_noise`).

Both model families run: the 0.25/0.1 degree family passes ``lead_time`` as a
``timedelta``, Aurora 1.5 passes ``lead_times`` in hours. The backbone consumes and
returns spatially sharded tokens with the full channel width, so the Perceiver
encoder and decoder only see the longitude split (D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from flash_aurora.engine.distributed.collectives import (
    all_gather_last_dim,
    all_reduce_sum_fp32,
    reduce_scatter_last_dim_fp32,
)
from flash_aurora.engine.distributed.ensemble_noise import ShardedNoiseField
from flash_aurora.engine.distributed.longitude_partition import LongitudeShards, partition_levels
from flash_aurora.engine.distributed.process_mesh import ProcessMesh
from flash_aurora.engine.distributed.sharding import (
    keep_output_rows,
    shard_adaptive_modulation,
    shard_attention_heads,
    shard_mlp_hidden,
    shard_slice,
)
from flash_aurora.engine.distributed.swin_halo import (
    gather_window_columns,
    plan_window_columns,
    scatter_window_columns,
)
from flash_aurora.models.aurora.model.film import AdaptiveLayerNorm
from flash_aurora.models.aurora.model.fourier import lead_time_expansion
from flash_aurora.models.aurora.model.swin3d import (
    MLP,
    BasicLayer3D,
    PatchMerging3D,
    PatchSplitting3D,
    Swin3DTransformerBackbone,
    Swin3DTransformerBlock,
    WindowAttention,
    compute_3d_shifted_window_mask,
    crop_3d,
    get_two_sidded_padding,
    pad_3d,
    window_partition_3d,
    window_reverse_3d,
)
from flash_aurora.models.aurora.model.util import maybe_adjust_windows


@dataclass(frozen=True)
class LevelGrid:
    """One backbone level: its global resolution and this rank's longitude shard."""

    resolution: tuple[int, int, int]
    shards: LongitudeShards
    spatial_rank: int

    @property
    def local_width(self) -> int:
        return self.shards.shard_width(self.spatial_rank)

    @property
    def local_resolution(self) -> tuple[int, int, int]:
        levels, height, _ = self.resolution
        return levels, height, self.local_width

    @property
    def is_east_most(self) -> bool:
        return self.shards.is_east_most(self.spatial_rank)


def distributed_layer_norm(
    x: torch.Tensor,
    *,
    full_width: int,
    eps: float,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """LayerNorm without affine over channels sharded across ``group`` (deviation D1).

    Two passes, as in the single-device kernel: the mean first, then the biased
    variance of the centred values. Each pass all-reduces one value per token.
    """
    mean = all_reduce_sum_fp32(x.sum(dim=-1, keepdim=True), group) / full_width
    centred = x - mean
    variance = all_reduce_sum_fp32((centred * centred).sum(dim=-1, keepdim=True), group) / full_width
    return centred * torch.rsqrt(variance + eps)


class ChannelShardedAdaptiveNorm(nn.Module):
    """``AdaptiveLayerNorm`` applied to a channel shard.

    The modulation projection keeps the shift and scale rows of this rank's channels,
    so per-token conditioning ``(B, L, D_t)`` never expands to the full ``2D`` width.
    This module replaces the fused Triton AdaLN at every precision tier, because that
    kernel computes the statistics over local channels only. It keeps the single-device
    dtype contract: FP32 branch input, and the tier's output cast.
    """

    def __init__(self, norm: AdaptiveLayerNorm, channels: slice, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.full_width = norm.ln.normalized_shape[0]
        shard_adaptive_modulation(norm.ln_modulation[-1], channels)
        self.norm = norm
        self.group = group

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = self.norm.ln_modulation(c)
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        shift, scale = modulation.chunk(2, dim=-1)
        normalized = distributed_layer_norm(
            self.norm._adaln_branch_input(x),
            full_width=self.full_width,
            eps=self.norm.ln.eps,
            group=self.group,
        )
        return self.norm._finalize_adaln_output(normalized * (self.norm.scale_bias + scale) + shift)


class ChannelShardedWindowAttention(nn.Module):
    """Attention over local heads; all-gather in, reduce-scatter out (both on channels)."""

    def __init__(self, attention: WindowAttention, channels: slice, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group
        full_bias = shard_attention_heads(
            attention, dist.get_rank(group), dist.get_world_size(group)
        )
        self.output_bias = nn.Buffer(full_bias[channels].clone())
        self.attention = attention

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor | None,
        rollout_step: int,
    ) -> torch.Tensor:
        full_width = all_gather_last_dim(windows, self.group)
        partial = self.attention(full_width, mask=mask, rollout_step=rollout_step)
        return reduce_scatter_last_dim_fp32(partial, self.group) + self.output_bias


class ChannelShardedMLP(nn.Module):
    """MLP over local hidden units; all-gather in, reduce-scatter out (both on channels)."""

    def __init__(self, mlp: MLP, channels: slice, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group
        full_bias = shard_mlp_hidden(mlp, dist.get_rank(group), dist.get_world_size(group))
        self.output_bias = nn.Buffer(full_bias[channels].clone())
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = self.mlp(all_gather_last_dim(x, self.group))
        return reduce_scatter_last_dim_fp32(partial, self.group) + self.output_bias


class DomainParallelSwinBlock(nn.Module):
    """One 3D Swin block on a ``1/c x 1/s`` activation shard."""

    def __init__(self, block: Swin3DTransformerBlock, mesh: ProcessMesh) -> None:
        super().__init__()
        channels = shard_slice(block.dim, mesh.coordinate.channel, mesh.shape.channel)
        self.window_size = block.window_size
        self.shift_size = block.shift_size
        self.spatial_group = mesh.spatial_group
        self.attn = ChannelShardedWindowAttention(block.attn, channels, mesh.channel_group)
        self.mlp = ChannelShardedMLP(block.mlp, channels, mesh.channel_group)
        self.norm1 = ChannelShardedAdaptiveNorm(block.norm1, channels, mesh.channel_group)
        self.norm2 = ChannelShardedAdaptiveNorm(block.norm2, channels, mesh.channel_group)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        grid: LevelGrid,
        rollout_step: int,
    ) -> torch.Tensor:
        """``x`` is ``(B, C * H * W_r, D_c)``; ``c`` is ``(B, D_t)`` or ``(B, C * H * W_r, D_t)``."""
        x = x + self.norm1(self._attend(x, grid, rollout_step), c)
        return x + self.norm2(self.mlp(x), c)

    def _attend(self, x: torch.Tensor, grid: LevelGrid, rollout_step: int) -> torch.Tensor:
        C, H, W = grid.resolution
        B, L, D = x.shape
        ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, grid.resolution)
        pad_c, pad_h, pad_w = (-C) % ws[0], (-H) % ws[1], (-W) % ws[2]
        pad_left, pad_right, _, _ = get_two_sidded_padding(0, pad_w)
        plan = plan_window_columns(grid.shards, ws[2], ss[2], pad_left, pad_right)

        tokens = x.view(B, C, H, grid.local_width, D)
        tokens = gather_window_columns(tokens, plan, self.spatial_group)
        tokens = torch.roll(tokens, shifts=(-ss[0], -ss[1]), dims=(1, 2))
        tokens = pad_3d(tokens, (pad_c, pad_h, 0))
        _, padded_c, padded_h, attention_w, _ = tokens.shape

        windows = window_partition_3d(tokens, ws).reshape(-1, ws[0] * ws[1] * ws[2], D)
        mask = self._local_mask(grid.resolution, ws, ss, plan.window_range(grid.spatial_rank), x)
        attended = self.attn(windows, mask, rollout_step)

        attended = attended.view(-1, ws[0], ws[1], ws[2], D)
        tokens = window_reverse_3d(attended, ws, padded_c, padded_h, attention_w)
        tokens = crop_3d(tokens, (pad_c, pad_h, 0))
        tokens = torch.roll(tokens, shifts=(ss[0], ss[1]), dims=(1, 2))
        tokens = scatter_window_columns(tokens, plan, self.spatial_group)
        return tokens.reshape(B, L, D)

    @staticmethod
    def _local_mask(
        res: tuple[int, int, int],
        ws: tuple[int, int, int],
        ss: tuple[int, int, int],
        window_range: tuple[int, int],
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        """Rows of the global shifted-window mask for this rank's longitude windows."""
        if all(s == 0 for s in ss):
            return None
        C, H, W = res
        mask, _ = compute_3d_shifted_window_mask(C, H, W, ws, ss, like.device, like.dtype, warped=True)
        windows_c = -(-C // ws[0])
        windows_h = -(-H // ws[1])
        tokens_per_window = ws[0] * ws[1] * ws[2]
        mask = mask.view(windows_c, windows_h, -1, tokens_per_window, tokens_per_window)
        first, end = window_range
        return mask[:, :, first:end].reshape(-1, tokens_per_window, tokens_per_window)


class DomainParallelPatchMerging(nn.Module):
    """Patch merging: all-gather channels, merge and normalize locally, column-parallel reduction."""

    def __init__(self, merging: PatchMerging3D, mesh: ProcessMesh) -> None:
        super().__init__()
        keep_output_rows(merging.reduction, mesh.coordinate.channel, mesh.shape.channel)
        self.merging = merging
        self.channel_group = mesh.channel_group

    def forward(self, x: torch.Tensor, grid: LevelGrid) -> torch.Tensor:
        # Merge-aligned shards: only the east-most rank can hold an odd width, and the
        # local merge pads it on the east, as the single-device merge does.
        x = all_gather_last_dim(x, self.channel_group)
        return self.merging(x, grid.local_resolution)


class DomainParallelPatchSplitting(nn.Module):
    """Patch splitting: all-gather channels, split and normalize locally, column-parallel ``lin2``."""

    def __init__(self, splitting: PatchSplitting3D, mesh: ProcessMesh) -> None:
        super().__init__()
        keep_output_rows(splitting.lin2, mesh.coordinate.channel, mesh.shape.channel)
        self.splitting = splitting
        self.channel_group = mesh.channel_group

    def forward(self, x: torch.Tensor, grid: LevelGrid, crop: tuple[int, int, int]) -> torch.Tensor:
        # Undoing the merge padding crops the east edge, which only the east-most rank holds.
        crop_c, crop_h, crop_w = crop
        local_crop = (crop_c, crop_h, crop_w if grid.is_east_most else 0)
        x = all_gather_last_dim(x, self.channel_group)
        return self.splitting(x, grid.local_resolution, local_crop)


class DomainParallelLayer(nn.Module):
    """One backbone stage: its blocks, then optional patch merging or splitting."""

    def __init__(self, layer: BasicLayer3D, mesh: ProcessMesh) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(DomainParallelSwinBlock(block, mesh) for block in layer.blocks)
        self.downsample = (
            DomainParallelPatchMerging(layer.downsample, mesh) if layer.downsample is not None else None
        )
        self.upsample = (
            DomainParallelPatchSplitting(layer.upsample, mesh) if layer.upsample is not None else None
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        grid: LevelGrid,
        crop: tuple[int, int, int] = (0, 0, 0),
        rollout_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.blocks:
            x = block(x, c, grid, rollout_step)
        if self.downsample is not None:
            return self.downsample(x, grid), x
        if self.upsample is not None:
            return self.upsample(x, grid, crop), x
        return x, None


def _hours_tensor_expansion(backbone: Swin3DTransformerBackbone):
    """Lead-time expansion of the Aurora 1.5 backbone, which takes an hours tensor."""
    from flash_aurora.models.aurora_v1p5.model.fourier import (
        lead_time_expansion as v1p5_lead_time_expansion,
        lead_time_expansion_v3,
    )

    if backbone.use_updated_lead_time_embedding:
        return lead_time_expansion_v3
    return v1p5_lead_time_expansion


class DomainTensorParallelBackbone(nn.Module):
    """Swin3D backbone on spatially sharded, full-channel tokens in and out.

    Mirrors ``Swin3DTransformerBackbone.forward`` of either model family level by
    level; only the token layout differs.
    """

    def __init__(self, backbone: Swin3DTransformerBackbone, mesh: ProcessMesh) -> None:
        super().__init__()
        self.window_size = backbone.window_size
        self.embed_dim = backbone.embed_dim
        self.num_encoder_layers = backbone.num_encoder_layers
        self.num_decoder_layers = backbone.num_decoder_layers
        self.time_mlp = backbone.time_mlp
        self.hours_expansion = (
            _hours_tensor_expansion(backbone)
            if hasattr(backbone, "use_updated_lead_time_embedding")
            else None
        )
        self.stochastic = bool(getattr(backbone, "stochastic", False))
        if self.stochastic:
            self.noise_mlp = backbone.noise_mlp
            self.context_down_layers = backbone.context_down_layers
            self.noise = ShardedNoiseField()
        self.encoder_layers = nn.ModuleList(
            DomainParallelLayer(layer, mesh) for layer in backbone.encoder_layers
        )
        self.decoder_layers = nn.ModuleList(
            DomainParallelLayer(layer, mesh) for layer in backbone.decoder_layers
        )
        self.channel_group = mesh.channel_group
        self.channel_rank = mesh.coordinate.channel
        self.num_channel = mesh.shape.channel
        self.spatial_rank = mesh.coordinate.spatial
        self.num_spatial = mesh.shape.spatial

    def forward(
        self,
        x: torch.Tensor,
        *,
        rollout_step: int,
        patch_res: tuple[int, int, int],
        lead_time: timedelta | None = None,
        lead_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` is ``(B, C * H * W_r, D)``; ``patch_res`` is the global ``(C, H, W)``.

        Pass ``lead_time`` for the 0.25/0.1 family or ``lead_times`` for Aurora 1.5.
        """
        grids = self.level_grids(patch_res)
        _, padded_outs = self.encoder_specs(patch_res)
        c = self._lead_time_context(x, lead_time=lead_time, lead_times=lead_times)
        if self.stochastic:
            c = c.unsqueeze(1) + self.noise_mlp(self._draw_noise(x, grids[0]))
        x = x[..., shard_slice(x.shape[-1], self.channel_rank, self.num_channel)].contiguous()

        skips = []
        saved_cs = []
        for i, layer in enumerate(self.encoder_layers):
            saved_cs.append(c)
            x, x_unscaled = layer(x, c, grids[i], rollout_step=rollout_step)
            if self.stochastic and i < self.num_encoder_layers - 1:
                c = self.context_down_layers[i](c, grids[i].local_resolution)
            skips.append(x_unscaled)
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(x, c, grids[index], padded_outs[index - 1], rollout_step=rollout_step)
            if 0 < i < self.num_decoder_layers - 1:
                x = x + skips[index - 1]
            if self.stochastic:
                c = saved_cs[index - 1]
        # The last stage concatenates the first skip, like Pangu; gather each half separately
        # so the channel order is [x, skip] as on one device.
        return torch.cat(
            [
                all_gather_last_dim(x, self.channel_group),
                all_gather_last_dim(skips[0], self.channel_group),
            ],
            dim=-1,
        )

    def reset_noise(self) -> None:
        if self.stochastic:
            self.noise.reset()

    def set_noise_accumulation(self, n: int = 0) -> None:
        if self.stochastic:
            self.noise.set_accumulation(n)

    def _lead_time_context(
        self,
        x: torch.Tensor,
        *,
        lead_time: timedelta | None,
        lead_times: torch.Tensor | None,
    ) -> torch.Tensor:
        if lead_times is not None:
            if self.hours_expansion is None:
                raise ValueError("the 0.25/0.1 degree backbone takes `lead_time`, not `lead_times`")
            return self.time_mlp(self.hours_expansion(lead_times, self.embed_dim).to(dtype=x.dtype))
        if lead_time is None:
            raise ValueError("pass `lead_time` (0.25/0.1 family) or `lead_times` (Aurora 1.5)")
        lead_hours = lead_time / timedelta(hours=1)
        weight_dtype = next(self.time_mlp.parameters()).dtype
        hours = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        return self.time_mlp(lead_time_expansion(hours, self.embed_dim).to(dtype=weight_dtype))

    def _draw_noise(self, x: torch.Tensor, grid: LevelGrid) -> torch.Tensor:
        return self.noise.draw(
            batch_size=x.shape[0],
            resolution=grid.resolution,
            embed_dim=self.embed_dim,
            columns=grid.shards.columns(self.spatial_rank),
            device=x.device,
            dtype=x.dtype,
        )

    def level_grids(self, patch_res: tuple[int, int, int]) -> tuple[LevelGrid, ...]:
        """Global resolution and this rank's merge-aligned shard at every level."""
        if patch_res[0] % self.window_size[0] != 0:
            raise ValueError(f"levels {patch_res[0]} must be divisible by {self.window_size[0]}")
        all_res, _ = self.encoder_specs(patch_res)
        shards = partition_levels(patch_res[2], self.num_encoder_layers, self.num_spatial)
        return tuple(
            LevelGrid(resolution=res, shards=level_shards, spatial_rank=self.spatial_rank)
            for res, level_shards in zip(all_res, shards, strict=True)
        )

    def encoder_specs(
        self, patch_res: tuple[int, int, int]
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """Global resolution and merge padding of each level, as on one device."""
        return Swin3DTransformerBackbone.get_encoder_specs(self, patch_res)  # reads num_encoder_layers
