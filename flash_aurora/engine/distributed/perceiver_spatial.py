"""Longitude-sharded Perceiver encoder and decoder for 4D inference (deviation D5).

BEAST has no Perceiver. Aurora's encoder and decoder attend across pressure levels
at each patch and never across neighbouring patches, and the patch embedding uses
non-overlapping patches. A cut on patch boundaries therefore needs no halo. The
position encoding pools the global latitude and longitude of each patch, so every
shard passes its slice of the global coordinates, never shard-local indices.

The shard is the finest level of the backbone's merge-aligned partition, so encoder
output, backbone, and decoder input agree on every rank's columns. Both modules are
replicated across channel ranks and shard only on the spatial axis. The decoder
all-gathers its predictions along longitude, so every rank finishes the step with
the full field and the rollout advances in lockstep.

Both model families run: keyword lead-time arguments (``lead_time`` for the
0.25/0.1 family, ``lead_times`` for Aurora 1.5) pass through unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from flash_aurora.engine.distributed.collectives import all_gather_uneven_last_dim
from flash_aurora.engine.distributed.longitude_partition import LongitudeShards, partition_levels
from flash_aurora.engine.distributed.process_mesh import ProcessMesh
from flash_aurora.models.aurora.batch import Batch


def _slice_longitude(tensor: torch.Tensor, columns: slice) -> torch.Tensor:
    return tensor[..., columns].contiguous()


def slice_batch_longitude(batch: Batch, columns: slice) -> Batch:
    """Restrict every field and the coordinates of ``batch`` to longitude ``columns``."""
    metadata = batch.metadata
    lat = metadata.lat if metadata.lat.dim() == 1 else _slice_longitude(metadata.lat, columns)
    return dataclasses.replace(
        batch,
        surf_vars={k: _slice_longitude(v, columns) for k, v in batch.surf_vars.items()},
        static_vars={k: _slice_longitude(v, columns) for k, v in batch.static_vars.items()},
        atmos_vars={k: _slice_longitude(v, columns) for k, v in batch.atmos_vars.items()},
        metadata=dataclasses.replace(
            metadata, lat=lat, lon=_slice_longitude(metadata.lon, columns)
        ),
    )


class _PatchShard:
    """This rank's patch columns on the finest backbone level, and their grid points."""

    def __init__(self, mesh: ProcessMesh, patch_size: int, num_levels: int) -> None:
        self.rank = mesh.coordinate.spatial
        self.size = mesh.shape.spatial
        self.patch_size = patch_size
        self.num_levels = num_levels

    def patch_shards(self, patch_width: int) -> LongitudeShards:
        return partition_levels(patch_width, self.num_levels, self.size)[0]

    def grid_columns(self, grid_width: int) -> slice:
        if grid_width % self.patch_size != 0:
            raise ValueError(f"grid width {grid_width} is not a multiple of patch {self.patch_size}")
        patches = self.patch_shards(grid_width // self.patch_size).columns(self.rank)
        return slice(patches.start * self.patch_size, patches.stop * self.patch_size)

    def grid_widths(self, grid_width: int) -> tuple[int, ...]:
        shards = self.patch_shards(grid_width // self.patch_size)
        return tuple(shards.shard_width(rank) * self.patch_size for rank in range(self.size))


class LongitudeShardedEncoder(nn.Module):
    """Encodes this rank's longitude shard; returns full-channel tokens of that shard."""

    def __init__(self, encoder: nn.Module, mesh: ProcessMesh, num_levels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.shard = _PatchShard(mesh, encoder.patch_size, num_levels)

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    @property
    def latent_levels(self) -> int:
        return self.encoder.latent_levels

    def forward(self, batch: Batch, **lead_time_kwargs: Any) -> torch.Tensor:
        columns = self.shard.grid_columns(batch.metadata.lon.shape[-1])
        return self.encoder(slice_batch_longitude(batch, columns), **lead_time_kwargs)


class LongitudeShardedDecoder(nn.Module):
    """Decodes this rank's longitude shard, then all-gathers the prediction along longitude."""

    def __init__(self, decoder: nn.Module, mesh: ProcessMesh, num_levels: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.shard = _PatchShard(mesh, decoder.patch_size, num_levels)
        self.spatial_group = mesh.spatial_group

    @property
    def patch_size(self) -> int:
        return self.decoder.patch_size

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        **lead_time_kwargs: Any,
    ) -> Batch:
        levels, height, width = patch_res
        local_width = self.shard.patch_shards(width).shard_width(self.shard.rank)
        grid_width = batch.metadata.lon.shape[-1]
        local = self.decoder(
            x,
            slice_batch_longitude(batch, self.shard.grid_columns(grid_width)),
            patch_res=(levels, height, local_width),
            **lead_time_kwargs,
        )
        widths = self.shard.grid_widths(grid_width)
        return dataclasses.replace(
            local,
            surf_vars={k: self._gather(v, widths) for k, v in local.surf_vars.items()},
            static_vars=batch.static_vars,
            atmos_vars={k: self._gather(v, widths) for k, v in local.atmos_vars.items()},
            metadata=dataclasses.replace(
                local.metadata,
                lat=batch.metadata.lat.to(local.metadata.lat.dtype),
                lon=batch.metadata.lon.to(local.metadata.lon.dtype),
            ),
        )

    def _gather(self, field: torch.Tensor, widths: tuple[int, ...]) -> torch.Tensor:
        return all_gather_uneven_last_dim(field.contiguous(), widths, self.spatial_group)
