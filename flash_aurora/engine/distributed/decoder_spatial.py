"""Spatially partition Perceiver3D decoder across two GPUs (west / east patch columns)."""

from __future__ import annotations

import contextlib
import copy
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from flash_aurora.aurora.batch import Batch, Metadata

from flash_aurora.aurora.model.util import check_lat_lon_dtype, unpatchify
from flash_aurora.engine.runtime.cuda_memory import release_tensor_storage, trim_cuda_cache

if TYPE_CHECKING:
    from flash_aurora.aurora.model.aurora import Aurora
    from flash_aurora.aurora.model.decoder import Perceiver3DDecoder

_DECODER_SPATIAL_REPLICA_ATTR = "_flash_aurora_decoder_spatial_replica"


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def apply_decoder_spatial_placement(
    model: Aurora,
    *,
    decoder_device: str,
    decoder_spatial_parallel: bool,
    decoder_spatial_devices: tuple[str, ...],
) -> None:
    """Place decoder module(s) for unified or spatially split inference."""
    decoder = model.decoder
    decoder.to(decoder_device)

    existing = getattr(model, _DECODER_SPATIAL_REPLICA_ATTR, None)
    if existing is not None:
        delattr(model, _DECODER_SPATIAL_REPLICA_ATTR)

    if not decoder_spatial_parallel or len(decoder_spatial_devices) < 2:
        return

    secondary = decoder_spatial_devices[0]
    if secondary == decoder_device:
        return

    replica = copy.deepcopy(decoder)
    replica.to(secondary)
    replica.eval()
    setattr(model, _DECODER_SPATIAL_REPLICA_ATTR, replica)


def decoder_spatial_replica(model: Aurora) -> Perceiver3DDecoder | None:
    return getattr(model, _DECODER_SPATIAL_REPLICA_ATTR, None)


def clear_decoder_spatial_replica(model: Aurora) -> None:
    if hasattr(model, _DECODER_SPATIAL_REPLICA_ATTR):
        delattr(model, _DECODER_SPATIAL_REPLICA_ATTR)


def _split_backbone_tokens(
    x: torch.Tensor,
    patch_res: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int]:
    """Split ``(B, C*H*W, D)`` tokens into west / east columns in patch space."""
    c_levels, h_patches, w_patches = patch_res
    x_hw = rearrange(
        x,
        "B (C H W) D -> B H W C D",
        C=c_levels,
        H=h_patches,
        W=w_patches,
    )
    w_west = w_patches // 2
    w_east = w_patches - w_west
    patch_res_west = (c_levels, h_patches, w_west)
    patch_res_east = (c_levels, h_patches, w_east)
    x_west = rearrange(
        x_hw[:, :, :w_west, :, :],
        "B H W C D -> B (C H W) D",
        C=c_levels,
        H=h_patches,
        W=w_west,
    )
    x_east = rearrange(
        x_hw[:, :, w_west:, :, :],
        "B H W C D -> B (C H W) D",
        C=c_levels,
        H=h_patches,
        W=w_east,
    )
    return x_west, x_east, patch_res_west, patch_res_east, w_west, w_east


def _forward_decoder_slice(
    decoder: Perceiver3DDecoder,
    x: torch.Tensor,
    *,
    surf_vars: tuple[str, ...],
    atmos_vars: tuple[str, ...],
    atmos_levels: tuple[int | float, ...],
    patch_res: tuple[int, int, int],
    height: int,
    width: int,
    lead_time: timedelta,
    autocast_bf16: bool,
    use_tensor_core: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_aurora.aurora.model.custom_op_paths import (
        backbone_tf32_matmul_context,
        encoder_decoder_autocast,
    )

    c_levels, h_patches, w_patches = patch_res
    batch_size = x.shape[0]

    with encoder_decoder_autocast(enabled=autocast_bf16):
        with backbone_tf32_matmul_context(enabled=use_tensor_core):
            x = rearrange(
                x,
                "B (C H W) D -> B (H W) C D",
                C=c_levels,
                H=h_patches,
                W=w_patches,
            )

            x_surf = torch.stack(
                [decoder.surf_heads[name](x[..., :1, :]) for name in surf_vars],
                dim=-1,
            )
            x_surf = x_surf.reshape(*x_surf.shape[:3], -1)
            surf_preds = unpatchify(x_surf, len(surf_vars), height, width, decoder.patch_size)
            surf_preds = surf_preds.squeeze(2)

            levels_embed = decoder._levels_embed(
                atmos_levels, device=x.device, dtype=x.dtype
            )
            levels_embed = levels_embed.expand(batch_size, x.size(1), -1, -1)
            x_atmos = decoder.deaggregate_levels(
                levels_embed,
                x[..., 1:, :],
                decoder.level_decoder,
            )
            if decoder.separate_perceiver:
                x_atmos_alternate = decoder.deaggregate_levels(
                    levels_embed,
                    x[..., 1:, :],
                    decoder.level_decoder_alternate,
                )
            else:
                x_atmos_alternate = x_atmos

            head_inputs = [
                x_atmos if name not in decoder.separate_perceiver else x_atmos_alternate
                for name in atmos_vars
            ]
            if not decoder.level_condition:
                stacked = torch.stack(
                    [decoder.atmos_heads[name](tensor) for name, tensor in zip(atmos_vars, head_inputs)],
                    dim=-1,
                )
            else:
                stacked = torch.stack(
                    [
                        decoder.atmos_heads[name](tensor, levels=atmos_levels)
                        for name, tensor in zip(atmos_vars, head_inputs)
                    ],
                    dim=-1,
                )
            stacked = stacked.reshape(*stacked.shape[:3], -1)
            atmos_preds = unpatchify(stacked, len(atmos_vars), height, width, decoder.patch_size)

    return surf_preds, atmos_preds


def forward_decoder_spatial_parallel(
    model: Aurora,
    x: torch.Tensor,
    batch: Batch,
    *,
    patch_res: tuple[int, int, int],
    lead_time: timedelta,
    spatial_devices: tuple[torch.device, torch.device],
    autocast_bf16: bool,
    use_tensor_core: bool,
) -> Batch:
    """Run decoder on west / east spatial halves concurrently on two devices."""
    decoder = model.decoder
    replica = decoder_spatial_replica(model)
    if replica is None:
        raise RuntimeError("decoder spatial replica is not installed on the model")

    west_dev, east_dev = spatial_devices
    if west_dev == east_dev:
        from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

        with _device_context(east_dev):
            with torch.inference_mode():
                return run_with_encoder_decoder_routing(
                    decoder.forward,
                    x,
                    batch,
                    patch_res,
                    lead_time,
                    autocast_bf16=autocast_bf16,
                    use_tensor_core=use_tensor_core,
                )

    surf_vars = tuple(batch.surf_vars.keys())
    atmos_vars = tuple(batch.atmos_vars.keys())
    atmos_levels = batch.metadata.atmos_levels
    surf_vars += tuple(f"{name}_mod" for name in surf_vars if name in decoder.modulation_heads)
    atmos_vars += tuple(f"{name}_mod" for name in atmos_vars if name in decoder.modulation_heads)

    lat, lon = batch.metadata.lat, batch.metadata.lon
    check_lat_lon_dtype(lat, lon)
    lat = lat.to(dtype=torch.float32)
    lon = lon.to(dtype=torch.float32)
    height = lat.shape[0]
    full_width = lon.shape[-1]
    patch_size = decoder.patch_size

    x_west, x_east, patch_res_west, patch_res_east, w_west, w_east = _split_backbone_tokens(
        x, patch_res
    )
    width_west = w_west * patch_size
    width_east = w_east * patch_size

    west_stream = torch.cuda.Stream(device=west_dev)
    east_stream = torch.cuda.Stream(device=east_dev)
    surf_west: torch.Tensor | None = None
    surf_east: torch.Tensor | None = None
    atmos_west: torch.Tensor | None = None
    atmos_east: torch.Tensor | None = None

    with torch.cuda.stream(west_stream):
        with _device_context(west_dev):
            with torch.inference_mode():
                x_local = x_west.to(west_dev, non_blocking=True)
                surf_west, atmos_west = _forward_decoder_slice(
                    replica,
                    x_local,
                    surf_vars=surf_vars,
                    atmos_vars=atmos_vars,
                    atmos_levels=atmos_levels,
                    patch_res=patch_res_west,
                    height=height,
                    width=width_west,
                    lead_time=lead_time,
                    autocast_bf16=autocast_bf16,
                    use_tensor_core=use_tensor_core,
                )
                if x_local.is_cuda:
                    release_tensor_storage(x_local)

    with torch.cuda.stream(east_stream):
        with _device_context(east_dev):
            with torch.inference_mode():
                x_local = x_east.to(east_dev, non_blocking=True)
                surf_east, atmos_east = _forward_decoder_slice(
                    decoder,
                    x_local,
                    surf_vars=surf_vars,
                    atmos_vars=atmos_vars,
                    atmos_levels=atmos_levels,
                    patch_res=patch_res_east,
                    height=height,
                    width=width_east,
                    lead_time=lead_time,
                    autocast_bf16=autocast_bf16,
                    use_tensor_core=use_tensor_core,
                )
                if x_local.is_cuda:
                    release_tensor_storage(x_local)
                    trim_cuda_cache(east_dev)

    west_stream.synchronize()
    east_stream.synchronize()

    assert surf_west is not None and surf_east is not None
    assert atmos_west is not None and atmos_east is not None

    merge_device = west_dev
    surf_merged = torch.cat(
        [
            surf_west.to(merge_device, non_blocking=True),
            surf_east.to(merge_device, non_blocking=True),
        ],
        dim=-1,
    )
    atmos_merged = torch.cat(
        [
            atmos_west.to(merge_device, non_blocking=True),
            atmos_east.to(merge_device, non_blocking=True),
        ],
        dim=-1,
    )
    if merge_device.type == "cuda":
        torch.cuda.synchronize(merge_device)

    assert surf_merged.shape[-1] == full_width
    assert atmos_merged.shape[-1] == full_width

    return Batch(
        {v: surf_merged[:, i] for i, v in enumerate(surf_vars)},
        batch.static_vars,
        {v: atmos_merged[:, i] for i, v in enumerate(atmos_vars)},
        Metadata(
            lat=lat,
            lon=lon,
            time=tuple(t + lead_time for t in batch.metadata.time),
            atmos_levels=atmos_levels,
            rollout_step=batch.metadata.rollout_step + 1,
        ),
    )
