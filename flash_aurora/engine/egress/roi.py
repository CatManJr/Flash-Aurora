from __future__ import annotations

import numpy as np
import torch
from flash_aurora.models.aurora import Batch, Metadata

from flash_aurora.engine.egress.crs import (
    BATCH_CRS,
    batch_longitude_for_wgs84,
    normalize_longitude,
    roi_longitude_sort_order,
    roi_longitude_values,
)
from flash_aurora.engine.egress.export_options import RoiBounds, RoiGeoJson, RoiSpec, coerce_mask
from flash_aurora.engine.egress.mask import Mask


def _lon_mask(lons: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lon_min = normalize_longitude(lon_min)
    lon_max = normalize_longitude(lon_max)
    if lon_min <= lon_max:
        return (lon_min <= lons) & (lons <= lon_max)
    return (lon_min <= lons) | (lons <= lon_max)


def _slice_tensor_last_dims(tensor: torch.Tensor, lat_idx: np.ndarray, lon_idx: np.ndarray) -> torch.Tensor:
    lat_sel = torch.as_tensor(lat_idx, device=tensor.device)
    lon_sel = torch.as_tensor(lon_idx, device=tensor.device)
    return tensor[..., lat_sel, :][..., :, lon_sel]


def _batch_grid(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lat_grid, lon_grid


def _crop_batch(batch: Batch, inside: np.ndarray) -> Batch:
    if not inside.any():
        raise ValueError("mask does not intersect the batch grid")

    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    lat_idx = np.nonzero(inside.any(axis=1))[0]
    lon_idx = np.nonzero(inside.any(axis=0))[0]
    lon_selected = lon[lon_idx]
    lon_order = roi_longitude_sort_order(lon_selected)
    lon_idx = lon_idx[lon_order]
    lon_values = roi_longitude_values(lon_selected, lon_order)
    inside_crop = inside[lat_idx][:, lon_idx]

    def _mask_tensor(tensor: torch.Tensor) -> torch.Tensor:
        cropped = _slice_tensor_last_dims(tensor, lat_idx, lon_idx)
        if inside_crop.all():
            return cropped
        mask = torch.as_tensor(inside_crop, device=cropped.device)
        while mask.dim() < cropped.dim():
            mask = mask.unsqueeze(0)
        return torch.where(mask, cropped, torch.full_like(cropped, float("nan")))

    lat_c = torch.as_tensor(lat[lat_idx], dtype=batch.metadata.lat.dtype, device=batch.metadata.lat.device)
    lon_c = torch.as_tensor(lon_values, dtype=batch.metadata.lon.dtype, device=batch.metadata.lon.device)
    return Batch(
        surf_vars={k: _mask_tensor(v) for k, v in batch.surf_vars.items()},
        static_vars={k: _mask_tensor(v) for k, v in batch.static_vars.items()},
        atmos_vars={k: _mask_tensor(v) for k, v in batch.atmos_vars.items()},
        metadata=Metadata(
            lat=lat_c,
            lon=lon_c,
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
            rollout_step=batch.metadata.rollout_step,
        ),
    )


def _inside_from_bounds(batch: Batch, mask: Mask) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = mask.wgs84_bounds()
    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    lat_mask = (lat_min <= lat) & (lat <= lat_max)
    lon_mask = _lon_mask(lon, lon_min, lon_max)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            f"mask bounds do not intersect the batch grid "
            f"(lat [{lat.min():.2f}, {lat.max():.2f}], lon [{lon.min():.2f}, {lon.max():.2f}])"
        )
    return lat_mask[:, None] & lon_mask[None, :]


def _inside_from_polygon(batch: Batch, mask: Mask) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    ring = mask.wgs84_exterior_ring()
    lat_grid, lon_grid = _batch_grid(batch)
    inside = MplPath(ring).contains_points(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
    inside = inside.reshape(lat_grid.shape)
    if not inside.any():
        raise ValueError("polygon mask does not intersect the batch grid")
    return inside


def _inside_from_raster(batch: Batch, mask: Mask) -> np.ndarray:
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:
        raise ImportError(
            "Raster masks require rasterio."
        ) from exc

    assert mask.raster_path is not None
    lat_grid, lon_grid = _batch_grid(batch)
    with rasterio.open(mask.raster_path) as dataset:
        src_crs = mask.source_crs
        if dataset.crs is not None:
            src_crs = mask.source_crs if mask.source_crs != BATCH_CRS else str(dataset.crs)
        lon_wgs84 = batch_longitude_for_wgs84(lon_grid)
        xs, ys = warp_transform(
            BATCH_CRS,
            src_crs,
            lon_wgs84.ravel().tolist(),
            lat_grid.ravel().tolist(),
        )
        samples = np.fromiter(
            (value[0] for value in dataset.sample(zip(xs, ys), indexes=mask.raster_band)),
            dtype=np.float64,
            count=lat_grid.size,
        ).reshape(lat_grid.shape)
        nodata = mask.raster_nodata if mask.raster_nodata is not None else dataset.nodata

    inside = np.isfinite(samples)
    if nodata is not None:
        inside &= samples != nodata
    inside &= samples != 0
    if not inside.any():
        raise ValueError(f"raster mask does not intersect the batch grid: {mask.raster_path}")
    return inside


def apply_mask(batch: Batch, mask: Mask | None) -> Batch:
    if mask is None:
        return batch
    if mask.kind == "bounds":
        inside = _inside_from_bounds(batch, mask)
    elif mask.kind == "polygon":
        inside = _inside_from_polygon(batch, mask)
    else:
        inside = _inside_from_raster(batch, mask)
    return _crop_batch(batch, inside)


def clip_batch_to_bounds(batch: Batch, bounds: RoiBounds) -> Batch:
    return apply_mask(batch, bounds.to_mask())


def clip_batch_to_geojson(batch: Batch, roi: RoiGeoJson) -> Batch:
    return apply_mask(batch, roi.to_mask())


def apply_roi(batch: Batch, roi: RoiSpec | None) -> Batch:
    return apply_mask(batch, coerce_mask(roi))
