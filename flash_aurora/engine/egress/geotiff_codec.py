from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flash_aurora.models.aurora import Batch

from flash_aurora.engine.egress.crs import BATCH_CRS, normalize_crs, roi_longitude_plot_bounds


def _require_rasterio():
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, calculate_default_transform, reproject
    except ImportError as exc:
        raise ImportError(
            "GeoTIFF export requires rasterio."
        ) from exc
    return rasterio, from_bounds, Resampling, calculate_default_transform, reproject


def _surface_field(batch: Batch, variable: str) -> np.ndarray:
    if variable not in batch.surf_vars:
        available = ", ".join(sorted(batch.surf_vars))
        raise KeyError(f"surface variable {variable!r} not in batch; available: {available}")
    tensor = batch.surf_vars[variable]
    if tensor.dim() == 4:
        tensor = tensor[0, -1]
    elif tensor.dim() == 3:
        tensor = tensor[-1]
    elif tensor.dim() != 2:
        raise ValueError(f"surface variable {variable!r} must reduce to 2-D [lat, lon]")
    return tensor.detach().float().cpu().numpy()


def _georeference(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("GeoTIFF export expects 1-D latitude and longitude coordinates")
    array = np.asarray(data, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    # Aurora batches store latitude north-to-south (decreasing). GDAL row 0 is north,
    # which already matches tensor row 0. Only flip when latitude increases south-to-north.
    if lat[0] < lat[-1]:
        array = np.flipud(array)
    south = float(np.min(lat))
    north = float(np.max(lat))
    west, east = roi_longitude_plot_bounds(lon)
    return array, west, south, east, north


def write_surface_geotiff(
    batch: Batch,
    path: Path | str,
    variable: str,
    *,
    nodata: float = -9999.0,
    crs: str | int = "EPSG:3857",
) -> Path:
    """Write one surface field as a georeferenced single-band GeoTIFF."""
    rasterio, from_bounds, Resampling, calculate_default_transform, reproject = _require_rasterio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs = normalize_crs(crs)

    data = _surface_field(batch, variable)
    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    array, west, south, east, north = _georeference(data, lat, lon)
    array = np.where(np.isfinite(array), array, nodata).astype(np.float32, copy=False)

    height, width = array.shape
    src_crs = BATCH_CRS
    src_transform = from_bounds(west, south, east, north, width, height)

    if dst_crs == src_crs:
        dst_array = array
        dst_transform = src_transform
        dst_height, dst_width = height, width
    else:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            width,
            height,
            west,
            south,
            east,
            north,
        )
        dst_array = np.full((dst_height, dst_width), nodata, dtype=np.float32)
        reproject(
            source=array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(dst_array, 1)
    return path


def geotiff_variables(batch: Batch, variables: tuple[str, ...] | None) -> tuple[str, ...]:
    if variables:
        return variables
    if batch.surf_vars:
        return (next(iter(batch.surf_vars)),)
    raise ValueError("batch has no surface variables for GeoTIFF export")
