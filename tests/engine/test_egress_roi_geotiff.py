from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from flash_aurora.models.aurora import Batch, Metadata
from flash_aurora.engine.egress.crs import BATCH_CRS, DEFAULT_EXPORT_CRS
from flash_aurora.engine.egress.export_options import ExportOptions, RoiBounds, RoiGeoJson
from flash_aurora.engine.egress.geotiff_codec import write_surface_geotiff
from flash_aurora.engine.egress.mask import Mask
from flash_aurora.engine.egress.roi import apply_mask, apply_roi, clip_batch_to_bounds
from flash_aurora.engine.egress.step_writer import RolloutStepWriter


def _grid_batch() -> Batch:
    lat = torch.linspace(90.0, -90.0, 181)
    lon = torch.linspace(0.0, 359.0, 360)
    height, width = len(lat), len(lon)
    return Batch(
        surf_vars={"msl": torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(),
            rollout_step=0,
        ),
    )


def test_mask_california_sized_for_quarter_degree_grid() -> None:
    batch = _grid_batch()
    mask = Mask.california()
    clipped = apply_mask(batch, mask)

    lat = clipped.metadata.lat.detach().cpu().numpy()
    lon = clipped.metadata.lon.detach().cpu().numpy()
    lon_min, lat_min, lon_max, lat_max = mask.wgs84_bounds()
    assert lat.min() >= lat_min
    assert lat.max() <= lat_max
    assert lon.min() >= lon_min
    assert lon.max() <= lon_max

    lat_res = abs(float(lat[0] - lat[1])) if len(lat) > 1 else 1.0
    lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 1.0
    assert abs(len(lat) - ((42.0 - 32.5) / lat_res + 1)) <= 1
    assert abs(len(lon) - ((246.0 - 235.0) / lon_res + 1)) <= 1
    if lat_res == 0.25:
        assert len(lat) == 39
        assert len(lon) == 45


def test_mask_western_europe_dateline_wrap() -> None:
    batch = _grid_batch()
    mask = Mask.western_europe()
    clipped = apply_mask(batch, mask)

    lat = clipped.metadata.lat.detach().cpu().numpy()
    lon = clipped.metadata.lon.detach().cpu().numpy()
    lon_min, lat_min, lon_max, lat_max = mask.wgs84_bounds()

    assert lon_min > lon_max  # 350 -> 15 wrap in mask definition
    assert lat.min() >= lat_min
    assert lat.max() <= lat_max
    assert lon.min() >= -10.5
    assert lon.max() <= 15.5
    assert (np.diff(lon) > 0).all()


def test_mask_north_africa_dateline_wrap() -> None:
    batch = _grid_batch()
    mask = Mask.north_africa()
    clipped = apply_mask(batch, mask)

    lat = clipped.metadata.lat.detach().cpu().numpy()
    lon = clipped.metadata.lon.detach().cpu().numpy()
    lon_min, lat_min, lon_max, lat_max = mask.wgs84_bounds()

    assert lon_min > lon_max  # 343 -> 40 wrap in mask definition
    assert lat.min() >= lat_min
    assert lat.max() <= lat_max
    assert lon.min() >= -18.0
    assert lon.max() <= 40.5
    assert (np.diff(lon) > 0).all()
    assert len(lon) <= 65
    assert len(lat) <= 25


def test_mask_from_bounds_pacific_tc_panel() -> None:
    batch = _grid_batch()
    mask = Mask.pacific_tc_panel()
    clipped = apply_mask(batch, mask)

    lat = clipped.metadata.lat.detach().cpu().numpy()
    lon = clipped.metadata.lon.detach().cpu().numpy()
    lon_min, lat_min, lon_max, lat_max = mask.wgs84_bounds()
    assert lat.min() >= lat_min
    assert lat.max() <= lat_max
    assert lon.min() >= lon_min
    assert lon.max() <= lon_max
    assert clipped.surf_vars["msl"].shape[-2:] == (len(lat), len(lon))


def test_mask_from_corners_matches_bounds() -> None:
    batch = _grid_batch()
    mask = Mask.from_corners(
        [(120.0, 20.0), (140.0, 20.0), (140.0, 45.0), (120.0, 45.0)],
        crs=BATCH_CRS,
    )
    clipped = apply_mask(batch, mask)
    assert clipped.surf_vars["msl"].shape[-2] > 0
    assert clipped.surf_vars["msl"].shape[-1] > 0


def test_mask_from_geojson_masks_outside_polygon(tmp_path: Path) -> None:
    batch = _grid_batch()
    geojson = {
        "type": "Polygon",
        "coordinates": [[[125.0, 25.0], [135.0, 25.0], [130.0, 35.0], [125.0, 25.0]]],
    }
    path = tmp_path / "roi.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    clipped = apply_mask(batch, Mask.from_geojson(path))
    values = clipped.surf_vars["msl"][0, 0].detach().cpu().numpy()
    assert np.isfinite(values).any()
    assert np.isnan(values).any()


def test_mask_from_raster_western_hemisphere_on_360_grid() -> None:
    rasterio = pytest.importorskip("rasterio")
    from pathlib import Path

    raster_path = Path(__file__).resolve().parents[2] / "docs" / "roi" / "michigan_mask.tif"
    if not raster_path.is_file():
        pytest.skip("bundled Michigan ROI raster not present")

    batch = _grid_batch()
    clipped = apply_mask(batch, Mask.from_raster(raster_path))
    assert clipped.surf_vars["msl"].shape[-2] > 0
    assert clipped.surf_vars["msl"].shape[-1] > 0


def test_mask_from_raster(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    height, width = 90, 180
    west, south, east, north = 120.0, 20.0, 140.0, 45.0
    data = np.ones((height, width), dtype=np.uint8)
    transform = from_bounds(west, south, east, north, width, height)
    raster_path = tmp_path / "mask.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=BATCH_CRS,
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data, 1)

    batch = _grid_batch()
    clipped = apply_mask(batch, Mask.from_raster(raster_path))
    assert clipped.surf_vars["msl"].shape[-2] > 0


def test_legacy_roi_bounds_still_works() -> None:
    batch = _grid_batch()
    clipped = clip_batch_to_bounds(batch, RoiBounds(lat_min=20, lat_max=45, lon_min=120, lon_max=140))
    assert clipped.surf_vars["msl"].shape[-2] > 0


def test_legacy_roi_geojson_still_works(tmp_path: Path) -> None:
    batch = _grid_batch()
    geojson = {
        "type": "Polygon",
        "coordinates": [[[125.0, 25.0], [135.0, 25.0], [130.0, 35.0], [125.0, 25.0]]],
    }
    path = tmp_path / "roi.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")
    clipped = apply_roi(batch, RoiGeoJson(path=path))
    assert clipped.surf_vars["msl"].shape[-2] > 0


def test_rollout_step_writer_netcdf_with_mask(tmp_path: Path) -> None:
    batch = _grid_batch()
    writer = RolloutStepWriter(
        tmp_path,
        ExportOptions(format="netcdf", mask=Mask.pacific_tc_panel()),
    )
    paths = writer.write_step(0, batch)
    assert len(paths) == 1
    assert paths[0].name == "prediction-000.nc"
    assert paths[0].is_file()


def test_rollout_step_writer_geotiff_web_mercator(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    batch = _grid_batch()
    writer = RolloutStepWriter(
        tmp_path,
        ExportOptions(
            format="geotiff",
            mask=Mask.pacific_tc_panel(),
            variables=("msl",),
        ),
    )
    paths = writer.write_step(0, batch)
    assert len(paths) == 1
    assert paths[0].name == "prediction-000-msl.tif"

    with rasterio.open(paths[0]) as dataset:
        assert dataset.crs.to_epsg() == 3857
        assert dataset.count == 1
        data = dataset.read(1)
        assert data.shape[0] > 0 and data.shape[1] > 0


def test_write_surface_geotiff_preserves_north_south_orientation(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    lat = torch.linspace(42.0, 33.0, 37)
    lon = torch.linspace(235.0, 246.0, 45)
    height, width = len(lat), len(lon)
    values = torch.zeros(1, 1, height, width)
    for row, latitude in enumerate(lat):
        values[0, 0, row, :] = float(latitude)
    batch = Batch(
        surf_vars={"msl": values},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(),
            rollout_step=0,
        ),
    )
    path = write_surface_geotiff(batch, tmp_path / "orient.tif", "msl", crs=BATCH_CRS)
    with rasterio.open(path) as dataset:
        data = dataset.read(1)
        assert dataset.bounds.top == pytest.approx(42.0)
        assert dataset.bounds.bottom == pytest.approx(33.0)
        assert data[0].mean() == pytest.approx(42.0, abs=0.5)
        assert data[-1].mean() == pytest.approx(33.0, abs=0.5)


def test_write_surface_geotiff_epsg4326(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    batch = apply_mask(_grid_batch(), Mask.pacific_tc_panel())
    path = write_surface_geotiff(batch, tmp_path / "msl.tif", "msl", crs=BATCH_CRS)
    assert path.is_file()


def test_mask_default_export_crs_is_web_mercator() -> None:
    mask = Mask.from_bounds(lat_min=0, lat_max=1, lon_min=0, lon_max=1)
    assert mask.export_crs == DEFAULT_EXPORT_CRS
