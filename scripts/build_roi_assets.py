#!/usr/bin/env python3
"""Build US state ROI assets (shapefile, GeoJSON, raster mask) for docs/roi."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

STATES = {
    "california": "California",
    "georgia": "Georgia",
    "michigan": "Michigan",
    "texas": "Texas",
}

STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
GRID_STEP = 0.25
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "docs" / "roi"


def _snap_down(value: float, step: float = GRID_STEP) -> float:
    return float(np.floor(value / step) * step)


def _snap_up(value: float, step: float = GRID_STEP) -> float:
    return float(np.ceil(value / step) * step)


def _load_states_geojson() -> dict:
    with urllib.request.urlopen(STATES_GEOJSON_URL, timeout=60) as response:  # noqa: S310
        return json.load(response)


def main() -> None:
    try:
        import fiona
        import rasterio
        from rasterio import features
        from rasterio.transform import from_origin
        from shapely.geometry import shape
    except ImportError as exc:
        raise SystemExit(f"Missing geo dependency: {exc}") from exc

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _load_states_geojson()
    by_name = {
        str(feature["properties"]["name"]): feature
        for feature in payload["features"]
    }

    manifest: dict[str, dict[str, object]] = {}

    for slug, state_name in STATES.items():
        feature = by_name[state_name]
        geometry = shape(feature["geometry"])
        minx, miny, maxx, maxy = geometry.bounds
        west = _snap_down(minx)
        south = _snap_down(miny)
        east = _snap_up(maxx)
        north = _snap_up(maxy)

        state_dir = OUTPUT_ROOT / slug
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        shp_path = state_dir / f"{slug}.shp"
        schema = {
            "geometry": feature["geometry"]["type"],
            "properties": {"name": "str:32", "state": "str:32"},
        }
        with fiona.open(
            shp_path,
            "w",
            driver="ESRI Shapefile",
            crs="EPSG:4326",
            schema=schema,
        ) as sink:
            sink.write(
                {
                    "type": "Feature",
                    "properties": {"name": slug, "state": state_name},
                    "geometry": feature["geometry"],
                }
            )

        geojson_path = OUTPUT_ROOT / f"{slug}.geojson"
        geojson_path.write_text(
            json.dumps(
                {
                    "type": "Feature",
                    "properties": {"name": slug, "state": state_name},
                    "geometry": feature["geometry"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        width = int(round((east - west) / GRID_STEP))
        height = int(round((north - south) / GRID_STEP))
        transform = from_origin(west, north, GRID_STEP, GRID_STEP)
        mask_array = features.rasterize(
            [(geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        raster_path = OUTPUT_ROOT / f"{slug}_mask.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
            compress="deflate",
        ) as dataset:
            dataset.write(mask_array, 1)

        manifest[slug] = {
            "state": state_name,
            "shapefile": f"{slug}/{slug}.shp",
            "geojson": f"{slug}.geojson",
            "raster": f"{slug}_mask.tif",
            "bounds": {
                "lat_min": south,
                "lat_max": north,
                "lon_min": west,
                "lon_max": east,
            },
        }
        print(f"wrote {slug}: {width}x{height} mask")

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
