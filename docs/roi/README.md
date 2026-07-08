# ROI assets for `example_roi_export.ipynb`

Boundary geometries for four U.S. states, aligned to the Aurora 0.25° grid for raster masks. Used in **Part B** of [`example_roi_export.ipynb`](../example_roi_export.ipynb) (0.1° HRES): California (GeoJSON), Georgia (bounds), Michigan (raster), Texas (shapefile). **Part A** uses built-in `RoiBatch.example_batch()` presets instead.

## Files per state

| State | Shapefile | GeoJSON | Raster mask |
| --- | --- | --- | --- |
| California | `california/california.shp` | `california.geojson` | `california_mask.tif` |
| Georgia | `georgia/georgia.shp` | `georgia.geojson` | `georgia_mask.tif` |
| Michigan | `michigan/michigan.shp` | `michigan.geojson` | `michigan_mask.tif` |
| Texas | `texas/texas.shp` | `texas.geojson` | `texas_mask.tif` |

`manifest.json` lists bounds and relative paths.

## Regenerate

```bash
python scripts/build_roi_assets.py
```

## Source

State polygons are derived from the [PublicaMundi US states GeoJSON](https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json) (simplified cartographic boundaries). Raster masks use a 0.25° grid with `rasterio.features.rasterize` over each state's snapped bounding box.
