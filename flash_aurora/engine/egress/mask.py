from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from flash_aurora.engine.egress.crs import (
    BATCH_CRS,
    DEFAULT_EXPORT_CRS,
    envelope_wgs84,
    normalize_crs,
    normalize_longitude,
    transform_points,
)

MaskKind = Literal["bounds", "polygon", "raster"]


def _as_path(value: Path | str) -> Path:
    return Path(value).expanduser()


def _parse_geojson_geometry(payload: dict[str, Any]) -> list[list[tuple[float, float]]]:
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            raise ValueError("GeoJSON FeatureCollection is empty")
        geometry = features[0]["geometry"]
    elif payload.get("type") == "Feature":
        geometry = payload["geometry"]
    elif payload.get("type") in {"Polygon", "MultiPolygon"}:
        geometry = payload
    else:
        raise ValueError(f"Unsupported GeoJSON root type: {payload.get('type')!r}")

    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        rings = geometry["coordinates"]
    elif geom_type == "MultiPolygon":
        rings = geometry["coordinates"][0]
    else:
        raise ValueError(f"Only Polygon geometries are supported (got {geom_type!r})")

    return [
        [(float(x), float(y)) for x, y in ring]
        for ring in rings
    ]


def _load_geojson(source: Path | str | dict[str, Any]) -> list[list[tuple[float, float]]]:
    if isinstance(source, dict):
        return _parse_geojson_geometry(source)
    payload = json.loads(_as_path(source).read_text(encoding="utf-8"))
    return _parse_geojson_geometry(payload)


def _load_shapefile_rings(
    path: Path | str,
    source_crs: str | None,
) -> tuple[list[list[tuple[float, float]]], str]:
    path = _as_path(path)
    try:
        import fiona
    except ImportError as exc:
        raise ImportError(
            "Shapefile masks require fiona."
        ) from exc

    with fiona.open(path) as dataset:
        detected_crs = source_crs
        if detected_crs is None and dataset.crs is not None:
            detected_crs = normalize_crs(str(dataset.crs))
        if detected_crs is None:
            detected_crs = BATCH_CRS
        for feature in dataset:
            geometry = feature.get("geometry")
            if geometry is None:
                continue
            geom_type = geometry.get("type")
            if geom_type == "Polygon":
                rings = [
                    [(float(x), float(y)) for x, y in ring]
                    for ring in geometry["coordinates"]
                ]
                return rings, detected_crs
            if geom_type == "MultiPolygon":
                rings = [
                    [(float(x), float(y)) for x, y in ring]
                    for ring in geometry["coordinates"][0]
                ]
                return rings, detected_crs
    raise ValueError(f"No polygon geometry found in shapefile: {path}")


@dataclass(frozen=True)
class Mask:
    """Spatial mask for ROI clipping and GeoTIFF reprojection.

    Four creation modes:

    * **bounds** — axis-aligned window (``from_bounds``)
    * **polygon** — quadrilateral or GeoJSON/shapefile ring (``from_corners``, ``from_geojson``, ``from_shapefile``)
    * **raster** — georeferenced raster with valid pixels (``from_raster``)

    ``source_crs`` is the CRS of the mask definition. Aurora batches use geographic
    EPSG:4326 internally. ``export_crs`` defaults to Web Mercator (EPSG:3857).
    """

    kind: MaskKind
    source_crs: str = BATCH_CRS
    export_crs: str = DEFAULT_EXPORT_CRS
    bounds: tuple[float, float, float, float] | None = None
    rings: tuple[tuple[tuple[float, float], ...], ...] | None = None
    raster_path: Path | None = None
    raster_band: int = 1
    raster_nodata: float | None = None

    @classmethod
    def from_bounds(
        cls,
        *,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        crs: str | int = BATCH_CRS,
        export_crs: str | int = DEFAULT_EXPORT_CRS,
    ) -> Mask:
        """Axis-aligned lat/lon (or projected) bounding box."""
        west, south, east, north = float(lon_min), float(lat_min), float(lon_max), float(lat_max)
        return cls(
            kind="bounds",
            source_crs=normalize_crs(crs),
            export_crs=normalize_crs(export_crs),
            bounds=(west, south, east, north),
        )

    @classmethod
    def from_corners(
        cls,
        points: Sequence[tuple[float, float]],
        *,
        crs: str | int = BATCH_CRS,
        export_crs: str | int = DEFAULT_EXPORT_CRS,
    ) -> Mask:
        """Quadrilateral (or general polygon) from corner vertices ``(x, y)`` in ``crs``."""
        ring = tuple((float(x), float(y)) for x, y in points)
        if len(ring) < 3:
            raise ValueError("from_corners requires at least three points")
        if ring[0] != ring[-1]:
            ring = (*ring, ring[0])
        return cls(
            kind="polygon",
            source_crs=normalize_crs(crs),
            export_crs=normalize_crs(export_crs),
            rings=(ring,),
        )

    @classmethod
    def from_raster(
        cls,
        path: Path | str,
        *,
        band: int = 1,
        crs: str | int | None = None,
        nodata: float | None = None,
        export_crs: str | int = DEFAULT_EXPORT_CRS,
    ) -> Mask:
        """Mask from valid pixels in a georeferenced raster (non-nodata, non-zero)."""
        raster_path = _as_path(path)
        source_crs = normalize_crs(crs) if crs is not None else BATCH_CRS
        if crs is None:
            try:
                import rasterio
            except ImportError as exc:
                raise ImportError(
                    "Raster masks require rasterio."
                ) from exc
            with rasterio.open(raster_path) as dataset:
                if dataset.crs is not None:
                    source_crs = normalize_crs(str(dataset.crs))
        return cls(
            kind="raster",
            source_crs=source_crs,
            export_crs=normalize_crs(export_crs),
            raster_path=raster_path,
            raster_band=band,
            raster_nodata=nodata,
        )

    @classmethod
    def from_shapefile(
        cls,
        path: Path | str,
        *,
        crs: str | int | None = None,
        export_crs: str | int = DEFAULT_EXPORT_CRS,
    ) -> Mask:
        """Mask from the first polygon in an ESRI shapefile."""
        shapefile_path = _as_path(path)
        rings, source_crs = _load_shapefile_rings(
            shapefile_path,
            normalize_crs(crs) if crs is not None else None,
        )
        return cls(
            kind="polygon",
            source_crs=source_crs,
            export_crs=normalize_crs(export_crs),
            rings=tuple(tuple(ring) for ring in rings),
        )

    @classmethod
    def from_geojson(
        cls,
        source: Path | str | dict[str, Any],
        *,
        crs: str | int = BATCH_CRS,
        export_crs: str | int = DEFAULT_EXPORT_CRS,
    ) -> Mask:
        """Mask from GeoJSON / JSON polygon geometry."""
        rings = _load_geojson(source)
        return cls(
            kind="polygon",
            source_crs=normalize_crs(crs),
            export_crs=normalize_crs(export_crs),
            rings=tuple(tuple(ring) for ring in rings),
        )

    @classmethod
    def california(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """Axis-aligned envelope of California, snapped for 0.25° grids (~38×44 cells)."""
        return cls.from_bounds(
            lat_min=32.5,
            lat_max=42.0,
            lon_min=235.0,
            lon_max=246.0,
            export_crs=export_crs,
        )

    @classmethod
    def china(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """Mainland China envelope for 0.25° regional export (~145×248 cells)."""
        return cls.from_bounds(
            lat_min=18.0,
            lat_max=54.0,
            lon_min=73.0,
            lon_max=135.0,
            export_crs=export_crs,
        )

    @classmethod
    def usa(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """Continental United States envelope for 0.25° regional export (~100×240 cells)."""
        return cls.from_bounds(
            lat_min=24.5,
            lat_max=49.5,
            lon_min=235.0,
            lon_max=295.0,
            export_crs=export_crs,
        )

    @classmethod
    def western_europe(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """Western Europe envelope (lon wrap 350°–15°E) for 0.25° regional export."""
        return cls.from_bounds(
            lat_min=36.0,
            lat_max=61.0,
            lon_min=350.0,
            lon_max=15.0,
            export_crs=export_crs,
        )

    @classmethod
    def north_africa(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """North Africa envelope (Maghreb to Egypt; lon wrap 343°–40°E) for 0.25° export."""
        return cls.from_bounds(
            lat_min=15.0,
            lat_max=37.5,
            lon_min=343.0,
            lon_max=40.0,
            export_crs=export_crs,
        )

    @classmethod
    def pacific_tc_panel(cls, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        """Match ``example_tc_tracking.ipynb`` plot extent (20–45°N, 120–140°E)."""
        return cls.from_bounds(
            lat_min=20.0,
            lat_max=45.0,
            lon_min=120.0,
            lon_max=140.0,
            export_crs=export_crs,
        )

    def wgs84_bounds(self) -> tuple[float, float, float, float]:
        """Geographic envelope ``(lon_min, lat_min, lon_max, lat_max)`` for clipping.

        When ``lon_min > lon_max`` the longitude window crosses the 0° meridian
        (Aurora's 0–360° grid). :func:`flash_aurora.engine.egress.roi._lon_mask`
        interprets that as a wrap-around interval.
        """
        if self.kind == "bounds":
            assert self.bounds is not None
            west, south, east, north = self.bounds
            if normalize_crs(self.source_crs) == BATCH_CRS:
                return (
                    normalize_longitude(west),
                    south,
                    normalize_longitude(east),
                    north,
                )
            lon_min, lat_min, lon_max, lat_max = envelope_wgs84(
                west, south, east, north, self.source_crs
            )
            return (
                normalize_longitude(lon_min),
                lat_min,
                normalize_longitude(lon_max),
                lat_max,
            )

        if self.kind == "polygon":
            assert self.rings is not None
            exterior = self.rings[0]
            transformed = transform_points(list(exterior), self.source_crs, BATCH_CRS)
            lons = [normalize_longitude(point[0]) for point in transformed]
            lats = [point[1] for point in transformed]
            return min(lons), min(lats), max(lons), max(lats)

        assert self.raster_path is not None
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "Raster masks require rasterio."
            ) from exc
        with rasterio.open(self.raster_path) as dataset:
            west, south, east, north = dataset.bounds
            src_crs = self.source_crs if self.source_crs != BATCH_CRS else normalize_crs(str(dataset.crs))
        lon_min, lat_min, lon_max, lat_max = envelope_wgs84(west, south, east, north, src_crs)
        return normalize_longitude(lon_min), lat_min, normalize_longitude(lon_max), lat_max

    def wgs84_exterior_ring(self) -> list[tuple[float, float]]:
        if self.kind != "polygon":
            raise ValueError("wgs84_exterior_ring is only defined for polygon masks")
        assert self.rings is not None
        transformed = transform_points(list(self.rings[0]), self.source_crs, BATCH_CRS)
        return [(normalize_longitude(lon), lat) for lon, lat in transformed]
