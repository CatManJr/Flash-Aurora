from __future__ import annotations

"""Coordinate reference helpers for egress masks and GeoTIFF export."""

BATCH_CRS = "EPSG:4326"
DEFAULT_EXPORT_CRS = "EPSG:3857"


def normalize_crs(crs: str | int) -> str:
    if isinstance(crs, int):
        return f"EPSG:{crs}"
    value = str(crs).strip()
    if value.upper().startswith("EPSG:"):
        return value.upper().replace("epsg:", "EPSG:")
    if value.isdigit():
        return f"EPSG:{value}"
    return value


def require_pyproj():
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "CRS transforms require pyproj (installed with rasterio)."
        ) from exc
    return Transformer


def transform_points(
    points: list[tuple[float, float]],
    src_crs: str | int,
    dst_crs: str | int = BATCH_CRS,
) -> list[tuple[float, float]]:
    if not points:
        return []
    src = normalize_crs(src_crs)
    dst = normalize_crs(dst_crs)
    if src == dst:
        return list(points)
    Transformer = require_pyproj()
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    xs, ys = zip(*points)
    out_x, out_y = transformer.transform(xs, ys)
    return list(zip(out_x, out_y, strict=True))


def envelope_wgs84(
    west: float,
    south: float,
    east: float,
    north: float,
    src_crs: str | int,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) in EPSG:4326."""
    corners = [(west, south), (east, south), (east, north), (west, north)]
    transformed = transform_points(corners, src_crs, BATCH_CRS)
    lons = [point[0] for point in transformed]
    lats = [point[1] for point in transformed]
    return min(lons), min(lats), max(lons), max(lats)


def normalize_longitude(lon: float) -> float:
    return float(lon % 360.0)
