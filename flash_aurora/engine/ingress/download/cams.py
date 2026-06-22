from __future__ import annotations

import zipfile
from pathlib import Path

from flash_aurora.engine.core.config import STANDARD_LEVELS
from flash_aurora.engine.ingress.download.ads import ads_client, ads_retrieve
from flash_aurora.engine.ingress.download.paths import ensure_directory, normalize_path

CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"

CAMS_SURFACE_VARS: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
)

CAMS_ATMOS_VARS: tuple[str, ...] = (
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "specific_humidity",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
)


def _cams_retrieve_params(day: str) -> dict:
    return {
        "type": "forecast",
        "leadtime_hour": "0",
        "variable": list(CAMS_SURFACE_VARS + CAMS_ATMOS_VARS),
        "pressure_level": [str(level) for level in STANDARD_LEVELS],
        "date": day,
        "time": ["00:00", "12:00"],
        "format": "netcdf_zip",
    }


def _unpack_cams_zip(zip_path: Path, *, surf_path: Path, atmos_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        if not surf_path.is_file():
            surf_path.write_bytes(zf.read("data_sfc.nc"))
        if not atmos_path.is_file():
            atmos_path.write_bytes(zf.read("data_plev.nc"))


def download_cams_day(cache_dir: Path | str, day: str) -> dict[str, Path]:
    """Download and unpack CAMS analysis NetCDF for ``day`` (``YYYY-MM-DD``)."""
    cache_dir = normalize_path(cache_dir)
    ensure_directory(cache_dir)

    surf_path = cache_dir / f"{day}-cams-surface-level.nc"
    atmos_path = cache_dir / f"{day}-cams-atmospheric.nc"
    zip_path = cache_dir / f"{day}-cams.nc.zip"
    paths = {"surface": surf_path, "atmospheric": atmos_path}

    if surf_path.is_file() and atmos_path.is_file():
        return paths

    if not zip_path.is_file():
        client = ads_client()
        ads_retrieve(client, CAMS_DATASET, _cams_retrieve_params(day), str(zip_path))

    _unpack_cams_zip(zip_path, surf_path=surf_path, atmos_path=atmos_path)
    return paths
