from __future__ import annotations

from pathlib import Path

from flash_aurora.engine.core.config import STANDARD_LEVELS
from flash_aurora.engine.core.redaction import safe_config_label, sanitize_exception
from flash_aurora.engine.ingress.download.credentials import (
    CDS_DEFAULT_URL,
    DownloadCredentials,
    active_download_credentials,
    merge_credentials,
)
from flash_aurora.engine.ingress.download.paths import cdsapirc_path, ensure_directory, normalize_path


class CdsConfigError(FileNotFoundError):
    """Raised when CDS credentials are missing."""


def require_cdsapi():
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "CDS downloads require cdsapi. Install with: uv pip install cdsapi netcdf4"
        ) from exc
    return cdsapi


def cds_client():
    active = active_download_credentials()
    merged = merge_credentials(active)
    settings = merged.cds_settings()
    cdsapi = require_cdsapi()
    if settings is not None:
        url, key = settings
        try:
            return cdsapi.Client(url=url, key=key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize CDS client: {sanitize_exception(exc)}") from None

    config_path = cdsapirc_path()
    if not config_path.is_file():
        raise CdsConfigError(
            "Missing CDS credentials. Pass cds_api_key to DataDownloader.ensure(), "
            f"set CDSAPI_KEY, or create {safe_config_label(config_path)} "
            "(see https://cds.climate.copernicus.eu/how-to-api). "
            "If you used getpass(), the string in parentheses is only a prompt—not your API key."
        )
    try:
        return cdsapi.Client()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize CDS client: {sanitize_exception(exc)}") from None


def _cds_retrieve(client, dataset: str, params: dict, target: str) -> None:
    try:
        client.retrieve(dataset, params, target)
    except Exception as exc:
        raise RuntimeError(f"CDS retrieve failed: {sanitize_exception(exc)}") from None


def download_era5_static(
    cache_dir: Path | str,
    *,
    year: str = "2023",
    month: str = "01",
    day: str = "01",
) -> Path:
    target = normalize_path(cache_dir) / "static.nc"
    if target.is_file():
        return target

    ensure_directory(target.parent)
    client = cds_client()
    _cds_retrieve(
        client,
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": year,
            "month": month,
            "day": day,
            "time": "00:00",
            "format": "netcdf",
        },
        str(target),
    )
    return target


def download_era5_surface(cache_dir: Path | str, day: str) -> Path:
    target = normalize_path(cache_dir) / f"{day}-surface-level.nc"
    if target.is_file():
        return target

    year, month, dd = day.split("-")
    ensure_directory(target.parent)
    client = cds_client()
    _cds_retrieve(
        client,
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ],
            "year": year,
            "month": month,
            "day": dd,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(target),
    )
    return target


def download_era5_atmospheric(cache_dir: Path | str, day: str) -> Path:
    target = normalize_path(cache_dir) / f"{day}-atmospheric.nc"
    if target.is_file():
        return target

    year, month, dd = day.split("-")
    ensure_directory(target.parent)
    client = cds_client()
    _cds_retrieve(
        client,
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ],
            "pressure_level": [str(level) for level in STANDARD_LEVELS],
            "year": year,
            "month": month,
            "day": dd,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(target),
    )
    return target


def download_era5_day(cache_dir: Path | str, day: str, *, include_static: bool = True) -> dict[str, Path]:
    cache_dir = normalize_path(cache_dir)
    paths: dict[str, Path] = {}
    if include_static:
        paths["static"] = download_era5_static(cache_dir)
    paths["surface"] = download_era5_surface(cache_dir, day)
    paths["atmospheric"] = download_era5_atmospheric(cache_dir, day)
    return paths
