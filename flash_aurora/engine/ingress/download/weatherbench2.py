from __future__ import annotations

from pathlib import Path

from flash_aurora.engine.core.redaction import sanitize_exception
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.ingress.download.paths import ensure_directory, normalize_path

WB2_HRES_T0_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

WB2_HRES_SURFACE_VARS: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)

WB2_HRES_ATMOS_VARS: tuple[str, ...] = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
)


def require_weatherbench2_deps():
    try:
        import fsspec  # noqa: F401
        import xarray as xr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "WeatherBench2 downloads require fsspec and xarray. "
            "Install with: uv pip install gcsfs zarr"
        ) from exc

    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise ImportError("WeatherBench2 downloads require zarr: uv pip install zarr") from exc


def open_hres_t0_store():
    require_weatherbench2_deps()
    import xarray as xr

    # WeatherBench2 is a public GCS bucket; anonymous access avoids ADC warnings/hangs.
    return xr.open_zarr(
        WB2_HRES_T0_URL,
        storage_options={"token": "anon"},
        chunks=None,
    )


def download_hres_t0_day(cache_dir: Path | str, day: str) -> dict[str, Path]:
    cache_dir = normalize_path(cache_dir)
    ensure_directory(cache_dir)

    surface_path = cache_dir / f"{day}-surface-level.nc"
    atmospheric_path = cache_dir / f"{day}-atmospheric.nc"
    paths = {"surface": surface_path, "atmospheric": atmospheric_path}

    if surface_path.is_file() and atmospheric_path.is_file():
        return paths

    ds = open_hres_t0_store()
    day_slice = ds.sel(time=day)

    try:
        if not surface_path.is_file():
            ds_surf = day_slice[list(WB2_HRES_SURFACE_VARS)].compute()
            ds_surf.to_netcdf(surface_path, engine=NETCDF_ENGINE)

        if not atmospheric_path.is_file():
            ds_atmos = day_slice[list(WB2_HRES_ATMOS_VARS)].compute()
            ds_atmos.to_netcdf(atmospheric_path, engine=NETCDF_ENGINE)
    except Exception as exc:
        raise RuntimeError(
            f"WeatherBench2 download failed for {day}: {sanitize_exception(exc)}"
        ) from None

    return paths
