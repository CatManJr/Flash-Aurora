from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flash_aurora.engine.core.config import SourceProfile

GRIB_IFS_SURF_VARS: tuple[str, ...] = ("2t", "10u", "10v", "msl", "z", "slt", "lsm")
GRIB_IFS_ATMOS_VARS: tuple[str, ...] = ("z", "t", "u", "v", "q")
GRIB_IFS_ATMOS_HOURS: tuple[int, ...] = (0, 6, 12, 18)

SOURCE_CACHE_SUBDIRS: dict[str, str] = {
    "cds_era5": "era5",
    "wb2_hres": "hres_t0",
    "grib_ifs_0.1": "hres_0.1",
    "cams": "cams",
    "wb2_wam": "wave",
}


def cache_subdir(source: SourceProfile) -> str:
    try:
        return SOURCE_CACHE_SUBDIRS[source.name]
    except KeyError as exc:
        raise KeyError(f"No cache layout registered for source {source.name!r}") from exc


def day_token(valid_time: datetime) -> str:
    return valid_time.strftime("%Y-%m-%d")


def hres_01_netcdf_paths(cache_dir: Path, day: str) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "surface": cache_dir / f"{day}-surface-level.nc",
        "atmospheric_00": cache_dir / f"{day}-atmospheric-00.nc",
        "atmospheric_06": cache_dir / f"{day}-atmospheric-06.nc",
    }


def grib_ifs_paths(cache_dir: Path, day: str) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    paths: dict[str, Path] = {}
    for var in GRIB_IFS_SURF_VARS:
        paths[f"surf_{var}"] = cache_dir / f"surf_{var}_{day}.grib"
    for var in GRIB_IFS_ATMOS_VARS:
        for hour in GRIB_IFS_ATMOS_HOURS:
            paths[f"atmos_{var}_{hour:02d}"] = cache_dir / f"atmos_{var}_{day}_{hour:02d}.grib"
    return paths


def hres_01_netcdf_complete(cache_dir: Path, day: str) -> bool:
    return all(path.is_file() for path in hres_01_netcdf_paths(cache_dir, day).values())


def expected_paths(source: SourceProfile, valid_time: datetime, cache_dir: Path) -> dict[str, Path]:
    day = day_token(valid_time)
    cache_dir = Path(cache_dir)

    if source.name in {"cds_era5", "wb2_hres"}:
        return {
            "static": cache_dir / "static.nc",
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-atmospheric.nc",
        }

    if source.name == "wb2_wam":
        wave_grib = cache_dir / f"{day}-wave.grib"
        wave_nc = cache_dir / f"{day}-wave.nc"
        wave = wave_grib if wave_grib.is_file() else wave_nc
        return {
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-atmospheric.nc",
            "wave": wave if wave.is_file() else wave_grib,
        }

    if source.name == "cams":
        return {
            "surface": cache_dir / f"{day}-cams-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-cams-atmospheric.nc",
        }

    if source.name == "grib_ifs_0.1":
        if hres_01_netcdf_complete(cache_dir, day):
            return hres_01_netcdf_paths(cache_dir, day)
        return grib_ifs_paths(cache_dir, day)

    raise KeyError(f"No expected path layout for source {source.name!r}")


def missing_keys(source: SourceProfile, valid_time: datetime, cache_dir: Path) -> tuple[str, ...]:
    paths = expected_paths(source, valid_time, cache_dir)
    return tuple(key for key, path in paths.items() if not path.is_file())
