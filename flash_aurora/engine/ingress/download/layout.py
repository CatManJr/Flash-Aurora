from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flash_aurora.engine.core.config import SourceProfile

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
        return {
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric_00": cache_dir / f"{day}-atmospheric-00.nc",
            "atmospheric_06": cache_dir / f"{day}-atmospheric-06.nc",
        }

    raise KeyError(f"No expected path layout for source {source.name!r}")


def missing_keys(source: SourceProfile, valid_time: datetime, cache_dir: Path) -> tuple[str, ...]:
    paths = expected_paths(source, valid_time, cache_dir)
    return tuple(key for key, path in paths.items() if not path.is_file())
