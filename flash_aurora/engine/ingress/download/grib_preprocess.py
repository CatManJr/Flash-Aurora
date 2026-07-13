from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from flash_aurora.engine.core.config import STANDARD_ATMOS, STANDARD_LEVELS
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.redaction import sanitize_exception
from flash_aurora.engine.ingress.download.layout import (
    grib_ifs_paths,
    hres_01_netcdf_complete,
    hres_01_netcdf_paths,
)
from flash_aurora.engine.ingress.download.paths import normalize_path

CFGRIB_ENGINE = "cfgrib"

GRIB_SURF_FIELDS: tuple[tuple[str, str], ...] = (
    ("2t", "t2m"),
    ("10u", "u10"),
    ("10v", "v10"),
    ("msl", "msl"),
)

_CFGRIB_INSTALL_HINT = (
    "GRIB ingress requires cfgrib plus a discoverable ecCodes native library. "
    "Install with: uv sync  (or: uv pip install cfgrib eccodes ecmwflibs)"
)


def _bootstrap_eccodes_from_ecmwflibs() -> bool:
    """Make findlibs see the hashed libeccodes shipped by ecmwflibs.

    On Linux, the PyPI ``eccodes`` wheel is often pure-Python and does not ship
    ``libeccodes``. ``ecmwflibs`` provides the shared object, but ``findlibs``
    does not resolve that package by default, so cfgrib fails with
    "Cannot find the ecCodes library".
    """
    try:
        import findlibs
    except ImportError:
        return False

    if findlibs.find("eccodes") is not None:
        return True

    try:
        import ecmwflibs
    except ImportError:
        return False

    library_path = ecmwflibs.find("eccodes")
    if not library_path:
        return False

    import ctypes

    lib_dir = Path(library_path).resolve().parent
    # Preload sibling libs first so dlopen of libeccodes can resolve deps.
    for shared_object in sorted(
        lib_dir.glob("lib*.so*"),
        key=lambda p: p.name == Path(library_path).name,
    ):
        try:
            ctypes.CDLL(str(shared_object), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue

    original_find = findlibs.find

    def find_with_ecmwflibs(lib_name: str, pkg_name: str | None = None) -> str | None:
        found = original_find(lib_name, pkg_name)
        if found is not None:
            return found
        bare = lib_name.removeprefix("lib").split(".so", 1)[0].split("-", 1)[0]
        if bare == "eccodes" or lib_name in {"eccodes", "libeccodes", "libeccodes.so"}:
            return library_path
        return None

    findlibs.find = find_with_ecmwflibs  # type: ignore[method-assign]
    return findlibs.find("eccodes") is not None


def require_cfgrib() -> None:
    _bootstrap_eccodes_from_ecmwflibs()
    try:
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise ImportError(_CFGRIB_INSTALL_HINT) from exc
    except RuntimeError as exc:
        # gribapi raises RuntimeError when the native library is missing.
        raise ImportError(
            f"{_CFGRIB_INSTALL_HINT} ({sanitize_exception(exc)})"
        ) from exc

    try:
        import xarray as xr

        if CFGRIB_ENGINE not in xr.backends.list_engines():
            raise ImportError(
                "cfgrib is installed but xarray cannot load the cfgrib engine. "
                f"{_CFGRIB_INSTALL_HINT}"
            )
    except ImportError:
        raise
    except Exception as exc:
        raise ImportError(
            f"cfgrib engine check failed: {sanitize_exception(exc)}. "
            f"{_CFGRIB_INSTALL_HINT}"
        ) from exc


def _level_coord(dataset: xr.Dataset) -> str:
    if "isobaricInhPa" in dataset.coords:
        return "isobaricInhPa"
    if "pressure_level" in dataset.coords:
        return "pressure_level"
    return "level"


def _open_grib(path: Path) -> xr.Dataset:
    require_cfgrib()
    return xr.open_dataset(path, engine=CFGRIB_ENGINE)


def materialize_hres_01_netcdf(
    cache_dir: Path | str,
    day: str,
    *,
    levels: tuple[int | float, ...] = STANDARD_LEVELS,
) -> dict[str, Path]:
    """Convert downloaded UCAR GRIB files into NetCDF cache for hres_0.1 ingress."""
    cache_dir = normalize_path(cache_dir)
    grib_paths = grib_ifs_paths(cache_dir, day)
    nc_paths = hres_01_netcdf_paths(cache_dir, day)
    if hres_01_netcdf_complete(cache_dir, day):
        return nc_paths

    if not grib_paths["surf_2t"].is_file() or not grib_paths["atmos_t_00"].is_file():
        raise FileNotFoundError(
            f"Cannot materialize NetCDF under {cache_dir}: GRIB inputs for {day} are incomplete."
        )

    try:
        if not nc_paths["surface"].is_file():
            with _open_grib(grib_paths["surf_2t"]) as ref_ds:
                time_values = ref_ds["time"].values[:2]
                lat = ref_ds["latitude"].values
                lon = ref_ds["longitude"].values

            surf_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
            for aurora, file_var in GRIB_SURF_FIELDS:
                with _open_grib(grib_paths[f"surf_{aurora}"]) as surf_ds:
                    surf_vars[file_var] = (
                        ("time", "latitude", "longitude"),
                        np.ascontiguousarray(surf_ds[file_var].values[:2]),
                    )

            surface = xr.Dataset(
                data_vars=surf_vars,
                coords={
                    "time": time_values,
                    "latitude": lat,
                    "longitude": lon,
                },
            )
            surface.to_netcdf(nc_paths["surface"], engine=NETCDF_ENGINE)

        for hour, dest_key in ((0, "atmospheric_00"), (6, "atmospheric_06")):
            dest = nc_paths[dest_key]
            if dest.is_file():
                continue
            sample = _open_grib(grib_paths[f"atmos_t_{hour:02d}"])
            with sample:
                level_name = _level_coord(sample)
                lat = sample["latitude"].values
                lon = sample["longitude"].values
                level_values = list(levels)

            frame_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
            for var in STANDARD_ATMOS:
                with _open_grib(grib_paths[f"atmos_{var}_{hour:02d}"]) as atmos_ds:
                    selected = atmos_ds[var].sel({level_name: level_values})
                    frame_vars[var] = (
                        (level_name, "latitude", "longitude"),
                        np.ascontiguousarray(selected.values),
                    )

            atmospheric = xr.Dataset(
                data_vars=frame_vars,
                coords={
                    level_name: level_values,
                    "latitude": lat,
                    "longitude": lon,
                },
            )
            atmospheric.to_netcdf(dest, engine=NETCDF_ENGINE)
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"HRES 0.1 GRIB to NetCDF conversion failed for {day}: {sanitize_exception(exc)}"
        ) from None

    if not hres_01_netcdf_complete(cache_dir, day):
        raise RuntimeError(f"HRES 0.1 NetCDF cache incomplete after conversion for {day}")

    return nc_paths
