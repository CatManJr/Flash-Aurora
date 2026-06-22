from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

from flash_aurora.engine.ingress.download.http import fetch_bytes
from flash_aurora.engine.ingress.download.progress import download_progress_enabled
from flash_aurora.engine.ingress.download.layout import (
    GRIB_IFS_ATMOS_HOURS,
    GRIB_IFS_ATMOS_VARS,
    GRIB_IFS_SURF_VARS,
    grib_ifs_paths,
    hres_01_netcdf_complete,
)
from flash_aurora.engine.ingress.download.paths import ensure_directory, normalize_path

UCAR_RDA_BASE = "https://data.rda.ucar.edu/d113001"

VAR_NUMS: dict[str, str] = {
    "2t": "167",
    "10u": "165",
    "10v": "166",
    "msl": "151",
    "t": "130",
    "u": "131",
    "v": "132",
    "q": "133",
    "z": "129",
    "slt": "043",
    "lsm": "172",
}


def surf_grib_url(date: datetime, var: str) -> str:
    var_num = VAR_NUMS[var]
    y, m, d = date.year, date.month, date.day
    return (
        f"{UCAR_RDA_BASE}/ec.oper.an.sfc/{y}{m:02d}/"
        f"ec.oper.an.sfc.128_{var_num}_{var}.regn1280sc.{y}{m:02d}{d:02d}.grb"
    )


def atmos_grib_url(date: datetime, var: str, hour: int) -> str:
    var_num = VAR_NUMS[var]
    prefix = "uv" if var in {"u", "v"} else "sc"
    y, m, d = date.year, date.month, date.day
    return (
        f"{UCAR_RDA_BASE}/ec.oper.an.pl/{y}{m:02d}/"
        f"ec.oper.an.pl.128_{var_num}_{var}.regn1280{prefix}.{y}{m:02d}{d:02d}{hour:02d}.grb"
    )


def iter_grib_downloads(date: datetime, cache_dir: Path) -> tuple[tuple[Path, str], ...]:
    day = date.strftime("%Y-%m-%d")
    items: list[tuple[Path, str]] = []
    for var in GRIB_IFS_SURF_VARS:
        path = grib_ifs_paths(cache_dir, day)[f"surf_{var}"]
        items.append((path, surf_grib_url(date, var)))
    for var in GRIB_IFS_ATMOS_VARS:
        for hour in GRIB_IFS_ATMOS_HOURS:
            key = f"atmos_{var}_{hour:02d}"
            path = grib_ifs_paths(cache_dir, day)[key]
            items.append((path, atmos_grib_url(date, var, hour)))
    return tuple(items)


def download_ifs_analysis_day(cache_dir: Path | str, day: str) -> dict[str, Path]:
    """Download IFS HRES 0.1° analysis GRIB files from UCAR RDA (example_hres_0.1.ipynb)."""
    cache_dir = normalize_path(cache_dir)
    ensure_directory(cache_dir)
    date = datetime.strptime(day, "%Y-%m-%d")

    all_items = iter_grib_downloads(date, cache_dir)
    pending = tuple((path, url) for path, url in all_items if not path.is_file())
    skipped = len(all_items) - len(pending)

    show_progress = download_progress_enabled()
    iterator: Iterable[tuple[Path, str]] = pending
    if show_progress and pending:
        from tqdm.auto import tqdm

        iterator = tqdm(
            pending,
            desc="UCAR GRIB files",
            unit="file",
            initial=0,
            total=len(all_items),
        )
        if skipped:
            iterator.set_postfix_str(f"skipped {skipped} cached")
            iterator.update(skipped)
    elif skipped and show_progress:
        print(f"UCAR GRIB: all {skipped} files already cached")

    for path, url in iterator:
        try:
            content = fetch_bytes(url, label=path.name, progress=show_progress)
        except RuntimeError as exc:
            raise RuntimeError(
                f"UCAR RDA download failed for {path.name}: {exc}"
            ) from None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    result_paths = grib_ifs_paths(cache_dir, day)
    if all(path.is_file() for path in result_paths.values()) and not hres_01_netcdf_complete(
        cache_dir, day
    ):
        try:
            from flash_aurora.engine.ingress.download.grib_preprocess import materialize_hres_01_netcdf

            materialize_hres_01_netcdf(cache_dir, day)
        except ImportError as exc:
            warnings.warn(
                f"GRIB download finished but NetCDF preprocess was skipped: {exc}. "
                "Install cfgrib before building the initial condition.",
                stacklevel=2,
            )

    return result_paths
