from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from flash_aurora.engine.core.config import SourceProfile
from flash_aurora.engine.ingress.download import cds, cams, grib_ifs, layout, mars, weatherbench2
from flash_aurora.engine.ingress.download.parallel import run_labeled_tasks


class DownloadBackendError(NotImplementedError):
    """Raised when a source has no automated downloader yet."""


@dataclass(frozen=True)
class DownloadOutcome:
    paths: dict[str, Path]
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]


class DownloadBackend(Protocol):
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome: ...


def _partition(keys: tuple[str, ...], before: set[str], after: dict[str, Path]) -> DownloadOutcome:
    downloaded = tuple(key for key in keys if key not in before and after[key].is_file())
    skipped = tuple(key for key in keys if key in before and after[key].is_file())
    return DownloadOutcome(paths=after, downloaded=downloaded, skipped=skipped)


def _keys_and_before(
    source: SourceProfile,
    valid_time: datetime,
    cache_dir: Path,
) -> tuple[tuple[str, ...], set[str]]:
    keys = tuple(layout.expected_paths(source, valid_time, cache_dir))
    before = {key for key in keys if layout.expected_paths(source, valid_time, cache_dir)[key].is_file()}
    return keys, before


class CdsEra5Backend:
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys, before = _keys_and_before(source, valid_time, cache_dir)
        cds.download_era5_day(cache_dir, day, include_static=True, workers=workers)
        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class Wb2HresBackend:
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys, before = _keys_and_before(source, valid_time, cache_dir)

        run_labeled_tasks(
            (
                ("hres", lambda: weatherbench2.download_hres_t0_day(cache_dir, day, workers=workers)),
                ("static", lambda: cds.download_era5_static(cache_dir)),
            ),
            workers=min(workers, 2),
            description="WB2 HRES + static",
        )

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class Wb2WamBackend:
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys, before = _keys_and_before(source, valid_time, cache_dir)

        run_labeled_tasks(
            (
                ("hres", lambda: weatherbench2.download_hres_t0_day(cache_dir, day, workers=workers)),
                ("wave", lambda: mars.download_wave_grib(cache_dir, day)),
            ),
            workers=min(workers, 2),
            description="WB2 HRES + MARS wave",
        )

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class CamsBackend:
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys, before = _keys_and_before(source, valid_time, cache_dir)
        cams.download_cams_day(cache_dir, day)
        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class GribIfsBackend:
    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys, before = _keys_and_before(source, valid_time, cache_dir)

        if not layout.hres_01_netcdf_complete(cache_dir, day):
            grib_ifs.download_ifs_analysis_day(cache_dir, day, workers=workers)

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class UnsupportedBackend:
    label: str

    def __init__(self, label: str, *, doc: str) -> None:
        self.label = label
        self._doc = doc

    def ensure(
        self,
        source: SourceProfile,
        valid_time: datetime,
        cache_dir: Path,
        *,
        workers: int = 1,
    ) -> DownloadOutcome:
        raise DownloadBackendError(
            f"Automatic download for {self.label} is not implemented yet. {self._doc}"
        )


DEFAULT_BACKENDS: dict[str, DownloadBackend] = {
    "cds_era5": CdsEra5Backend(),
    "wb2_hres": Wb2HresBackend(),
    "wb2_wam": Wb2WamBackend(),
    "cams": CamsBackend(),
    "grib_ifs_0.1": GribIfsBackend(),
}


def get_backend(source: SourceProfile) -> DownloadBackend:
    backend = DEFAULT_BACKENDS.get(source.name)
    if backend is None:
        raise KeyError(f"No download backend registered for source {source.name!r}")
    return backend
