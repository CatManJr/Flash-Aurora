from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from flash_aurora.engine.core.config import SourceProfile
from flash_aurora.engine.ingress.download import cds, cams, layout, mars, weatherbench2


class DownloadBackendError(NotImplementedError):
    """Raised when a source has no automated downloader yet."""


@dataclass(frozen=True)
class DownloadOutcome:
    paths: dict[str, Path]
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]


class DownloadBackend(Protocol):
    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome: ...


def _partition(keys: tuple[str, ...], before: set[str], after: dict[str, Path]) -> DownloadOutcome:
    downloaded = tuple(key for key in keys if key not in before and after[key].is_file())
    skipped = tuple(key for key in keys if key in before and after[key].is_file())
    return DownloadOutcome(paths=after, downloaded=downloaded, skipped=skipped)


class CdsEra5Backend:
    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys = tuple(layout.expected_paths(source, valid_time, cache_dir))
        before = {key for key in keys if layout.expected_paths(source, valid_time, cache_dir)[key].is_file()}
        paths = cds.download_era5_day(cache_dir, day, include_static=True)
        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class Wb2HresBackend:
    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys = tuple(layout.expected_paths(source, valid_time, cache_dir))
        before = {key for key in keys if layout.expected_paths(source, valid_time, cache_dir)[key].is_file()}

        weatherbench2.download_hres_t0_day(cache_dir, day)
        cds.download_era5_static(cache_dir)

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class Wb2WamBackend:
    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys = tuple(layout.expected_paths(source, valid_time, cache_dir))
        before = {key for key in keys if layout.expected_paths(source, valid_time, cache_dir)[key].is_file()}

        weatherbench2.download_hres_t0_day(cache_dir, day)
        mars.download_wave_grib(cache_dir, day)

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class CamsBackend:
    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome:
        day = layout.day_token(valid_time)
        keys = tuple(layout.expected_paths(source, valid_time, cache_dir))
        before = {key for key in keys if layout.expected_paths(source, valid_time, cache_dir)[key].is_file()}

        cams.download_cams_day(cache_dir, day)

        expected = layout.expected_paths(source, valid_time, cache_dir)
        return _partition(keys, before, expected)


class UnsupportedBackend:
    label: str

    def __init__(self, label: str, *, doc: str) -> None:
        self.label = label
        self._doc = doc

    def ensure(self, source: SourceProfile, valid_time: datetime, cache_dir: Path) -> DownloadOutcome:
        raise DownloadBackendError(
            f"Automatic download for {self.label} is not implemented yet. {self._doc}"
        )


DEFAULT_BACKENDS: dict[str, DownloadBackend] = {
    "cds_era5": CdsEra5Backend(),
    "wb2_hres": Wb2HresBackend(),
    "wb2_wam": Wb2WamBackend(),
    "cams": CamsBackend(),
    "grib_ifs_0.1": UnsupportedBackend(
        "HRES 0.1 GRIB",
        doc="Provide GRIB or NetCDF cache manually (see aurora/docs/example_hres_0.1.ipynb).",
    ),
}


def get_backend(source: SourceProfile) -> DownloadBackend:
    backend = DEFAULT_BACKENDS.get(source.name)
    if backend is None:
        raise KeyError(f"No download backend registered for source {source.name!r}")
    return backend
