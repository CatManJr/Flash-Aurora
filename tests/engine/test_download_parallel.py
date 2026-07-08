from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.download.downloader import DataDownloader
from flash_aurora.engine.ingress.download.grib_ifs import iter_grib_downloads
from flash_aurora.engine.ingress.download.options import DownloadOptions, default_download_workers
from flash_aurora.engine.ingress.download.parallel import run_labeled_tasks


def test_default_download_workers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASH_AURORA_DOWNLOAD_WORKERS", "8")
    assert default_download_workers() == 8

    monkeypatch.setenv("FLASH_AURORA_DOWNLOAD_WORKERS", "0")
    assert default_download_workers() == 1


def test_download_options_resolve() -> None:
    assert DownloadOptions.resolve(3).workers == 3
    assert DownloadOptions.resolve(None).workers >= 1


def test_run_labeled_tasks_sequential_preserves_order() -> None:
    calls: list[str] = []

    def make(label: str):
        def _fn() -> str:
            calls.append(label)
            return label

        return _fn

    results = run_labeled_tasks(
        (("a", make("a")), ("b", make("b")), ("c", make("c"))),
        workers=1,
    )
    assert results == {"a": "a", "b": "b", "c": "c"}
    assert calls == ["a", "b", "c"]


def test_run_labeled_tasks_propagates_download_credentials() -> None:
    from flash_aurora.engine.ingress.download.credentials import (
        DownloadCredentials,
        active_download_credentials,
        use_download_credentials,
    )

    creds = DownloadCredentials(cds_api_key="parallel-secret")

    def _read_active() -> str:
        active = active_download_credentials()
        if active is None or active.cds_settings() is None:
            raise RuntimeError("missing active creds in worker thread")
        return active.cds_settings()[1]

    with use_download_credentials(creds):
        out = run_labeled_tasks(tuple((f"t{i}", _read_active) for i in range(4)), workers=4)
    assert all(value == "parallel-secret" for value in out.values())


def test_run_labeled_tasks_parallel_runs_all() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def make(label: str):
        def _fn() -> str:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return label

        return _fn

    tasks = tuple((f"t{i}", make(f"t{i}")) for i in range(6))
    results = run_labeled_tasks(tasks, workers=4)
    assert len(results) == 6
    assert peak >= 2


def test_grib_ifs_parallel_downloads(tmp_path: Path) -> None:
    from datetime import datetime

    from flash_aurora.engine.ingress.download import grib_ifs

    date = datetime(2022, 5, 11, 6)
    day = "2022-05-11"
    calls: list[str] = []

    def fake_fetch(url: str, *, timeout: float = 120, progress: bool | None = None, label: str | None = None) -> bytes:
        calls.append(label or url)
        time.sleep(0.02)
        return b"grib"

    with patch("flash_aurora.engine.ingress.download.grib_ifs.fetch_bytes", side_effect=fake_fetch):
        with patch(
            "flash_aurora.engine.ingress.download.grib_preprocess.materialize_hres_01_netcdf",
        ):
            grib_ifs.download_ifs_analysis_day(tmp_path, day, workers=4)

    expected_count = len(iter_grib_downloads(date, tmp_path))
    assert len(calls) == expected_count


def test_data_downloader_workers_override(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config, workers=2)
    assert downloader.download_workers == 2
    assert downloader.with_workers(6).download_workers == 6


def test_ensure_cds_api_key_with_parallel_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    from flash_aurora.engine.ingress.download.cds import cds_client
    from flash_aurora.engine.ingress.download.credentials import active_download_credentials

    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config, workers=8)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"

    def fake_download(cache_dir: Path, day: str, *, include_static: bool = True, workers: int = 1):
        def _touch(path: Path) -> None:
            active = active_download_credentials()
            if active is None or active.cds_settings() is None:
                cds_client()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"nc")

        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "static": cache_dir / "static.nc",
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-atmospheric.nc",
        }

        tasks = tuple((key, lambda p=path: _touch(p)) for key, path in paths.items())
        run_labeled_tasks(tasks, workers=workers)
        return paths

    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with patch("flash_aurora.engine.ingress.download.backends.cds.download_era5_day", side_effect=fake_download):
        with patch("flash_aurora.engine.ingress.download.cds.require_cdsapi") as mocked:
            mocked.return_value.Client.return_value = MagicMock()
            downloader.ensure(valid_time, cache_dir=cache, cds_api_key="abc12345")


def test_ensure_passes_workers_to_backend(tmp_path: Path) -> None:
    from datetime import datetime

    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config, workers=2)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"

    def fake_download(cache_dir: Path, day: str, *, include_static: bool = True, workers: int = 1):
        assert workers == 5
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "static": cache_dir / "static.nc",
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-atmospheric.nc",
        }
        for path in paths.values():
            path.write_bytes(b"nc")
        return paths

    with patch("flash_aurora.engine.ingress.download.backends.cds.download_era5_day", side_effect=fake_download):
        downloader.ensure(valid_time, cache_dir=cache, workers=5, cds_api_key="test-key")
