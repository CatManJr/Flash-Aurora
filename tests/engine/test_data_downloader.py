from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.download.downloader import DataDownloader, DownloadRequest
from flash_aurora.engine.ingress.download.layout import expected_paths, missing_keys
from flash_aurora.engine.ingress.download.paths import normalize_path, user_config_file


def test_normalize_path_expands_user(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "era5"
    nested.mkdir(parents=True)
    assert normalize_path(nested) == nested.resolve()


def test_user_config_file_is_under_home() -> None:
    path = user_config_file(".cdsapirc")
    assert path.name == ".cdsapirc"
    assert path.parent == Path.home()


def test_expected_paths_era5_layout(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    valid_time = datetime(2023, 1, 1, 6)
    paths = expected_paths(config.source, valid_time, tmp_path / "era5")
    assert paths["surface"].name == "2023-01-01-surface-level.nc"
    assert paths["static"].name == "static.nc"


def test_missing_detects_absent_files(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"
    cache.mkdir()
    missing = missing_keys(config.source, valid_time, cache)
    assert missing == ("static", "surface", "atmospheric")


def test_resolve_cache_dir_uses_preset_subdir(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config)
    cache = downloader.resolve_cache_dir()
    assert cache == (tmp_path / "era5").resolve()


def test_resolve_cache_dir_honours_explicit_override(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    downloader = DataDownloader(config)
    custom = tmp_path / "custom-era5"
    cache = downloader.resolve_cache_dir(DownloadRequest(valid_time=datetime(2023, 1, 1, 6), cache_dir=custom))
    assert cache == custom.resolve()


def test_ensure_era5_calls_cds_backends(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"

    def fake_download(cache_dir: Path, day: str, *, include_static: bool = True):
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
        result = downloader.ensure(valid_time, cache_dir=cache)

    assert result.complete
    assert set(result.paths) == {"static", "surface", "atmospheric"}
    assert result.downloaded == ("static", "surface", "atmospheric")


def test_ensure_skips_existing_files(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    downloader = DataDownloader(config)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"
    cache.mkdir()
    for name in ("static.nc", "2023-01-01-surface-level.nc", "2023-01-01-atmospheric.nc"):
        (cache / name).write_bytes(b"nc")

    with patch("flash_aurora.engine.ingress.download.backends.cds.download_era5_day") as mocked:
        result = downloader.ensure(valid_time, cache_dir=cache)

    mocked.assert_not_called()
    assert result.skipped == ("static", "surface", "atmospheric")
    assert result.downloaded == ()


def test_ingest_request_downloads_when_missing(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    downloader = DataDownloader(config)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"

    with patch.object(downloader, "ensure") as ensure_mock:
        ensure_mock.return_value = MagicMock()
        request = downloader.ingest_request(valid_time, cache_dir=cache, time_index=1)

    ensure_mock.assert_called_once()
    assert request.cache_dir == cache.resolve()
    assert request.time_index == 1


def test_cams_backend_downloads_when_missing(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("cams")
    downloader = DataDownloader(config)
    valid_time = datetime(2022, 6, 11, 12)
    cache = tmp_path / "cams"

    def fake_download(cache_dir: Path, day: str):
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "surface": cache_dir / f"{day}-cams-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-cams-atmospheric.nc",
        }
        for path in paths.values():
            path.write_bytes(b"nc")
        return paths

    with patch("flash_aurora.engine.ingress.download.backends.cams.download_cams_day", side_effect=fake_download):
        result = downloader.ensure(valid_time, cache_dir=cache, ads_api_key="abc12345")

    assert result.complete
    assert set(result.paths) == {"surface", "atmospheric"}
    assert result.downloaded == ("surface", "atmospheric")


def test_cams_backend_skips_existing_files(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("cams")
    downloader = DataDownloader(config)
    valid_time = datetime(2022, 6, 11, 12)
    cache = tmp_path / "cams"
    cache.mkdir()
    for name in ("2022-06-11-cams-surface-level.nc", "2022-06-11-cams-atmospheric.nc"):
        (cache / name).write_bytes(b"nc")

    with patch("flash_aurora.engine.ingress.download.backends.cams.download_cams_day") as mocked:
        result = downloader.ensure(valid_time, cache_dir=cache)

    mocked.assert_not_called()
    assert result.skipped == ("surface", "atmospheric")
    assert result.downloaded == ()


def test_ads_client_requires_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.ads import ads_client

    monkeypatch.delenv("ADSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("flash_aurora.engine.ingress.download.paths.user_home", lambda: fake_home)
    with pytest.raises(FileNotFoundError, match="Missing ADS credentials"):
        ads_client()

    (fake_home / ".cdsapirc").write_text("url: https://cds.example\nkey: test-key\n")
    with patch("flash_aurora.engine.ingress.download.ads.require_cdsapi") as mocked:
        mocked.return_value.Client.return_value = MagicMock()
        client = ads_client()
    mocked.return_value.Client.assert_called_once_with(
        url="https://ads.atmosphere.copernicus.eu/api",
        key="test-key",
    )
    assert client is mocked.return_value.Client.return_value


def test_ads_client_accepts_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.ads import ads_client
    from flash_aurora.engine.ingress.download.credentials import DownloadCredentials, use_download_credentials

    monkeypatch.delenv("ADSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with patch("flash_aurora.engine.ingress.download.ads.require_cdsapi") as mocked:
        mocked.return_value.Client.return_value = MagicMock()
        with use_download_credentials(DownloadCredentials(ads_api_key="super-secret-key")):
            ads_client()
        mocked.return_value.Client.assert_called_once_with(
            url="https://ads.atmosphere.copernicus.eu/api",
            key="super-secret-key",
        )


def test_ads_client_falls_back_to_cds_key_with_ads_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.ads import ads_client
    from flash_aurora.engine.ingress.download.credentials import DownloadCredentials, use_download_credentials

    monkeypatch.delenv("ADSAPI_KEY", raising=False)
    with patch("flash_aurora.engine.ingress.download.ads.require_cdsapi") as mocked:
        mocked.return_value.Client.return_value = MagicMock()
        with use_download_credentials(DownloadCredentials(cds_api_key="shared-copernicus-key")):
            ads_client()
        mocked.return_value.Client.assert_called_once_with(
            url="https://ads.atmosphere.copernicus.eu/api",
            key="shared-copernicus-key",
        )


def test_read_cdsapirc_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.paths import read_cdsapirc_key

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("flash_aurora.engine.ingress.download.paths.user_home", lambda: fake_home)
    assert read_cdsapirc_key() is None

    (fake_home / ".cdsapirc").write_text("url: https://cds.climate.copernicus.eu/api\nkey: my-key\n")
    assert read_cdsapirc_key() == "my-key"


def test_cds_client_requires_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.cds import cds_client

    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("flash_aurora.engine.ingress.download.paths.user_home", lambda: fake_home)
    with pytest.raises(FileNotFoundError, match="Missing CDS credentials"):
        cds_client()

    (fake_home / ".cdsapirc").write_text("url: https://example\nkey: test\n")
    with patch("flash_aurora.engine.ingress.download.cds.require_cdsapi") as mocked:
        mocked.return_value.Client.return_value = MagicMock()
        client = cds_client()
    assert client is mocked.return_value.Client.return_value


def test_cds_client_accepts_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.cds import cds_client
    from flash_aurora.engine.ingress.download.credentials import DownloadCredentials, use_download_credentials

    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with patch("flash_aurora.engine.ingress.download.cds.require_cdsapi") as mocked:
        mocked.return_value.Client.return_value = MagicMock()
        with use_download_credentials(DownloadCredentials(cds_api_key="super-secret-key")):
            cds_client()
        mocked.return_value.Client.assert_called_once_with(
            url="https://cds.climate.copernicus.eu/api",
            key="super-secret-key",
        )


def test_ensure_accepts_cds_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    config.asset_root = tmp_path
    downloader = DataDownloader(config)
    valid_time = datetime(2023, 1, 1, 6)
    cache = tmp_path / "era5"

    def fake_download(cache_dir: Path, day: str, *, include_static: bool = True):
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "static": cache_dir / "static.nc",
            "surface": cache_dir / f"{day}-surface-level.nc",
            "atmospheric": cache_dir / f"{day}-atmospheric.nc",
        }
        for path in paths.values():
            path.write_bytes(b"nc")
        return paths

    with patch("flash_aurora.engine.ingress.download.backends.cds.download_era5_day", side_effect=fake_download) as mocked:
        downloader.ensure(valid_time, cache_dir=cache, cds_api_key="abc12345")

    mocked.assert_called_once()
