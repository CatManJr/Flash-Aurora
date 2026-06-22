from __future__ import annotations

import warnings
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


def test_grib_ifs_expected_paths_use_grib_layout(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    valid_time = datetime(2022, 5, 11, 6)
    cache = tmp_path / "hres_0.1"
    paths = expected_paths(config.source, valid_time, cache)
    assert paths["surf_2t"].name == "surf_2t_2022-05-11.grib"
    assert paths["atmos_t_06"].name == "atmos_t_2022-05-11_06.grib"


def test_missing_hres_01_detects_absent_grib_files(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    valid_time = datetime(2022, 5, 11, 6)
    cache = tmp_path / "hres_0.1"
    cache.mkdir()
    missing = missing_keys(config.source, valid_time, cache)
    assert "surf_2t" in missing
    assert "atmos_t_00" in missing


def test_missing_hres_01_skips_when_netcdf_cache_complete(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    valid_time = datetime(2022, 5, 11, 6)
    cache = tmp_path / "hres_0.1"
    cache.mkdir()
    for name in (
        "2022-05-11-surface-level.nc",
        "2022-05-11-atmospheric-00.nc",
        "2022-05-11-atmospheric-06.nc",
    ):
        (cache / name).write_bytes(b"nc")
    assert missing_keys(config.source, valid_time, cache) == ()


def test_grib_ifs_backend_downloads_when_missing(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    downloader = DataDownloader(config)
    valid_time = datetime(2022, 5, 11, 6)
    cache = tmp_path / "hres_0.1"

    def fake_download(cache_dir: Path, day: str):
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = expected_paths(config.source, valid_time, cache_dir)
        for path in paths.values():
            path.write_bytes(b"grib")
        return paths

    with patch(
        "flash_aurora.engine.ingress.download.backends.grib_ifs.download_ifs_analysis_day",
        side_effect=fake_download,
    ):
        result = downloader.ensure(valid_time, cache_dir=cache)

    assert result.complete
    assert result.downloaded
    assert result.paths["surf_2t"].is_file()


def test_grib_ifs_urls_match_upstream_layout() -> None:
    from flash_aurora.engine.ingress.download.grib_ifs import atmos_grib_url, surf_grib_url

    date = datetime(2022, 5, 11, 6)
    assert surf_grib_url(date, "2t").endswith("ec.oper.an.sfc.128_167_2t.regn1280sc.20220511.grb")
    assert atmos_grib_url(date, "t", 6).endswith("ec.oper.an.pl.128_130_t.regn1280sc.2022051106.grb")
    assert atmos_grib_url(date, "u", 12).endswith("ec.oper.an.pl.128_131_u.regn1280uv.2022051112.grb")


def test_mars_client_accepts_explicit_ecmwf_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash_aurora.engine.ingress.download.credentials import DownloadCredentials, use_download_credentials
    from flash_aurora.engine.ingress.download.mars import mars_service

    monkeypatch.delenv("ECMWF_API_KEY", raising=False)
    with patch("flash_aurora.engine.ingress.download.mars.require_ecmwfapi") as mocked:
        mocked.return_value.ECMWFService.return_value = MagicMock()
        with use_download_credentials(
            DownloadCredentials(ecmwf_api_key="secret-key", ecmwf_email="user@example.com")
        ):
            with mars_service() as client:
                assert client is mocked.return_value.ECMWFService.return_value
        mocked.return_value.ECMWFService.assert_called_once()


def test_fetch_bytes_retries_without_verify_on_ucar_ssl_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from requests.exceptions import SSLError

    import flash_aurora.engine.ingress.download.http as http_module
    from flash_aurora.engine.ingress.download.http import fetch_bytes

    http_module._ucar_insecure_tls = False
    http_module._insecure_warnings_suppressed = False
    url = (
        "https://data.rda.ucar.edu/d113001/ec.oper.an.sfc/202205/"
        "ec.oper.an.sfc.128_167_2t.regn1280sc.20220511.grb"
    )
    calls: list[bool] = []

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.content = payload

        def raise_for_status(self) -> None:
            return None

    def fake_get(target: str, *, timeout: float, verify: bool, stream: bool = False):
        calls.append(verify)
        if verify:
            raise SSLError("certificate has expired")
        return FakeResponse(b"grib")

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("FLASH_AURORA_SSL_VERIFY", "1")

    with pytest.warns(UserWarning, match="UCAR RDA TLS"):
        assert fetch_bytes(url, timeout=10, progress=False) == b"grib"
    assert calls == [True, False]

    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fetch_bytes(url, timeout=10, progress=False) == b"grib"
    assert calls == [False]
