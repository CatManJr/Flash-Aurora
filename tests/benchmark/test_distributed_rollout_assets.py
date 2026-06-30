"""Tests for distributed rollout benchmark asset helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _distributed_rollout_assets import ensure_preset_assets, preset_assets_ready
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.download.layout import cache_subdir


def test_preset_assets_ready_hres_missing_checkpoint(tmp_path: Path) -> None:
    ready, reason = preset_assets_ready("hres_0.1", tmp_path)
    assert ready is False
    assert "missing checkpoint" in reason


def test_preset_assets_ready_hres_complete_netcdf_cache(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    (tmp_path / config.variant.checkpoint_filename).write_bytes(b"ckpt")
    (tmp_path / config.variant.static_pickle).write_bytes(b"static")
    cache = tmp_path / cache_subdir(config.source)
    cache.mkdir()
    for name in (
        "2022-05-11-surface-level.nc",
        "2022-05-11-atmospheric-00.nc",
        "2022-05-11-atmospheric-06.nc",
    ):
        (cache / name).write_bytes(b"nc")

    ready, reason = preset_assets_ready("hres_0.1", tmp_path)
    assert ready is True
    assert reason == ""


def test_ensure_preset_assets_downloads_missing_hres_files(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("hres_0.1")
    cache = tmp_path / cache_subdir(config.source)
    cache.mkdir(parents=True)
    for name in (
        "2022-05-11-surface-level.nc",
        "2022-05-11-atmospheric-00.nc",
        "2022-05-11-atmospheric-06.nc",
    ):
        (cache / name).write_bytes(b"nc")

    store = MagicMock()

    def fake_fetch(filename: str, **kwargs: object) -> Path:
        path = tmp_path / filename
        path.write_bytes(b"asset")
        return path

    store.fetch_hub_file.side_effect = fake_fetch

    downloader = MagicMock()
    downloader.missing.return_value = ()
    downloader.ensure.return_value = MagicMock(downloaded=(), skipped=())

    with (
        patch("_distributed_rollout_assets.AssetStore", return_value=store),
        patch("_distributed_rollout_assets.DataDownloader", return_value=downloader),
        patch("_distributed_rollout_assets.apply_hub_endpoint"),
    ):
        ensure_preset_assets(
            "hres_0.1",
            tmp_path,
            hf_mirror=True,
            prompt=False,
            verbose=False,
        )

    assert store.fetch_hub_file.call_count == 2
    assert (tmp_path / config.variant.checkpoint_filename).is_file()
    assert (tmp_path / config.variant.static_pickle).is_file()


def test_ensure_preset_assets_raises_when_still_incomplete(tmp_path: Path) -> None:
    def fake_fetch(filename: str, **kwargs: object) -> Path:
        path = tmp_path / filename
        path.write_bytes(b"asset")
        return path

    downloader = MagicMock()
    downloader.missing.return_value = ("surf_2t",)
    downloader.ensure.return_value = MagicMock(downloaded=(), skipped=())

    with (
        patch("_distributed_rollout_assets.AssetStore") as store_cls,
        patch("_distributed_rollout_assets.DataDownloader", return_value=downloader),
    ):
        store = store_cls.return_value
        store.fetch_hub_file.side_effect = fake_fetch
        with pytest.raises(FileNotFoundError, match="missing ingress"):
            ensure_preset_assets(
                "hres_0.1",
                tmp_path,
                hf_mirror=False,
                prompt=False,
                verbose=False,
            )
