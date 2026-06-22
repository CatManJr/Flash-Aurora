from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from flash_aurora.engine.core.hub import (
    DEFAULT_HF_ENDPOINT,
    HF_MIRROR_ENDPOINT,
    HubDownloadOptions,
    apply_hub_endpoint,
    detect_mainland_china,
    download_hub_file,
    normalize_hub_endpoint,
    resolve_hub_endpoint,
)
from flash_aurora.engine.core.paths import AssetStore


def test_normalize_hub_endpoint_strips_trailing_slash() -> None:
    assert normalize_hub_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"


def test_resolve_hub_endpoint_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_mainland_china.cache_clear()
    monkeypatch.setenv("HF_ENDPOINT", "https://example.com")
    assert resolve_hub_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"


def test_resolve_hub_endpoint_auto_mirror_in_china(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_mainland_china.cache_clear()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("AURORA_AUTO_HF_MIRROR", "1")
    assert resolve_hub_endpoint() == HF_MIRROR_ENDPOINT


def test_apply_hub_endpoint_disables_xet_for_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_mainland_china.cache_clear()
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    apply_hub_endpoint(HF_MIRROR_ENDPOINT)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_apply_hub_endpoint_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert apply_hub_endpoint(HF_MIRROR_ENDPOINT) == HF_MIRROR_ENDPOINT
    assert resolve_hub_endpoint() == HF_MIRROR_ENDPOINT


def test_fetch_hub_file_uses_mirror_endpoint(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    with patch("flash_aurora.engine.core.paths.download_hub_file") as mocked:
        mocked.return_value = tmp_path / "aurora-0.25-pretrained.ckpt"
        store.fetch_hub_file(
            "aurora-0.25-pretrained.ckpt",
            repo="microsoft/aurora",
            allow_download=True,
            explicit=tmp_path,
            hub=HubDownloadOptions(
                endpoint=HF_MIRROR_ENDPOINT,
            ),
        )
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["endpoint"] == HF_MIRROR_ENDPOINT


def test_download_hub_file_calls_hf_hub_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detect_mainland_china.cache_clear()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    with patch("flash_aurora.engine.core.hub.hf_hub_download") as mocked:
        mocked.return_value = str(tmp_path / "weights.ckpt")
        path = download_hub_file(
            "weights.ckpt",
            repo="microsoft/aurora",
            local_dir=tmp_path,
            endpoint=DEFAULT_HF_ENDPOINT,
        )
    assert path == (tmp_path / "weights.ckpt").resolve()
    assert resolve_hub_endpoint() == DEFAULT_HF_ENDPOINT
    mocked.assert_called_once_with(
        repo_id="microsoft/aurora",
        filename="weights.ckpt",
        local_dir=str(tmp_path),
    )
