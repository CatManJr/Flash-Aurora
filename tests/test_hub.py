from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from engine.core.hub import (
    DEFAULT_HF_ENDPOINT,
    HF_MIRROR_ENDPOINT,
    apply_hub_endpoint,
    download_hub_file,
    normalize_hub_endpoint,
    resolve_hub_endpoint,
)
from engine.core.paths import AssetStore


def test_normalize_hub_endpoint_strips_trailing_slash() -> None:
    assert normalize_hub_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"


def test_resolve_hub_endpoint_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://example.com")
    assert resolve_hub_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"


def test_apply_hub_endpoint_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert apply_hub_endpoint(HF_MIRROR_ENDPOINT) == HF_MIRROR_ENDPOINT
    assert resolve_hub_endpoint() == HF_MIRROR_ENDPOINT


def test_fetch_hub_file_uses_mirror_endpoint(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    with patch("engine.core.paths.download_hub_file") as mocked:
        mocked.return_value = tmp_path / "aurora-0.25-pretrained.ckpt"
        store.fetch_hub_file(
            "aurora-0.25-pretrained.ckpt",
            repo="microsoft/aurora",
            allow_download=True,
            explicit=tmp_path,
            hub=__import__("engine.core.hub", fromlist=["HubDownloadOptions"]).HubDownloadOptions(
                endpoint=HF_MIRROR_ENDPOINT,
            ),
        )
    mocked.assert_called_once()
    assert mocked.call_args.kwargs["endpoint"] == HF_MIRROR_ENDPOINT


def test_download_hub_file_calls_hf_hub_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    with patch("huggingface_hub.hf_hub_download") as mocked:
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
