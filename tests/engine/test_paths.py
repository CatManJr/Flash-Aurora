from __future__ import annotations

from pathlib import Path

import pytest

from flash_aurora.engine.core.paths import AssetRootRequiredError, AssetStore, normalize_asset_path


def test_normalize_asset_path_expands_user(tmp_path: Path) -> None:
    nested = tmp_path / "assets"
    nested.mkdir()
    assert normalize_asset_path(nested) == nested.resolve()


def test_resolve_root_requires_explicit_path() -> None:
    store = AssetStore(root=None)
    with pytest.raises(AssetRootRequiredError):
        store.resolve_root()


def test_explicit_root_is_normalized(tmp_path: Path) -> None:
    custom = tmp_path / "weights"
    store = AssetStore(root=None)
    assert store.resolve_root(custom) == custom.resolve()


def test_store_root_is_used_when_explicit_missing(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path / "assets")
    assert store.resolve_root() == (tmp_path / "assets").resolve()


def test_fetch_hub_file_requires_local_file_when_download_disabled(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.fetch_hub_file(
            "missing.ckpt",
            repo="microsoft/aurora",
            allow_download=False,
            explicit=tmp_path,
        )


def test_ensure_root_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "data" / "assets"
    store = AssetStore(root=target)
    created = store.ensure_root()
    assert created.is_dir()


def test_library_source_has_no_autodl_paths() -> None:
    root = Path(__file__).resolve().parents[2] / "flash_aurora"
    forbidden = "/root/autodl-tmp"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            hits.append(str(path.relative_to(root)))
    assert hits == []
