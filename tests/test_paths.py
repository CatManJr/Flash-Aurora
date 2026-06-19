from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.paths import AssetStore, FETCHED_DIR_NAME


def test_default_root_is_workdir_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    store = AssetStore(root=None)
    assert store.resolve_root() == (tmp_path / FETCHED_DIR_NAME).resolve()


def test_explicit_root_overrides_default(tmp_path: Path) -> None:
    custom = tmp_path / "weights"
    store = AssetStore(root=None)
    assert store.resolve_root(custom) == custom.resolve()


def test_env_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "env_assets"
    env_dir.mkdir()
    monkeypatch.setenv("AURORA_HF_LOCAL_DIR", str(env_dir))
    store = AssetStore(root=None)
    assert store.resolve_root() == env_dir.resolve()


def test_user_cwd_controls_default_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    user_dir = tmp_path / "project"
    user_dir.mkdir()
    store = AssetStore(root=None)
    assert store.resolve_root(user_cwd=user_dir) == (user_dir / FETCHED_DIR_NAME).resolve()


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
    target = tmp_path / "data" / FETCHED_DIR_NAME
    store = AssetStore(root=target)
    created = store.ensure_root()
    assert created.is_dir()


def test_library_source_has_no_autodl_paths() -> None:
    root = Path(__file__).resolve().parents[1] / "engine"
    forbidden = "/root/autodl-tmp"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            hits.append(str(path.relative_to(root.parent)))
    assert hits == []
