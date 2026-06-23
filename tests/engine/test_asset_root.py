from __future__ import annotations

import pytest

from flash_aurora.engine.core.asset_root import (
    RelativeAssetRootError,
    default_asset_root,
    normalize_asset_root,
    resolve_asset_root,
)


def test_resolve_asset_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AURORA_ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    assert resolve_asset_root() == tmp_path.resolve()


def test_default_asset_root_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AURORA_ASSET_ROOT", "AURORA_HF_LOCAL_DIR", "FLASH_AURORA_ASSET_ROOT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="AURORA_ASSET_ROOT"):
        default_asset_root()


def test_normalize_asset_root_uses_env_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AURORA_ASSET_ROOT", str(tmp_path))
    assert normalize_asset_root(None) == tmp_path.resolve()


def test_normalize_asset_root_requires_absolute_path(tmp_path) -> None:
    with pytest.raises(RelativeAssetRootError, match="absolute path"):
        normalize_asset_root("assets")


def test_normalize_asset_root_accepts_absolute_path(tmp_path) -> None:
    root = tmp_path / "data" / "aurora"
    assert normalize_asset_root(root) == root.resolve()
