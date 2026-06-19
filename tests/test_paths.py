from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.paths import AssetStore, MissingAssetRootError


def test_resolve_root_requires_explicit_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    store = AssetStore(root=None)
    with pytest.raises(MissingAssetRootError):
        store.resolve_root()


def test_resolve_root_from_explicit(tmp_path: Path) -> None:
    store = AssetStore(root=None)
    assert store.resolve_root(tmp_path) == tmp_path.resolve()


def test_resolve_root_from_store() -> None:
    store = AssetStore(root=Path("/tmp/assets"))
    assert store.resolve_root() == Path("/tmp/assets").resolve()


def test_library_source_has_no_autodl_paths() -> None:
    root = Path(__file__).resolve().parents[1] / "engine"
    forbidden = "/root/autodl-tmp"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            hits.append(str(path.relative_to(root.parent)))
    assert hits == []
