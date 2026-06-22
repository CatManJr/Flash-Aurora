from __future__ import annotations

from pathlib import Path

import pytest

from flash_aurora.engine.core.paths import AssetRootRequiredError, AssetStore, normalize_asset_path, normalize_user_path


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


def test_normalize_user_path_uses_user_cwd(tmp_path: Path) -> None:
    base = tmp_path / "notebook"
    base.mkdir()
    resolved = normalize_user_path("assets", user_cwd=base)
    assert resolved == (base / "assets").resolve()


def _is_tutorial_notebook(path: Path, repo_root: Path) -> bool:
    """Example notebooks may pin team data-disk paths for saved run outputs."""
    try:
        rel = path.relative_to(repo_root / "docs")
    except ValueError:
        return False
    return rel.name.startswith("example_") and path.suffix == ".ipynb"


def test_library_source_has_no_autodl_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = "/root/autodl-tmp"
    scan_roots = (
        root / "flash_aurora",
        root / "benchmark",
        root / "scripts",
        root / "profiling",
        root / "tests",
        root / "docs",
    )
    generator = root / "scripts" / "generate_example_notebooks.py"
    hits: list[str] = []
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in scan_root.rglob("*"):
            if path.suffix not in {".py", ".md", ".sh", ".ipynb"}:
                continue
            if any(part in {".venv", "__pycache__", "node_modules"} for part in path.parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.resolve() == generator.resolve():
                continue
            if _is_tutorial_notebook(path, root):
                continue
            text = path.read_text(encoding="utf-8")
            if forbidden in text:
                hits.append(str(path.relative_to(root)))
    assert hits == []


def test_tutorial_notebooks_default_to_portable_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    era5 = root / "docs" / "example_era5.ipynb"
    assert era5.is_file()
    text = era5.read_text(encoding="utf-8")
    assert "ASSET_ROOT: Path | str | None = None" in text
    assert "# ASSET_ROOT = Path(" in text
