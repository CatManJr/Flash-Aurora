"""Default asset directory for local benchmarks (not part of the installed library)."""

from __future__ import annotations

from pathlib import Path


def default_asset_root() -> Path:
    """Return ``<cwd>/assets``, creating the directory if needed."""
    root = (Path.cwd() / "assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
