"""Resolve the local asset root from environment variables (data disk, not repo cwd)."""

from __future__ import annotations

import os
from pathlib import Path

_ASSET_ROOT_ENV_KEYS: tuple[str, ...] = (
    "AURORA_ASSET_ROOT",
    "AURORA_HF_LOCAL_DIR",
    "FLASH_AURORA_ASSET_ROOT",
)


class RelativeAssetRootError(ValueError):
    """Raised when ``asset_root`` is relative and would resolve under process cwd."""


def resolve_asset_root() -> Path | None:
    """Return the configured asset root, or ``None`` when unset."""
    for key in _ASSET_ROOT_ENV_KEYS:
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return None


def default_asset_root() -> Path:
    """Return the asset root or raise with setup instructions.

    Assets (checkpoints, ingress cache, guard state) should live on a data disk.
    Set ``AURORA_ASSET_ROOT`` to that directory; do not rely on ``./fetched`` or
    ``./assets`` under the repository on the system drive.
    """
    root = resolve_asset_root()
    if root is None:
        keys = ", ".join(_ASSET_ROOT_ENV_KEYS)
        raise RuntimeError(
            f"Asset root is not configured. Export one of ({keys}) pointing at your "
            "data-disk asset directory, for example:\n"
            "  export AURORA_ASSET_ROOT=/path/to/data/aurora"
        )
    return root


def normalize_asset_root(
    asset_root: Path | str | None,
    *,
    user_cwd: Path | None = None,
) -> Path:
    """Resolve checkpoints, ingress cache, and hub downloads under one absolute root.

    When ``asset_root`` is omitted, read ``AURORA_ASSET_ROOT`` (or legacy aliases).
    Explicit paths must be absolute so a repo checkout on the system drive cannot
    silently become the download target via ``Path.cwd()``.
    """
    del user_cwd  # kept for call-site compatibility; relative paths are rejected
    if asset_root is None:
        return default_asset_root()
    expanded = Path(asset_root).expanduser()
    if not expanded.is_absolute():
        keys = ", ".join(_ASSET_ROOT_ENV_KEYS)
        raise RelativeAssetRootError(
            f"asset_root must be an absolute path, got {asset_root!r}. "
            f"Pass asset_root='/path/to/data/aurora' or export one of ({keys})."
        )
    return expanded.resolve()
