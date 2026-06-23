"""Default asset directory for local benchmarks (not part of the installed library)."""

from __future__ import annotations

import os
from pathlib import Path


def default_asset_root() -> Path:
    """Return the benchmark asset directory.

    Resolution order:

    1. ``AURORA_ASSET_ROOT`` (same as ``flash_aurora`` checkpoint lookup)
    2. ``AURORA_HF_LOCAL_DIR``
    3. ``<cwd>/assets`` (created if missing)
    """
    for key in ("AURORA_ASSET_ROOT", "AURORA_HF_LOCAL_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    root = (Path.cwd() / "assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
