"""Default asset directory for local benchmarks (not part of the installed library)."""

from __future__ import annotations

import os
from pathlib import Path


def default_asset_root() -> Path:
    """Resolve checkpoint/data root from env or ``./assets`` under the current working directory."""
    for key in ("AURORA_ASSET_ROOT", "AURORA_HF_LOCAL_DIR"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return (Path.cwd() / "assets").resolve()
