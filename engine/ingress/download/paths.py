from __future__ import annotations

import os
import sys
from pathlib import Path


def normalize_path(path: Path | str) -> Path:
    """Expand ``~`` and resolve to an absolute path (Linux, macOS, Windows)."""
    return Path(path).expanduser().resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_home() -> Path:
    return Path.home()


def user_config_file(name: str) -> Path:
    """Return a config file under the user home directory."""
    if name.startswith("."):
        return user_home() / name
    return user_home() / name


def cdsapirc_path() -> Path:
    return user_config_file(".cdsapirc")


def ecmwfapirc_path() -> Path:
    return user_config_file(".ecmwfapirc")


def platform_label() -> str:
    return sys.platform


def env_asset_root() -> Path | None:
    for key in ("AURORA_HF_LOCAL_DIR", "FLASH_AURORA_ASSET_ROOT"):
        value = os.environ.get(key)
        if value:
            return normalize_path(value)
    return None
