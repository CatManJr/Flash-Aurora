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


def read_cdsapirc_key() -> str | None:
    """Read the API key from ``~/.cdsapirc`` without loading the file URL."""
    path = cdsapirc_path()
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("key:"):
            value = stripped.split(":", 1)[1].strip()
            return value or None
    return None


def ecmwfapirc_path() -> Path:
    return user_config_file(".ecmwfapirc")


def platform_label() -> str:
    return sys.platform

