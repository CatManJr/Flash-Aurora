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


def read_ecmwfapirc() -> dict[str, str] | None:
    """Parse ``~/.ecmwfapirc`` JSON (``url``, ``key``, ``email``).

    Returns ``None`` when the file is missing, unreadable, or incomplete.
    """
    path = ecmwfapirc_path()
    if not path.is_file():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, str] = {}
    for field in ("url", "key", "email"):
        value = payload.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parsed[field] = stripped
    if "key" in parsed and "email" in parsed:
        return parsed
    return None


def platform_label() -> str:
    return sys.platform

