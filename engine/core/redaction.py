from __future__ import annotations

import os
import re
from contextvars import ContextVar
from pathlib import Path

_ephemeral_literals: ContextVar[tuple[str, ...]] = ContextVar(
    "flash_aurora_ephemeral_redaction_literals",
    default=(),
)

SENSITIVE_ENV_SUBSTRINGS: tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "APIKEY",
    "PRIVATE",
    "CREDENTIAL",
    "SAS",
)

SENSITIVE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "FOUNDRY_TOKEN",
        "FOUNDRY_ENDPOINT",
        "BLOB_URL_WITH_SAS",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_STORAGE_CONNECTION_STRING",
        "CDSAPI_KEY",
        "CDSAPI_URL",
        "ECMWF_API_KEY",
        "ECMWF_API_URL",
        "ECMWF_API_EMAIL",
        "AURORA_HF_LOCAL_DIR",
        "FLASH_AURORA_ASSET_ROOT",
    }
)

SECRET_LINE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(key\s*[:=]\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)(token\s*[:=]\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)(password\s*[:=]\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)(email\s*[:=]\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)(sig=)[^&\s\"']+"), r"\1***"),
    (re.compile(r"(?i)([?&]sv=)[^&\s\"']+"), r"\1***"),
    (re.compile(r"(?i)([?&]se=)[^&\s\"']+"), r"\1***"),
    (re.compile(r"(?i)([?&]sp=)[^&\s\"']+"), r"\1***"),
    (re.compile(r"(?i)([?&]spr=)[^&\s\"']+"), r"\1***"),
)


def _is_sensitive_env_key(key: str) -> bool:
    if key in SENSITIVE_ENV_KEYS:
        return True
    upper = key.upper()
    return any(fragment in upper for fragment in SENSITIVE_ENV_SUBSTRINGS)


def _config_secret(path: Path, key_names: tuple[str, ...]) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        for name in key_names:
            prefix = f"{name.lower()}:"
            if lowered.startswith(prefix):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                return value or None
    return None


def _redaction_literals() -> tuple[str, ...]:
    # Longest first so partial overlaps do not leave suffixes behind.
    literals = list(_ephemeral_literals.get())
    literals.extend(
        value
        for key, value in os.environ.items()
        if value and len(value) >= 4 and _is_sensitive_env_key(key)
    )

    from engine.ingress.download.paths import cdsapirc_path, ecmwfapirc_path

    cds_key = _config_secret(cdsapirc_path(), ("key",))
    if cds_key:
        literals.append(cds_key)
    ecmwf_key = _config_secret(ecmwfapirc_path(), ("key",))
    if ecmwf_key:
        literals.append(ecmwf_key)

    return tuple(sorted(set(literals), key=len, reverse=True))


def push_ephemeral_literals(*values: str):
    current = _ephemeral_literals.get()
    added = tuple(value for value in values if value and len(value) >= 4)
    return _ephemeral_literals.set(current + added)


def pop_ephemeral_literals(token) -> None:
    _ephemeral_literals.reset(token)


def redact_text(text: str) -> str:
    """Remove known secret literals and common credential patterns from text."""
    if not text:
        return text
    redacted = text
    for literal in _redaction_literals():
        redacted = redacted.replace(literal, "***")
    for pattern, replacement in SECRET_LINE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def sanitize_exception(exc: BaseException) -> str:
    return redact_text(str(exc))


def safe_path(path: Path | str, *, base: Path | None = None) -> str:
    """Render a path for user-visible logs without exposing the full home directory."""
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        resolved = Path(path).expanduser()

    home = Path.home()
    try:
        if resolved == home:
            return "~"
        relative_home = resolved.relative_to(home)
        return "~/" + relative_home.as_posix()
    except ValueError:
        pass

    if base is not None:
        try:
            return Path(resolved).relative_to(base.expanduser().resolve()).as_posix()
        except (OSError, ValueError):
            pass

    return resolved.name


def safe_config_label(path: Path) -> str:
    """Show only the config filename, never the user's home path."""
    name = path.name
    if name.startswith("."):
        return f"~/{name}"
    return name
