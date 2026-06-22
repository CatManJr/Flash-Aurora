"""Copyright (c) Catman Jr. Licensed under the MIT license.

Shared Hugging Face Hub helpers: endpoint resolution and mainland-China mirror fallback.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

_CHINA_TZ_MARKERS = ("Shanghai", "Beijing", "Chongqing", "Urumqi", "Hong_Kong")
_CHINA_LOCALE_PREFIXES = ("zh_CN", "zh_Hans", "zh_SG")
_HF_PROBE_TIMEOUT_S = 2.0


def normalize_hub_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    cleaned = endpoint.strip().rstrip("/")
    return cleaned or None


def _env_override(name: str) -> bool | None:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return None


def _locale_suggests_china() -> bool:
    for var in ("LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES"):
        val = os.environ.get(var, "")
        if any(val.startswith(prefix) for prefix in _CHINA_LOCALE_PREFIXES):
            return True
    return False


def _timezone_suggests_china() -> bool:
    tz = os.environ.get("TZ", "")
    if any(marker in tz for marker in _CHINA_TZ_MARKERS):
        return True
    try:
        if any(marker in name for name in time.tzname for marker in _CHINA_TZ_MARKERS):
            return True
    except Exception:
        pass
    return False


def _huggingface_co_reachable(timeout: float = _HF_PROBE_TIMEOUT_S) -> bool:
    try:
        request = urllib.request.Request(
            DEFAULT_HF_ENDPOINT,
            method="HEAD",
            headers={"User-Agent": "flash-aurora"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@lru_cache(maxsize=1)
def detect_mainland_china() -> bool:
    """Best-effort guess whether Hub traffic should use the China mirror."""
    override = _env_override("AURORA_AUTO_HF_MIRROR")
    if override is not None:
        return override

    if _locale_suggests_china() or _timezone_suggests_china():
        return True

    return not _huggingface_co_reachable()


def _sync_hf_constants(endpoint: str) -> None:
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            f"{endpoint}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
        )
    except ImportError:
        pass


def _apply_xet_policy(endpoint: str) -> None:
    # Mirror hosts do not support Xet CAS; force regular HTTP downloads.
    if endpoint == HF_MIRROR_ENDPOINT:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def resolve_hub_endpoint(explicit: str | None = None) -> str:
    """Resolve Hub API base URL: explicit -> ``HF_ENDPOINT`` -> auto mirror -> official HF."""
    if explicit is not None:
        normalized = normalize_hub_endpoint(explicit)
        if normalized is not None:
            return normalized
    env = normalize_hub_endpoint(os.environ.get("HF_ENDPOINT"))
    if env is not None:
        return env
    if detect_mainland_china():
        return HF_MIRROR_ENDPOINT
    return DEFAULT_HF_ENDPOINT


def apply_hub_endpoint(endpoint: str | None = None) -> str:
    """Set ``HF_ENDPOINT`` (and mirror-related env) before Hub downloads."""
    resolved = resolve_hub_endpoint(endpoint)
    os.environ["HF_ENDPOINT"] = resolved
    _apply_xet_policy(resolved)
    _sync_hf_constants(resolved)
    return resolved


def hf_hub_download(*args: Any, endpoint: str | None = None, **kwargs: Any) -> str:
    """``huggingface_hub.hf_hub_download`` with automatic endpoint / Xet configuration."""
    apply_hub_endpoint(endpoint)
    from huggingface_hub import hf_hub_download as _hf_hub_download

    return _hf_hub_download(*args, **kwargs)


def download_hub_file(
    filename: str,
    *,
    repo: str,
    local_dir: Path | str,
    revision: str | None = None,
    endpoint: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a single file from Hugging Face Hub (or a mirror) into ``local_dir``."""
    kwargs: dict[str, str] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token
    downloaded = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(local_dir),
        endpoint=endpoint,
        **kwargs,
    )
    return Path(downloaded).resolve()
