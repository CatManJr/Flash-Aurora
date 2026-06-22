from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flash_aurora.engine.core.redaction import pop_ephemeral_literals, push_ephemeral_literals
from flash_aurora.hub import (
    DEFAULT_HF_ENDPOINT,
    HF_MIRROR_ENDPOINT,
    apply_hub_endpoint,
    detect_mainland_china,
    hf_hub_download,
    normalize_hub_endpoint,
    resolve_hub_endpoint,
)

__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "HF_MIRROR_ENDPOINT",
    "HubDownloadOptions",
    "apply_hub_endpoint",
    "detect_mainland_china",
    "download_hub_file",
    "hf_hub_download",
    "normalize_hub_endpoint",
    "resolve_hub_endpoint",
]


@dataclass(frozen=True)
class HubDownloadOptions:
    """Optional Hugging Face Hub settings for weight and static file downloads."""

    endpoint: str | None = None
    revision: str | None = None
    token: str | None = None


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
    apply_hub_endpoint(endpoint)
    redaction_token = push_ephemeral_literals(token) if token else None
    try:
        kwargs: dict[str, str] = {}
        if revision is not None:
            kwargs["revision"] = revision
        if token is not None:
            kwargs["token"] = token
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local_dir),
            **kwargs,
        )
        return Path(downloaded).resolve()
    finally:
        if redaction_token is not None:
            pop_ephemeral_literals(redaction_token)
