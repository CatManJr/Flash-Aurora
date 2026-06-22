from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from engine.core.redaction import pop_ephemeral_literals, push_ephemeral_literals

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class HubDownloadOptions:
    """Optional Hugging Face Hub settings for weight and static file downloads."""

    endpoint: str | None = None
    revision: str | None = None
    token: str | None = None


def normalize_hub_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    cleaned = endpoint.strip().rstrip("/")
    return cleaned or None


def resolve_hub_endpoint(explicit: str | None = None) -> str:
    """Resolve Hub API base URL: explicit arg -> ``HF_ENDPOINT`` env -> official HF."""
    if explicit is not None:
        normalized = normalize_hub_endpoint(explicit)
        if normalized is not None:
            return normalized
    env = normalize_hub_endpoint(os.environ.get("HF_ENDPOINT"))
    if env is not None:
        return env
    return DEFAULT_HF_ENDPOINT


def apply_hub_endpoint(endpoint: str | None = None) -> str:
    """Set ``HF_ENDPOINT`` before importing ``huggingface_hub`` (required for mirrors)."""
    resolved = resolve_hub_endpoint(endpoint)
    os.environ["HF_ENDPOINT"] = resolved
    return resolved


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
        from huggingface_hub import hf_hub_download

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
