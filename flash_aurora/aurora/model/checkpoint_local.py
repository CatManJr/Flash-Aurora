"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Resolve Aurora checkpoints from a local directory (e.g. AutoDL data disk).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from flash_aurora.aurora.model.aurora import Aurora

DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("AURORA_ASSET_ROOT", os.environ.get("AURORA_HF_LOCAL_DIR", ""))
    or "."
).expanduser()


def resolve_checkpoint_path(
    *,
    filename: str,
    checkpoint_dir: str | Path | None = None,
    explicit_path: str | Path | None = None,
    repo: str | None = None,
    revision: str | None = None,
    allow_hub_download: bool = True,
) -> Path:
    """Return a local checkpoint path, preferring ``checkpoint_dir/filename``.

    If the file is missing and ``allow_hub_download`` is True, download into
    ``checkpoint_dir`` via Hugging Face Hub.
    """
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    base = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR).expanduser().resolve()
    local = base / filename
    if local.is_file():
        return local

    if not allow_hub_download:
        raise FileNotFoundError(
            f"Checkpoint {filename!r} not found under {base}. "
            "Place weights there or pass an explicit path."
        )

    from flash_aurora.hub import hf_hub_download

    repo = repo or "microsoft/aurora"
    downloaded = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        local_dir=str(base),
    )
    return Path(downloaded).resolve()


def load_aurora_checkpoint_prefer_local(
    model: Aurora,
    *,
    checkpoint_dir: str | Path | None = None,
    path: str | Path | None = None,
    strict: bool = True,
    allow_hub_download: bool = True,
) -> Path:
    """Load weights for ``model``, preferring ``checkpoint_dir`` over Hub streaming."""
    ckpt_path = resolve_checkpoint_path(
        filename=model.default_checkpoint_name,
        checkpoint_dir=checkpoint_dir,
        explicit_path=path,
        repo=model.default_checkpoint_repo,
        revision=model.default_checkpoint_revision,
        allow_hub_download=allow_hub_download,
    )
    model.load_checkpoint_local(str(ckpt_path), strict=strict)
    return ckpt_path
