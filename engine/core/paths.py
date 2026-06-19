from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FETCHED_DIR_NAME = "fetched"

ENV_ASSET_ROOT_KEYS = ("AURORA_HF_LOCAL_DIR", "FLASH_AURORA_ASSET_ROOT")


@dataclass(frozen=True)
class AssetStore:
    """Resolve weight and data files under a user-controlled root.

    Priority: explicit asset_root, store root, environment variable,
    then ``{user_cwd}/fetched``. user_cwd is the caller working directory,
    usually captured as Path.cwd() when AuroraEngine is created.
    """

    root: Path | None = None

    def resolve_root(
        self,
        explicit: Path | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        if explicit is not None:
            return Path(explicit).expanduser().resolve()
        if self.root is not None:
            return self.root.expanduser().resolve()
        for key in ENV_ASSET_ROOT_KEYS:
            env = os.environ.get(key)
            if env:
                return Path(env).expanduser().resolve()
        base = (user_cwd or Path.cwd()).expanduser().resolve()
        return base / FETCHED_DIR_NAME

    def ensure_root(
        self,
        explicit: Path | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        root = self.resolve_root(explicit, user_cwd)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def join(
        self,
        filename: str,
        explicit: Path | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        return self.resolve_root(explicit, user_cwd) / filename

    def fetch_hub_file(
        self,
        filename: str,
        *,
        repo: str,
        allow_download: bool,
        explicit: Path | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        root = self.ensure_root(explicit, user_cwd)
        local = root / filename
        if local.is_file():
            return local
        if not allow_download:
            raise FileNotFoundError(
                f"Missing {filename!r} under {root}. "
                "Place the file there, set asset_root, or enable allow_hub_download."
            )
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(root),
        )
        return Path(downloaded).resolve()
