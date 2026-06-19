from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class MissingAssetRootError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class AssetStore:
    """Resolves checkpoint and pickle paths from config or environment."""

    root: Path | None = None

    @classmethod
    def from_env(cls) -> AssetStore:
        env = os.environ.get("AURORA_HF_LOCAL_DIR") or os.environ.get("FLASH_AURORA_ASSET_ROOT")
        if not env:
            return cls(root=None)
        return cls(root=Path(env).expanduser().resolve())

    def resolve_root(self, explicit: Path | None = None) -> Path:
        if explicit is not None:
            return Path(explicit).expanduser().resolve()
        if self.root is not None:
            return self.root
        raise MissingAssetRootError(
            "Set AURORA_HF_LOCAL_DIR or pass EngineConfig(asset_root=...)."
        )

    def join(self, filename: str, explicit_root: Path | None = None) -> Path:
        return self.resolve_root(explicit_root) / filename
