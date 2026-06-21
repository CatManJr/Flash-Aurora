from __future__ import annotations

from typing import Protocol

from aurora import Batch

from engine.core.config import EngineConfig
from engine.core.paths import AssetStore
from engine.ingress.adapters.request import IngestRequest


class DataSourceAdapter(Protocol):
    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch: ...


def resolve_cache_dir(request: IngestRequest, config: EngineConfig, subdir: str) -> Path:
    if request.cache_dir is not None:
        return request.cache_dir.expanduser().resolve()
    store = AssetStore(root=config.asset_root)
    return store.resolve_root(config.asset_root, config.user_cwd) / subdir


class StubAdapter:
    label: str = "adapter"

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        raise NotImplementedError(f"{self.label} is not implemented yet.")
