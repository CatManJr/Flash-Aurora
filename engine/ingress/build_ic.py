from __future__ import annotations

from pathlib import Path

from aurora import Batch

from engine.core.config import EngineConfig
from engine.core.paths import AssetStore
from engine.ingress.deserialize import BatchDeserializer
from engine.ingress.static import StaticFieldLoader


class InitialConditionBuilder:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)
        self._static = StaticFieldLoader(config, self._assets)

    def _fetch_input(self, filename: str) -> Path:
        return self._assets.fetch_hub_file(
            filename,
            repo=self._config.variant.hf_repo,
            allow_download=self._config.allow_hub_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
        )

    def from_pickle(self, filename: str) -> Batch:
        path = self._fetch_input(filename)
        batch = BatchDeserializer.from_pickle(path)
        static = self._static.load()
        return Batch(
            surf_vars=batch.surf_vars,
            static_vars=static,
            atmos_vars=batch.atmos_vars,
            metadata=batch.metadata,
        )

    def from_netcdf(self, filename: str) -> Batch:
        path = self._assets.join(
            filename,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
        )
        if not path.is_file():
            raise FileNotFoundError(f"NetCDF not found: {path}")
        return BatchDeserializer.from_netcdf(path)
