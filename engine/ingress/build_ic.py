from __future__ import annotations

from aurora import Batch

from engine.core.config import EngineConfig
from engine.core.paths import AssetStore
from engine.ingress.deserialize import BatchDeserializer
from engine.ingress.static import StaticFieldLoader


class InitialConditionBuilder:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)
        self._static = StaticFieldLoader(config.variant, self._assets)

    def from_pickle(self, filename: str) -> Batch:
        path = self._assets.join(filename, self._config.asset_root)
        batch = BatchDeserializer.from_pickle(path)
        static = self._static.load(self._config.asset_root)
        return Batch(
            surf_vars=batch.surf_vars,
            static_vars=static,
            atmos_vars=batch.atmos_vars,
            metadata=batch.metadata,
        )

    def from_netcdf(self, filename: str) -> Batch:
        path = self._assets.join(filename, self._config.asset_root)
        return BatchDeserializer.from_netcdf(path)
