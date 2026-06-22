from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora import Batch

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters import IngestRequest, get_adapter
from flash_aurora.engine.ingress.deserialize import BatchDeserializer
from flash_aurora.engine.ingress.static import StaticFieldLoader
from flash_aurora.engine.ingress.validator import BatchValidator


class InitialConditionBuilder:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)
        self._static = StaticFieldLoader(config, self._assets)
        self._validator = BatchValidator(config.variant)

    def _allowed_roots(self) -> tuple[Path, ...]:
        return self._assets.allowed_roots(self._config.asset_root, self._config.user_cwd)

    def _fetch_input(self, filename: str) -> Path:
        return self._assets.fetch_hub_file(
            filename,
            repo=self._config.variant.hf_repo,
            allow_download=self._config.allow_hub_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
            hub=self._config.hub_download_options(),
        )

    def _with_static(self, batch: Batch) -> Batch:
        static = self._static.load(lat=batch.metadata.lat, lon=batch.metadata.lon)
        return Batch(
            surf_vars=batch.surf_vars,
            static_vars=static,
            atmos_vars=batch.atmos_vars,
            metadata=batch.metadata,
        )

    def from_pickle(self, filename: str) -> Batch:
        path = self._fetch_input(filename)
        batch = BatchDeserializer.from_pickle(path, allowed_roots=self._allowed_roots())
        return self._with_static(batch)

    def from_netcdf_path(self, path: Path) -> Batch:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"NetCDF not found: {resolved}")
        roots = self._allowed_roots()
        if resolved.parent not in roots:
            roots = roots + (resolved.parent,)
        batch = BatchDeserializer.from_netcdf(resolved, allowed_roots=roots)
        return self._with_static(batch)

    def from_netcdf(self, filename: str) -> Batch:
        path = self._assets.join(
            filename,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
        )
        return self.from_netcdf_path(path)

    def from_source(self, request: IngestRequest) -> Batch:
        adapter = get_adapter(self._config.source)
        batch = adapter.build_initial_batch(request, self._config)
        self._validator.validate(batch)
        return batch
