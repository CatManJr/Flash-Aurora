from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.checkpoint import CheckpointLoader
from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.hooks import RolloutObserver
from flash_aurora.engine.core.hub import HF_MIRROR_ENDPOINT
from flash_aurora.engine.core.paths import AssetStore, normalize_asset_path, normalize_user_path
from flash_aurora.engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from flash_aurora.engine.core.rollout_session import RolloutSession
from flash_aurora.engine.egress.export import RolloutExporter
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.adapters import IngestRequest
from flash_aurora.engine.ingress.validator import BatchValidator
from flash_aurora.engine.runtime.graph_pool import GraphPool


class AuroraEngine:
    def __init__(
        self,
        config: EngineConfig,
        presets: PresetRegistry | None = None,
    ) -> None:
        self.config = config
        if self.config.user_cwd is None:
            self.config.user_cwd = Path.cwd()
        self._presets = presets or DEFAULT_PRESETS
        self._model: Aurora | None = None
        self._loader = CheckpointLoader(config)
        self._validator = BatchValidator(config.variant)
        self._graph_pool = GraphPool()
        self._exporter = RolloutExporter(self._resolved_export_dir())

    def _resolved_export_dir(self) -> Path:
        return normalize_user_path(
            self.config.export_dir,
            user_cwd=self.config.user_cwd,
        )

    def set_export_dir(self, export_dir: Path | str) -> Path:
        """Update the rollout export directory and refresh the exporter."""
        resolved = normalize_user_path(export_dir, user_cwd=self.config.user_cwd)
        self.config.export_dir = resolved
        self._exporter = RolloutExporter(resolved)
        return resolved

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        asset_root: Path | str,
        checkpoint_path: Path | str | None = None,
        allow_hub_download: bool | None = None,
        hf_endpoint: str | None = None,
        hf_mirror: bool = False,
        hf_revision: str | None = None,
        hf_token: str | None = None,
        export_dir: Path | str | None = None,
        inference_precision: str | None = None,
        presets: PresetRegistry | None = None,
    ) -> AuroraEngine:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        user_cwd = Path.cwd()
        config.user_cwd = user_cwd
        config.asset_root = normalize_user_path(asset_root, user_cwd=user_cwd)
        if checkpoint_path is not None:
            config.checkpoint_path = normalize_user_path(checkpoint_path, user_cwd=user_cwd)
        if export_dir is not None:
            config.export_dir = normalize_user_path(export_dir, user_cwd=user_cwd)
        if allow_hub_download is not None:
            config.allow_hub_download = allow_hub_download
        if hf_mirror:
            config.hf_endpoint = HF_MIRROR_ENDPOINT
        elif hf_endpoint is not None:
            config.hf_endpoint = hf_endpoint
        if hf_revision is not None:
            config.hf_revision = hf_revision
        if hf_token is not None:
            config.hf_token = hf_token
        if inference_precision is not None:
            config.inference_precision = inference_precision
        return cls(config, presets=registry)

    @property
    def fetched_dir(self) -> Path:
        store = AssetStore(root=self.config.asset_root)
        return store.resolve_root(self.config.asset_root, self.config.user_cwd)

    def _allowed_roots(self) -> tuple[Path, ...]:
        return AssetStore(root=self.config.asset_root).allowed_roots(
            self.config.asset_root,
            self.config.user_cwd,
        )

    def _builder(self) -> InitialConditionBuilder:
        return InitialConditionBuilder(self.config)

    @property
    def model(self) -> Aurora:
        if self._model is None:
            raise RuntimeError("Call load() before using the model.")
        return self._model

    def load(self) -> Aurora:
        model = self._loader.build_model()
        self._loader.load(model)
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        model.to(device)
        self._model = model
        return model

    def warmup(self) -> None:
        self._graph_pool.warmup(self.model)

    def predict(self, batch: Batch) -> Batch:
        self.validate(batch)
        with torch.inference_mode():
            return self.model.forward(batch)

    def run_from_netcdf(self, path: Path | str, steps: int = 1) -> list[Batch]:
        batch = self._builder().from_netcdf_path(Path(path))
        if steps == 1:
            return [self.predict(batch)]
        return list(self.rollout_stream(batch, steps))

    def run_from_adapter(self, request: IngestRequest, steps: int = 1) -> list[Batch]:
        batch = self._builder().from_source(request)
        if steps == 1:
            return [self.predict(batch)]
        return list(self.rollout_stream(batch, steps))

    def validate(self, batch: Batch) -> None:
        self._validator.validate(batch)

    def rollout_stream(
        self,
        batch: Batch,
        steps: int,
        observers: Iterable[RolloutObserver] | None = None,
    ) -> Generator[Batch, None, None]:
        self.validate(batch)
        session = RolloutSession(self.model, observers)
        yield from session.run(batch, steps)

    def rollout_and_export(
        self,
        batch: Batch,
        steps: int,
        *,
        export_dir: Path | str | None = None,
    ) -> Generator[Path, None, None]:
        if export_dir is not None:
            self.set_export_dir(export_dir)
        else:
            self.set_export_dir(self.config.export_dir)
        for step_index, prediction in enumerate(self.rollout_stream(batch, steps)):
            yield self._exporter.write_step(step_index, prediction)
