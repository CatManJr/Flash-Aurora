from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable

import torch
from aurora import Batch
from aurora.model.aurora import Aurora

from engine.core.checkpoint import CheckpointLoader
from engine.core.config import EngineConfig
from engine.core.hooks import RolloutObserver
from engine.core.hub import HF_MIRROR_ENDPOINT
from engine.core.paths import AssetStore, normalize_asset_path
from engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from engine.core.rollout_session import RolloutSession
from engine.egress.export import RolloutExporter
from engine.ingress.build_ic import InitialConditionBuilder
from engine.ingress.adapters import IngestRequest
from engine.ingress.validator import BatchValidator
from engine.runtime.graph_pool import GraphPool


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
        self._exporter = RolloutExporter(config.export_dir)

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
        presets: PresetRegistry | None = None,
    ) -> AuroraEngine:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        config.asset_root = normalize_asset_path(asset_root)
        if checkpoint_path is not None:
            config.checkpoint_path = normalize_asset_path(checkpoint_path)
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
    ) -> Generator[Path, None, None]:
        for step_index, prediction in enumerate(self.rollout_stream(batch, steps)):
            yield self._exporter.write_step(step_index, prediction)
