from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable

import torch
from aurora import Batch
from aurora.model.aurora import Aurora

from engine.core.checkpoint import CheckpointLoader
from engine.core.config import EngineConfig
from engine.core.hooks import RolloutObserver
from engine.core.paths import AssetStore
from engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from engine.core.rollout_session import RolloutSession
from engine.egress.export import RolloutExporter
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
        asset_root: Path | None = None,
        allow_hub_download: bool | None = None,
        presets: PresetRegistry | None = None,
    ) -> AuroraEngine:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        if asset_root is not None:
            config.asset_root = asset_root
        if allow_hub_download is not None:
            config.allow_hub_download = allow_hub_download
        return cls(config, presets=registry)

    @property
    def fetched_dir(self) -> Path:
        store = AssetStore(root=self.config.asset_root)
        return store.resolve_root(self.config.asset_root, self.config.user_cwd)

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
