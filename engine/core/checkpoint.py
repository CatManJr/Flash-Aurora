from __future__ import annotations

from pathlib import Path

from aurora.model.aurora import Aurora
from aurora.model.checkpoint_local import resolve_checkpoint_path

from engine.core.config import EngineConfig, ModelVariantSpec
from engine.core.model_registry import ModelFactory
from engine.core.paths import AssetStore


class CheckpointLoader:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)

    def load(self, model: Aurora) -> Path:
        variant = self._config.variant
        root = self._assets.resolve_root(self._config.asset_root)
        path = resolve_checkpoint_path(
            filename=variant.checkpoint_filename,
            checkpoint_dir=root,
            repo=variant.hf_repo,
            allow_hub_download=self._config.allow_hub_download,
        )
        model.load_checkpoint_local(str(path), strict=variant.strict_checkpoint)
        return path

    def build_model(self) -> Aurora:
        variant = self._config.variant
        model = ModelFactory.create(
            variant.model_class,
            use_lora=variant.use_lora,
            lora_mode=variant.lora_mode,
        )
        model.eval()
        return model
