from __future__ import annotations

from pathlib import Path

from aurora.model.aurora import Aurora

from engine.core.config import EngineConfig
from engine.core.model_registry import ModelFactory
from engine.core.paths import AssetStore


class CheckpointLoader:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)

    def load(self, model: Aurora) -> Path:
        variant = self._config.variant
        path = self._assets.fetch_hub_file(
            variant.checkpoint_filename,
            repo=variant.hf_repo,
            allow_download=self._config.allow_hub_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
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
