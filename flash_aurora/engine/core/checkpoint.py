from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.model_registry import ModelFactory
from flash_aurora.engine.core.paths import AssetStore, normalize_asset_path
from flash_aurora.engine.core.redaction import redact_text, safe_path


class CheckpointLoader:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._assets = AssetStore(root=config.asset_root)

    def load(self, model: Aurora) -> Path:
        variant = self._config.variant
        hub = self._config.hub_download_options()
        explicit = self._config.checkpoint_path
        if explicit is not None:
            path = normalize_asset_path(explicit)
            if path.is_file():
                model.load_checkpoint_local(str(path), strict=variant.strict_checkpoint)
                return path
            if not self._config.allow_hub_download:
                raise FileNotFoundError(
                    redact_text(
                        f"Missing checkpoint at {safe_path(path)}. "
                        "Pass a valid checkpoint_path or enable allow_hub_download."
                    )
                )

        path = self._assets.fetch_hub_file(
            variant.checkpoint_filename,
            repo=variant.hf_repo,
            allow_download=self._config.allow_hub_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
            hub=hub,
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
