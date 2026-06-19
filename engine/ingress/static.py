from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from engine.core.config import EngineConfig
from engine.core.paths import AssetStore


class StaticFieldLoader:
    def __init__(self, config: EngineConfig, assets: AssetStore) -> None:
        self._config = config
        self._assets = assets
        self._variant = config.variant

    def load(self) -> dict[str, torch.Tensor]:
        path = self._assets.fetch_hub_file(
            self._variant.static_pickle,
            repo=self._variant.hf_repo,
            allow_download=self._config.allow_hub_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
        )
        with open(path, "rb") as handle:
            raw: dict[str, np.ndarray] = pickle.load(handle)
        return {name: torch.from_numpy(raw[name]) for name in self._variant.static_vars}
