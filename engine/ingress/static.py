from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from engine.core.config import ModelVariantSpec
from engine.core.paths import AssetStore


class StaticFieldLoader:
    def __init__(self, variant: ModelVariantSpec, assets: AssetStore) -> None:
        self._variant = variant
        self._assets = assets

    def load(self, asset_root: Path | None = None) -> dict[str, torch.Tensor]:
        path = self._assets.join(self._variant.static_pickle, asset_root)
        if not path.is_file():
            raise FileNotFoundError(f"Static pickle not found: {path}")
        with open(path, "rb") as handle:
            raw: dict[str, np.ndarray] = pickle.load(handle)
        selected = {name: torch.from_numpy(raw[name]) for name in self._variant.static_vars}
        return selected
