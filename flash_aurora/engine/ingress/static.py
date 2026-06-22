from __future__ import annotations

import numpy as np
import torch

from flash_aurora.aurora.batch import interpolate_numpy

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.core.trusted_pickle import load_trusted_pickle


class StaticFieldLoader:
    def __init__(self, config: EngineConfig, assets: AssetStore) -> None:
        self._config = config
        self._assets = assets
        self._variant = config.variant

    def load(
        self,
        *,
        lat: torch.Tensor | None = None,
        lon: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        roots = self._assets.allowed_roots(self._config.asset_root, self._config.user_cwd)
        local = self._assets.join(
            self._variant.static_pickle,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
        )
        # Ingress NetCDF may be cached locally while HF static pickles are still missing
        # (common on partial asset mirrors). Fetch missing pickles even when the
        # checkpoint was resolved locally with allow_hub_download=False.
        allow_download = self._config.allow_hub_download or not local.is_file()
        path = self._assets.fetch_hub_file(
            self._variant.static_pickle,
            repo=self._variant.hf_repo,
            allow_download=allow_download,
            explicit=self._config.asset_root,
            user_cwd=self._config.user_cwd,
            hub=self._config.hub_download_options(),
        )
        payload = load_trusted_pickle(path, roots)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected static pickle dict, got {type(payload)!r}")
        raw: dict[str, np.ndarray] = payload

        if lat is None or lon is None:
            return {name: torch.from_numpy(np.asarray(raw[name])) for name in self._variant.static_vars}

        lat_np = lat.detach().cpu().numpy()
        lon_np = lon.detach().cpu().numpy()
        source_lat = np.linspace(90, -90, raw[next(iter(raw))].shape[0])
        source_lon = np.linspace(0, 360, raw[next(iter(raw))].shape[1], endpoint=False)
        return {
            name: torch.from_numpy(
                interpolate_numpy(raw[name], source_lat, source_lon, lat_np, lon_np)
            )
            for name in self._variant.static_vars
        }
