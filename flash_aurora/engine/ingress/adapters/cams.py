from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from flash_aurora.aurora import Batch, Metadata

from flash_aurora.engine.core.config import (
    CAMS_ATMOS_POLLUTION,
    CAMS_SURF_POLLUTION,
    EngineConfig,
    STANDARD_ATMOS,
    STANDARD_SURF,
)
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters.base import resolve_cache_dir
from flash_aurora.engine.ingress.adapters.era5 import CdsEra5Adapter
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.static import StaticFieldLoader


@dataclass(frozen=True)
class CamsPaths:
    surface: Path
    atmospheric: Path


CAMS_SURF_STANDARD_FIELDS: tuple[tuple[str, str], ...] = (
    ("t2m", "2t"),
    ("u10", "10u"),
    ("v10", "10v"),
    ("msl", "msl"),
)

CAMS_SURF_POLLUTION_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (name, name) for name in CAMS_SURF_POLLUTION
)

CAMS_ATMOS_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (name, name) for name in STANDARD_ATMOS + CAMS_ATMOS_POLLUTION
)


class CamsAdapter:
    """Build a Batch from cached CAMS NetCDF files (example_cams.ipynb)."""

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        for path in (paths.surface, paths.atmospheric):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing CAMS input {path}. Download from ADS or set IngestRequest.raw_paths."
                )

        open_kwargs = {"engine": NETCDF_ENGINE, "decode_timedelta": True}
        with xr.open_dataset(paths.surface, **open_kwargs) as surf_ds, xr.open_dataset(
            paths.atmospheric, **open_kwargs
        ) as atmos_ds:
            if "forecast_period" in surf_ds.dims:
                surf_ds = surf_ds.isel(forecast_period=0)
            if "forecast_period" in atmos_ds.dims:
                atmos_ds = atmos_ds.isel(forecast_period=0)
            batch = self._build_batch(surf_ds, atmos_ds, config)

        assets = AssetStore(root=config.asset_root)
        batch.static_vars = StaticFieldLoader(config, assets).load()
        return batch

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> CamsPaths:
        if request.raw_paths:
            try:
                return CamsPaths(
                    surface=Path(request.raw_paths["surface"]).expanduser().resolve(),
                    atmospheric=Path(request.raw_paths["atmospheric"]).expanduser().resolve(),
                )
            except KeyError as exc:
                raise ValueError(
                    "raw_paths must include 'surface' and 'atmospheric'"
                ) from exc

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "cams")
        return CamsPaths(
            surface=cache_dir / f"{day}-cams-surface-level.nc",
            atmospheric=cache_dir / f"{day}-cams-atmospheric.nc",
        )

    def _build_batch(
        self,
        surf_ds: xr.Dataset,
        atmos_ds: xr.Dataset,
        config: EngineConfig,
    ) -> Batch:
        surf_vars = {
            aurora: torch.from_numpy(np.ascontiguousarray(surf_ds[file_name].values)[None])
            for file_name, aurora in CAMS_SURF_STANDARD_FIELDS + CAMS_SURF_POLLUTION_FIELDS
        }
        atmos_vars = {
            aurora: torch.from_numpy(np.ascontiguousarray(atmos_ds[name].values)[None])
            for name, aurora in CAMS_ATMOS_FIELDS
        }

        valid_times = atmos_ds.valid_time.values.astype("datetime64[s]").tolist()
        level_coord = (
            atmos_ds["pressure_level"]
            if "pressure_level" in atmos_ds.coords
            else atmos_ds["level"]
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars={},
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=CdsEra5Adapter._coord_tensor(atmos_ds.latitude.values),
                lon=CdsEra5Adapter._coord_tensor(atmos_ds.longitude.values),
                time=(valid_times[-1],),
                atmos_levels=tuple(int(level) for level in level_coord.values),
                rollout_step=0,
            ),
        )
