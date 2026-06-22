from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from flash_aurora.aurora import Batch, Metadata

from flash_aurora.engine.core.config import EngineConfig, WAVE_SURF_WAM
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters.base import resolve_cache_dir
from flash_aurora.engine.ingress.adapters.era5 import CdsEra5Adapter
from flash_aurora.engine.ingress.adapters.hres_t0 import (
    WB2_HRES_ATMOS_FIELDS,
    WB2_HRES_SURF_FIELDS,
)
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.builder import TimeHistoryBuilder
from flash_aurora.engine.ingress.static import StaticFieldLoader


@dataclass(frozen=True)
class WavePaths:
    surface: Path
    atmospheric: Path
    wave: Path


WB2_WAVE_FIELDS: tuple[str, ...] = WAVE_SURF_WAM


class Wb2WamWaveAdapter:
    """Build a Batch from HRES T0 NetCDF plus WAM wave GRIB (example_wave.ipynb)."""

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        for path in (paths.surface, paths.atmospheric, paths.wave):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing wave scenario input {path}. "
                    "Download from WeatherBench2/ECMWF or set IngestRequest.raw_paths."
                )

        open_kwargs = {"engine": NETCDF_ENGINE, "decode_timedelta": True}
        with xr.open_dataset(paths.surface, **open_kwargs) as surf_ds, xr.open_dataset(
            paths.atmospheric, **open_kwargs
        ) as atmos_ds:
            wave_ds = self._open_wave_dataset(paths.wave)
            try:
                batch = self._build_batch(
                    surf_ds=surf_ds,
                    atmos_ds=atmos_ds,
                    wave_ds=wave_ds,
                    config=config,
                )
            finally:
                if hasattr(wave_ds, "close"):
                    wave_ds.close()

        assets = AssetStore(root=config.asset_root)
        batch.static_vars = StaticFieldLoader(config, assets).load()
        return batch

    def _open_wave_dataset(self, path: Path) -> xr.Dataset:
        if path.suffix.lower() == ".grib":
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        return xr.open_dataset(path, engine=NETCDF_ENGINE)

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> WavePaths:
        if request.raw_paths:
            try:
                return WavePaths(
                    surface=Path(request.raw_paths["surface"]).expanduser().resolve(),
                    atmospheric=Path(request.raw_paths["atmospheric"]).expanduser().resolve(),
                    wave=Path(request.raw_paths["wave"]).expanduser().resolve(),
                )
            except KeyError as exc:
                raise ValueError(
                    "raw_paths must include 'surface', 'atmospheric', and 'wave'"
                ) from exc

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "wave")
        wave_grib = cache_dir / f"{day}-wave.grib"
        wave_nc = cache_dir / f"{day}-wave.nc"
        wave = wave_grib if wave_grib.is_file() else wave_nc
        return WavePaths(
            surface=cache_dir / f"{day}-surface-level.nc",
            atmospheric=cache_dir / f"{day}-atmospheric.nc",
            wave=wave,
        )

    def _build_batch(
        self,
        *,
        surf_ds: xr.Dataset,
        atmos_ds: xr.Dataset,
        wave_ds: xr.Dataset,
        config: EngineConfig,
    ) -> Batch:
        if config.source.time_policy != "first_two":
            raise ValueError(
                f"Wave adapter requires source.time_policy='first_two', "
                f"got {config.source.time_policy!r}"
            )
        if not config.source.flip_lat:
            raise ValueError("Wave adapter requires source.flip_lat=True")

        history = TimeHistoryBuilder(config.source)

        surf_vars = {
            aurora: history.build_surf_history(surf_ds[wb2].values)
            for wb2, aurora in WB2_HRES_SURF_FIELDS
        }
        for name in WB2_WAVE_FIELDS:
            surf_vars[name] = history.build_wave_history(wave_ds[name].values)

        atmos_vars = {
            aurora: history.build_atmos_history(atmos_ds[wb2].values)
            for wb2, aurora in WB2_HRES_ATMOS_FIELDS
        }

        times = surf_ds.time.values.astype("datetime64[s]").tolist()
        if len(times) < 2:
            raise ValueError(f"Wave adapter requires at least two time steps, got {len(times)}")

        level_coord = atmos_ds["level"] if "level" in atmos_ds.coords else atmos_ds["pressure_level"]

        return Batch(
            surf_vars=surf_vars,
            static_vars={},
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=CdsEra5Adapter._coord_tensor(surf_ds.latitude.values[::-1]),
                lon=CdsEra5Adapter._coord_tensor(surf_ds.longitude.values),
                time=(times[1],),
                atmos_levels=tuple(int(level) for level in level_coord.values),
                rollout_step=0,
            ),
        )
