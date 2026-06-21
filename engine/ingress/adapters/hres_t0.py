from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import Batch, Metadata

from engine.core.config import EngineConfig, STANDARD_STATIC
from engine.core.netcdf_codec import NETCDF_ENGINE
from engine.ingress.adapters.base import resolve_cache_dir
from engine.ingress.adapters.era5 import CdsEra5Adapter
from engine.ingress.adapters.request import IngestRequest
from engine.ingress.builder import TimeHistoryBuilder


@dataclass(frozen=True)
class HresT0Paths:
    surface: Path
    atmospheric: Path
    static: Path


WB2_HRES_SURF_FIELDS: tuple[tuple[str, str], ...] = (
    ("2m_temperature", "2t"),
    ("10m_u_component_of_wind", "10u"),
    ("10m_v_component_of_wind", "10v"),
    ("mean_sea_level_pressure", "msl"),
)

WB2_HRES_ATMOS_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("specific_humidity", "q"),
    ("geopotential", "z"),
)

WB2_HRES_STATIC_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (name, name) for name in STANDARD_STATIC
)


class Wb2HresT0Adapter:
    """Build a Batch from cached WeatherBench2 HRES T0 NetCDF files.

    Layout and semantics follow ``example_hres_t0.ipynb`` and ``hres_t0_data.py``.
    """

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        for path in (paths.surface, paths.atmospheric, paths.static):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing HRES T0 input {path}. "
                    "Download from WeatherBench2 or set IngestRequest.raw_paths."
                )

        with xr.open_dataset(paths.surface, engine=NETCDF_ENGINE) as surf_ds, xr.open_dataset(
            paths.atmospheric, engine=NETCDF_ENGINE
        ) as atmos_ds, xr.open_dataset(paths.static, engine=NETCDF_ENGINE) as static_ds:
            return self._build_batch(
                surf_ds=surf_ds,
                atmos_ds=atmos_ds,
                static_ds=static_ds,
                config=config,
                time_index=request.time_index,
            )

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> HresT0Paths:
        if request.raw_paths:
            try:
                return HresT0Paths(
                    surface=Path(request.raw_paths["surface"]).expanduser().resolve(),
                    atmospheric=Path(request.raw_paths["atmospheric"]).expanduser().resolve(),
                    static=Path(request.raw_paths["static"]).expanduser().resolve(),
                )
            except KeyError as exc:
                raise ValueError(
                    "raw_paths must include 'surface', 'atmospheric', and 'static'"
                ) from exc

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "hres_t0")
        return HresT0Paths(
            surface=cache_dir / f"{day}-surface-level.nc",
            atmospheric=cache_dir / f"{day}-atmospheric.nc",
            static=cache_dir / "static.nc",
        )

    def _build_batch(
        self,
        *,
        surf_ds: xr.Dataset,
        atmos_ds: xr.Dataset,
        static_ds: xr.Dataset,
        config: EngineConfig,
        time_index: int,
    ) -> Batch:
        if config.source.time_policy != "pair":
            raise ValueError(
                f"HRES T0 adapter requires source.time_policy='pair', got {config.source.time_policy!r}"
            )
        if not config.source.flip_lat:
            raise ValueError("HRES T0 adapter requires source.flip_lat=True")

        history = TimeHistoryBuilder(config.source, time_index=time_index)

        surf_vars = {
            aurora: history.build_surf_history(surf_ds[wb2].values)
            for wb2, aurora in WB2_HRES_SURF_FIELDS
        }
        atmos_vars = {
            aurora: history.build_atmos_history(atmos_ds[wb2].values)
            for wb2, aurora in WB2_HRES_ATMOS_FIELDS
        }
        static_vars = {
            aurora: torch.from_numpy(CdsEra5Adapter._static_array(static_ds[name].values))
            for name, aurora in WB2_HRES_STATIC_FIELDS
        }

        times = surf_ds.time.values.astype("datetime64[s]").tolist()
        if time_index < 0 or time_index >= len(times):
            raise ValueError(f"time_index {time_index} out of range for {len(times)} steps")

        level_coord = atmos_ds["level"] if "level" in atmos_ds.coords else atmos_ds["pressure_level"]

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=CdsEra5Adapter._coord_tensor(surf_ds.latitude.values[::-1]),
                lon=CdsEra5Adapter._coord_tensor(surf_ds.longitude.values),
                time=(times[time_index],),
                atmos_levels=tuple(int(level) for level in level_coord.values),
                rollout_step=0,
            ),
        )
