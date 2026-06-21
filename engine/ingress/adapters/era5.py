from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import Batch, Metadata

from engine.core.config import EngineConfig, STANDARD_ATMOS, STANDARD_SURF, STANDARD_STATIC
from engine.core.netcdf_codec import NETCDF_ENGINE
from engine.ingress.adapters.base import resolve_cache_dir
from engine.ingress.adapters.request import IngestRequest
from engine.ingress.time import TimeHistoryPolicy


@dataclass(frozen=True)
class Era5Paths:
    surface: Path
    atmospheric: Path
    static: Path


CDS_ERA5_SURF_FIELDS: tuple[tuple[str, str], ...] = (
    ("t2m", "2t"),
    ("u10", "10u"),
    ("v10", "10v"),
    ("msl", "msl"),
)

CDS_ERA5_ATMOS_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (name, name) for name in STANDARD_ATMOS
)

CDS_ERA5_STATIC_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (name, name) for name in STANDARD_STATIC
)


class CdsEra5Adapter:
    """Build a Batch from cached CDS ERA5 NetCDF files (example_era5 layout)."""

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        for path in (paths.surface, paths.atmospheric, paths.static):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing ERA5 input {path}. Download with CDS API or set IngestRequest.raw_paths."
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

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> Era5Paths:
        if request.raw_paths:
            try:
                return Era5Paths(
                    surface=Path(request.raw_paths["surface"]).expanduser().resolve(),
                    atmospheric=Path(request.raw_paths["atmospheric"]).expanduser().resolve(),
                    static=Path(request.raw_paths["static"]).expanduser().resolve(),
                )
            except KeyError as exc:
                raise ValueError(
                    "raw_paths must include 'surface', 'atmospheric', and 'static'"
                ) from exc

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "era5")
        return Era5Paths(
            surface=cache_dir / f"{day}-surface-level.nc",
            atmospheric=cache_dir / f"{day}-atmospheric.nc",
            static=cache_dir / "static.nc",
        )

    @staticmethod
    def _coord_tensor(values) -> torch.Tensor:
        array = np.asarray(values)
        if not array.flags.writeable:
            array = array.copy()
        return torch.from_numpy(array)

    def _build_batch(
        self,
        *,
        surf_ds: xr.Dataset,
        atmos_ds: xr.Dataset,
        static_ds: xr.Dataset,
        config: EngineConfig,
        time_index: int,
    ) -> Batch:
        if config.source.time_policy != "first_two":
            raise ValueError(
                f"ERA5 adapter requires source.time_policy='first_two', got {config.source.time_policy!r}"
            )

        policy = TimeHistoryPolicy(config.source.time_policy)

        surf_vars = {
            aurora: self._history_tensor(policy.select_pair(surf_ds[cds].values))
            for cds, aurora in CDS_ERA5_SURF_FIELDS
        }

        static_vars = {
            aurora: torch.from_numpy(self._static_array(static_ds[cds].values))
            for cds, aurora in CDS_ERA5_STATIC_FIELDS
        }

        atmos_vars = {
            aurora: self._history_tensor(policy.select_pair(atmos_ds[cds].values))
            for cds, aurora in CDS_ERA5_ATMOS_FIELDS
        }

        lat = self._coord_tensor(surf_ds.latitude.values)
        lon = self._coord_tensor(surf_ds.longitude.values)
        time_coord = surf_ds["valid_time"] if "valid_time" in surf_ds.coords else surf_ds["time"]
        valid_times = time_coord.values.astype("datetime64[s]").tolist()
        if time_index < 0 or time_index >= len(valid_times):
            raise ValueError(f"time_index {time_index} out of range for {len(valid_times)} steps")

        level_coord = (
            atmos_ds["pressure_level"] if "pressure_level" in atmos_ds.coords else atmos_ds["level"]
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(valid_times[time_index],),
                atmos_levels=tuple(int(level) for level in level_coord.values),
                rollout_step=0,
            ),
        )

    @staticmethod
    def _history_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values)[None])

    @staticmethod
    def _static_array(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim >= 3:
            return np.ascontiguousarray(array[0])
        return np.ascontiguousarray(array)
