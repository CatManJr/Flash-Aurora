from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from flash_aurora.aurora import Batch, Metadata

from flash_aurora.engine.core.config import EngineConfig, STANDARD_ATMOS, STANDARD_SURF
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters.base import resolve_cache_dir
from flash_aurora.engine.ingress.adapters.era5 import CdsEra5Adapter
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.static import StaticFieldLoader


@dataclass(frozen=True)
class HresAnalysisPaths:
    cache_dir: Path
    day: str

    @property
    def surface_nc(self) -> Path:
        return self.cache_dir / f"{self.day}-surface-level.nc"

    @property
    def atmospheric_00_nc(self) -> Path:
        return self.cache_dir / f"{self.day}-atmospheric-00.nc"

    @property
    def atmospheric_06_nc(self) -> Path:
        return self.cache_dir / f"{self.day}-atmospheric-06.nc"

    def surf_grib(self, var: str) -> Path:
        return self.cache_dir / f"surf_{var}_{self.day}.grib"

    def atmos_grib(self, var: str, hour: int) -> Path:
        return self.cache_dir / f"atmos_{var}_{self.day}_{hour:02d}.grib"


GRIB_SURF_FIELDS: tuple[tuple[str, str], ...] = (
    ("2t", "t2m"),
    ("10u", "u10"),
    ("10v", "v10"),
    ("msl", "msl"),
)


class GribHresAnalysisAdapter:
    """Build a Batch from IFS HRES 0.1° analysis inputs (example_hres_0.1.ipynb).

    Supports either cached NetCDF intermediates or the notebook GRIB layout (cfgrib).
    Applies ``batch.regrid`` then injects HuggingFace static fields at target resolution.
    """

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        if self._has_netcdf_cache(paths):
            batch = self._build_from_netcdf(paths, config, request)
        elif self._has_grib_cache(paths):
            batch = self._build_from_grib(paths, config, request)
        else:
            raise FileNotFoundError(
                f"Missing HRES 0.1 inputs under {paths.cache_dir}. "
                "Provide NetCDF cache files or GRIB downloads from example_hres_0.1."
            )

        regrid_res = config.source.regrid_res if config.source.regrid_res is not None else 0.1
        batch = batch.regrid(res=regrid_res)

        assets = AssetStore(root=config.asset_root)
        static_loader = StaticFieldLoader(config, assets)
        batch.static_vars = static_loader.load()
        return batch

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> HresAnalysisPaths:
        if request.raw_paths:
            surface = request.raw_paths.get("surface")
            if surface is not None:
                surface = Path(surface).expanduser().resolve()
                cache_dir = surface.parent
                day = request.valid_time.strftime("%Y-%m-%d")
                return HresAnalysisPaths(cache_dir=cache_dir, day=day)

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "hres_0.1")
        return HresAnalysisPaths(cache_dir=cache_dir, day=day)

    @staticmethod
    def _has_netcdf_cache(paths: HresAnalysisPaths) -> bool:
        return (
            paths.surface_nc.is_file()
            and paths.atmospheric_00_nc.is_file()
            and paths.atmospheric_06_nc.is_file()
        )

    def _has_grib_cache(self, paths: HresAnalysisPaths) -> bool:
        return paths.surf_grib("2t").is_file() and paths.atmos_grib("t", 0).is_file()

    def _build_from_netcdf(
        self,
        paths: HresAnalysisPaths,
        config: EngineConfig,
        request: IngestRequest,
    ) -> Batch:
        atmospheric_00 = paths.atmospheric_00_nc
        atmospheric_06 = paths.atmospheric_06_nc
        if request.raw_paths:
            atmospheric_00 = Path(
                request.raw_paths.get("atmospheric_00", atmospheric_00)
            ).expanduser().resolve()
            atmospheric_06 = Path(
                request.raw_paths.get("atmospheric_06", atmospheric_06)
            ).expanduser().resolve()

        surface_path = paths.surface_nc
        if request.raw_paths.get("surface"):
            surface_path = Path(request.raw_paths["surface"]).expanduser().resolve()

        with xr.open_dataset(surface_path, engine=NETCDF_ENGINE) as surf_ds, xr.open_dataset(
            atmospheric_00, engine=NETCDF_ENGINE
        ) as atmos_00, xr.open_dataset(atmospheric_06, engine=NETCDF_ENGINE) as atmos_06:
            return self._assemble_batch(surf_ds, atmos_00, atmos_06, config, request)

    def _build_from_grib(
        self,
        paths: HresAnalysisPaths,
        config: EngineConfig,
        request: IngestRequest,
    ) -> Batch:
        try:
            with xr.open_dataset(paths.surf_grib("2t"), engine="cfgrib") as ref_ds:
                lat = CdsEra5Adapter._coord_tensor(ref_ds.latitude.values)
                lon = CdsEra5Adapter._coord_tensor(ref_ds.longitude.values)
        except ImportError as exc:
            raise ImportError(
                "Reading GRIB inputs requires cfgrib. Install with: pip install cfgrib"
            ) from exc

        surf_vars = {
            aurora: self._load_surf_grib(paths, aurora, file_var)
            for aurora, file_var in GRIB_SURF_FIELDS
        }
        atmos_vars = {
            name: self._load_atmos_grib(paths, name, config.variant.levels)
            for name in STANDARD_ATMOS
        }

        return Batch(
            surf_vars=surf_vars,
            static_vars={},
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(self._analysis_time(request),),
                atmos_levels=config.variant.levels,
                rollout_step=0,
            ),
        )

    def _assemble_batch(
        self,
        surf_ds: xr.Dataset,
        atmos_00: xr.Dataset,
        atmos_06: xr.Dataset,
        config: EngineConfig,
        request: IngestRequest,
    ) -> Batch:
        surf_vars = {
            aurora: torch.from_numpy(np.ascontiguousarray(surf_ds[cds].values[:2][None]))
            for aurora, cds in GRIB_SURF_FIELDS
        }
        atmos_vars = {
            name: self._stack_atmos_hours(atmos_00, atmos_06, name, config.variant.levels)
            for name in STANDARD_ATMOS
        }

        lat = CdsEra5Adapter._coord_tensor(surf_ds.latitude.values)
        lon = CdsEra5Adapter._coord_tensor(surf_ds.longitude.values)

        return Batch(
            surf_vars=surf_vars,
            static_vars={},
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(self._analysis_time(request),),
                atmos_levels=config.variant.levels,
                rollout_step=0,
            ),
        )

    @staticmethod
    def _analysis_time(request: IngestRequest) -> datetime:
        return request.valid_time.replace(hour=6, minute=0, second=0, microsecond=0)

    @staticmethod
    def _level_coord(dataset: xr.Dataset) -> str:
        if "isobaricInhPa" in dataset.coords:
            return "isobaricInhPa"
        if "pressure_level" in dataset.coords:
            return "pressure_level"
        return "level"

    def _stack_atmos_hours(
        self,
        atmos_00: xr.Dataset,
        atmos_06: xr.Dataset,
        var: str,
        levels: tuple[int | float, ...],
    ) -> torch.Tensor:
        level_name = self._level_coord(atmos_00)
        frame_00 = atmos_00[var].sel({level_name: list(levels)}).values
        frame_06 = atmos_06[var].sel({level_name: list(levels)}).values
        stacked = np.stack((frame_00, frame_06), axis=0)
        return torch.from_numpy(np.ascontiguousarray(stacked)[None])

    def _load_surf_grib(self, paths: HresAnalysisPaths, aurora: str, file_var: str) -> torch.Tensor:
        with xr.open_dataset(paths.surf_grib(aurora), engine="cfgrib") as dataset:
            data = np.ascontiguousarray(dataset[file_var].values[:2][None])
        return torch.from_numpy(data)

    def _load_atmos_grib(
        self,
        paths: HresAnalysisPaths,
        var: str,
        levels: tuple[int | float, ...],
    ) -> torch.Tensor:
        with xr.open_dataset(paths.atmos_grib(var, 0), engine="cfgrib") as ds_00, xr.open_dataset(
            paths.atmos_grib(var, 6), engine="cfgrib"
        ) as ds_06:
            level_name = self._level_coord(ds_00)
            frame_00 = ds_00[var].sel({level_name: list(levels)}).values
            frame_06 = ds_06[var].sel({level_name: list(levels)}).values
        stacked = np.stack((frame_00, frame_06), axis=0)
        return torch.from_numpy(np.ascontiguousarray(stacked)[None])
