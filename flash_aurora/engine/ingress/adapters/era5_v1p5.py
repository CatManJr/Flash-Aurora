"""CDS ERA5 ingress adapter for Aurora 1.5 (matches upstream example_v1p5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from flash_aurora.models.aurora import Batch, Metadata
from flash_aurora.models.aurora_v1p5.insolation import insolation

from flash_aurora.engine.core.config import EngineConfig, V1P5_SURF_OUTPUT_ONLY
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters.base import resolve_cache_dir
from flash_aurora.engine.ingress.adapters.era5 import CdsEra5Adapter, open_ingress_netcdf
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.static import StaticFieldLoader
from flash_aurora.engine.ingress.time import TimeHistoryPolicy

# CDS NetCDF / GRIB short names -> Aurora short names (upstream example_v1p5).
CDS_ERA5_V1P5_SURF_FIELDS: tuple[tuple[str, str], ...] = (
    ("t2m", "2t"),
    ("u10", "10u"),
    ("v10", "10v"),
    ("msl", "msl"),
    ("d2m", "2d"),
    ("tcwv", "tcwv"),
    ("tcc", "tcc"),
    ("u100", "100u"),
    ("v100", "100v"),
    ("sp", "sp"),
    ("lcc", "lcc"),
    ("mcc", "mcc"),
    ("hcc", "hcc"),
    ("skt", "skt"),
    ("stl1", "stl1"),
    ("swvl1", "swvl1"),
    ("siconc", "ci"),
    ("sd", "scaled_sd"),
)

CDS_ERA5_V1P5_ATMOS_FIELDS: tuple[tuple[str, str], ...] = (
    ("z", "z"),
    ("u", "u"),
    ("v", "v"),
    ("t", "t"),
    ("q", "q"),
)

_OUTPUT_ONLY: frozenset[str] = frozenset(V1P5_SURF_OUTPUT_ONLY)


@dataclass(frozen=True)
class Era5V1p5Paths:
    surface: Path
    atmospheric: Path


class CdsEra5V1p5Adapter:
    """Build a Batch for ``aurora_v1p5`` from CDS ERA5 NetCDF + HF static pickle."""

    def build_initial_batch(self, request: IngestRequest, config: EngineConfig) -> Batch:
        paths = self._resolve_paths(request, config)
        for path in (paths.surface, paths.atmospheric):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing ERA5 V1p5 input {path}. "
                    "Download with DataDownloader.ensure() (CDS) or set IngestRequest.raw_paths."
                )

        with open_ingress_netcdf(paths.surface) as surf_ds, open_ingress_netcdf(
            paths.atmospheric
        ) as atmos_ds:
            batch = self._build_batch(
                surf_ds=surf_ds,
                atmos_ds=atmos_ds,
                config=config,
                time_index=request.time_index,
            )

        assets = AssetStore(root=config.asset_root)
        batch.static_vars = StaticFieldLoader(config, assets).load(
            lat=batch.metadata.lat,
            lon=batch.metadata.lon,
        )
        return batch

    def _resolve_paths(self, request: IngestRequest, config: EngineConfig) -> Era5V1p5Paths:
        if request.raw_paths:
            try:
                return Era5V1p5Paths(
                    surface=Path(request.raw_paths["surface"]).expanduser().resolve(),
                    atmospheric=Path(request.raw_paths["atmospheric"]).expanduser().resolve(),
                )
            except KeyError as exc:
                raise ValueError(
                    "raw_paths must include 'surface' and 'atmospheric'"
                ) from exc

        day = request.valid_time.strftime("%Y-%m-%d")
        cache_dir = resolve_cache_dir(request, config, "era5_v1p5")
        return Era5V1p5Paths(
            surface=cache_dir / f"{day}-surface-level.nc",
            atmospheric=cache_dir / f"{day}-atmospheric.nc",
        )

    def _build_batch(
        self,
        *,
        surf_ds: xr.Dataset,
        atmos_ds: xr.Dataset,
        config: EngineConfig,
        time_index: int,
    ) -> Batch:
        if config.source.time_policy != "first_two":
            raise ValueError(
                f"ERA5 V1p5 adapter requires source.time_policy='first_two', "
                f"got {config.source.time_policy!r}"
            )

        policy = TimeHistoryPolicy(config.source.time_policy, time_index=time_index)
        missing: list[str] = []
        surf_vars: dict[str, torch.Tensor] = {}
        for cds_name, aurora_name in CDS_ERA5_V1P5_SURF_FIELDS:
            if cds_name not in surf_ds:
                missing.append(f"{aurora_name} (NetCDF '{cds_name}')")
                continue
            values = policy.select_pair(surf_ds[cds_name].values)
            values = np.nan_to_num(np.ascontiguousarray(values), nan=0.0).astype(np.float32)
            surf_vars[aurora_name] = torch.from_numpy(values[None])

        if missing:
            raise FileNotFoundError(
                "Aurora 1.5 surface IC is incomplete. Missing: "
                f"{', '.join(missing)}. Output-only vars "
                f"({', '.join(sorted(_OUTPUT_ONLY))}) are not required in the IC."
            )

        level_coord = (
            atmos_ds["pressure_level"] if "pressure_level" in atmos_ds.coords else atmos_ds["level"]
        )
        levels = tuple(int(level) for level in level_coord.values)
        expected_levels = config.variant.levels
        if set(levels) != set(expected_levels):
            raise ValueError(
                f"ERA5 atmospheric levels {levels} do not match expected {expected_levels}"
            )
        level_index = [levels.index(level) for level in expected_levels]

        atmos_vars: dict[str, torch.Tensor] = {}
        atmos_missing: list[str] = []
        for cds_name, aurora_name in CDS_ERA5_V1P5_ATMOS_FIELDS:
            if cds_name not in atmos_ds:
                atmos_missing.append(f"{aurora_name} (NetCDF '{cds_name}')")
                continue
            values = policy.select_pair(atmos_ds[cds_name].values[:, level_index, ...])
            values = np.nan_to_num(np.ascontiguousarray(values), nan=0.0).astype(np.float32)
            atmos_vars[aurora_name] = torch.from_numpy(values[None])
        if atmos_missing:
            raise FileNotFoundError(
                f"Aurora 1.5 atmospheric IC is incomplete. Missing: {', '.join(atmos_missing)}"
            )

        time_coord = surf_ds["valid_time"] if "valid_time" in surf_ds.coords else surf_ds["time"]
        valid_times = time_coord.values.astype("datetime64[s]").tolist()
        if time_index < 0 or time_index >= len(valid_times):
            raise ValueError(f"time_index {time_index} out of range for {len(valid_times)} steps")

        lat = CdsEra5Adapter._coord_tensor(surf_ds.latitude.values)
        lon_vals = np.asarray(surf_ds.longitude.values, dtype=np.float64)
        if float(lon_vals.min()) < 0:
            lon_vals = np.mod(lon_vals, 360.0)
        lon = CdsEra5Adapter._coord_tensor(lon_vals)

        history_times = (valid_times[0], valid_times[1])
        surf_vars["insolation"] = self._build_insolation_history(
            history_times=history_times,
            lat=lat,
            lon=lon,
            ref=next(iter(surf_vars.values())),
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars={},
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(valid_times[time_index],),
                atmos_levels=expected_levels,
                rollout_step=0,
            ),
        )

    @staticmethod
    def _build_insolation_history(
        *,
        history_times: tuple[object, object],
        lat: torch.Tensor,
        lon: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        lat_np = lat.detach().cpu().numpy().astype(np.float32)
        lon_np = lon.detach().cpu().numpy().astype(np.float32)
        frames = []
        for when in history_times:
            sol = insolation([when], lat_np, lon_np, enforce_2d=True)
            frames.append(sol[0])
        stacked = np.stack(frames, axis=0)[None, ...].astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(stacked)).to(dtype=ref.dtype)
