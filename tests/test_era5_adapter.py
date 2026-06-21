from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from engine.core.config import EngineConfig
from engine.core.netcdf_codec import INGRESS_NETCDF_ENGINE, NETCDF_ENGINE
from engine.core.presets import DEFAULT_PRESETS
from engine.ingress.adapters.era5 import CdsEra5Adapter, open_ingress_netcdf
from engine.ingress.adapters.registry import DEFAULT_ADAPTERS
from engine.ingress.adapters.request import IngestRequest
from engine.ingress.build_ic import InitialConditionBuilder


def write_era5_fixture(
    directory: Path,
    *,
    day: datetime = datetime(2023, 1, 1),
    height: int = 5,
    width: int = 8,
    levels: tuple[int, ...] = (850, 925, 1000),
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(90, -90, height)
    lon = np.linspace(0, 360, width, endpoint=False)
    times = np.array(
        [
            np.datetime64("2023-01-01T00:00:00"),
            np.datetime64("2023-01-01T06:00:00"),
            np.datetime64("2023-01-01T12:00:00"),
            np.datetime64("2023-01-01T18:00:00"),
        ],
        dtype="datetime64[s]",
    )

    surf = xr.Dataset(
        data_vars={
            "t2m": (("valid_time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "u10": (("valid_time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "v10": (("valid_time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "msl": (("valid_time", "latitude", "longitude"), np.random.randn(4, height, width)),
        },
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    static = xr.Dataset(
        data_vars={
            "z": (("latitude", "longitude"), np.random.randn(height, width)),
            "slt": (("latitude", "longitude"), np.random.randn(height, width)),
            "lsm": (("latitude", "longitude"), np.random.randn(height, width)),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    atmos = xr.Dataset(
        data_vars={
            name: (("valid_time", "pressure_level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width))
            for name in ("t", "u", "v", "q", "z")
        },
        coords={
            "valid_time": times,
            "pressure_level": list(levels),
            "latitude": lat,
            "longitude": lon,
        },
    )

    day_str = day.strftime("%Y-%m-%d")
    paths = {
        "surface": directory / f"{day_str}-surface-level.nc",
        "atmospheric": directory / f"{day_str}-atmospheric.nc",
        "static": directory / "static.nc",
    }
    surf.to_netcdf(paths["surface"], engine=NETCDF_ENGINE)
    static.to_netcdf(paths["static"], engine=NETCDF_ENGINE)
    atmos.to_netcdf(paths["atmospheric"], engine=NETCDF_ENGINE)
    return paths


def tiny_era5_config(levels: tuple[int, ...] = (850, 925, 1000)) -> EngineConfig:
    base = DEFAULT_PRESETS.get("era5_pretrained")
    variant = replace(base.variant, resolution=(5, 8), levels=levels)
    return EngineConfig(
        variant=variant,
        source=base.source,
        allow_hub_download=False,
    )


def test_registry_lists_all_doc_sources() -> None:
    names = set(DEFAULT_ADAPTERS.names())
    assert names == {"cds_era5", "wb2_hres", "grib_ifs_0.1", "cams", "wb2_wam"}


def test_era5_adapter_builds_notebook_layout(tmp_path: Path) -> None:
    paths = write_era5_fixture(tmp_path)
    config = tiny_era5_config()
    adapter = CdsEra5Adapter()
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
        time_index=1,
    )

    batch = adapter.build_initial_batch(request, config)

    assert set(batch.surf_vars) == {"2t", "10u", "10v", "msl"}
    assert set(batch.static_vars) == {"lsm", "slt", "z"}
    assert set(batch.atmos_vars) == {"t", "u", "v", "q", "z"}
    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)
    assert batch.atmos_vars["t"].shape == (1, 2, 3, 5, 8)
    assert batch.metadata.time == (datetime(2023, 1, 1, 6),)
    assert batch.metadata.atmos_levels == (850, 925, 1000)
    assert batch.metadata.rollout_step == 0
    assert batch.metadata.lat[0] > batch.metadata.lat[-1]


def test_from_source_validates_era5_batch(tmp_path: Path) -> None:
    paths = write_era5_fixture(tmp_path)
    config = tiny_era5_config()
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
    )
    batch = builder.from_source(request)
    assert batch.surf_vars["msl"].shape == (1, 2, 5, 8)


def test_open_ingress_netcdf_reads_netcdf4(tmp_path: Path) -> None:
    paths = write_era5_fixture(tmp_path)
    netcdf4_path = tmp_path / "surface-netcdf4.nc"
    with xr.open_dataset(paths["surface"], engine=NETCDF_ENGINE) as src:
        src.to_netcdf(netcdf4_path, engine=INGRESS_NETCDF_ENGINE)

    with open_ingress_netcdf(netcdf4_path) as ds:
        assert "t2m" in ds


def test_era5_default_paths_under_cache_dir(tmp_path: Path) -> None:
    paths = write_era5_fixture(tmp_path / "era5")
    config = tiny_era5_config()
    adapter = CdsEra5Adapter()
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        cache_dir=tmp_path / "era5",
    )
    batch = adapter.build_initial_batch(request, config)
    assert batch.metadata.time == (datetime(2023, 1, 1, 6),)
