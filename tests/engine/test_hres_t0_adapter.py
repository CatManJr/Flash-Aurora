from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from flash_aurora.engine.core.config import EngineConfig, STANDARD_LEVELS
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.adapters.hres_t0 import Wb2HresT0Adapter
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder


def write_hres_t0_fixture(
    directory: Path,
    *,
    day: datetime = datetime(2022, 5, 11),
    height: int = 5,
    width: int = 8,
    levels: tuple[int, ...] = (850, 925, 1000),
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    # WeatherBench2 stores latitude increasing south-to-north before the adapter flip.
    lat = np.linspace(-90, 90, height)
    lon = np.linspace(0, 360, width, endpoint=False)
    times = np.array(
        [
            np.datetime64("2022-05-11T00:00:00"),
            np.datetime64("2022-05-11T06:00:00"),
            np.datetime64("2022-05-11T12:00:00"),
            np.datetime64("2022-05-11T18:00:00"),
        ],
        dtype="datetime64[s]",
    )

    surf = xr.Dataset(
        data_vars={
            "2m_temperature": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "mean_sea_level_pressure": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    static = xr.Dataset(
        data_vars={
            "z": (("latitude", "longitude"), np.random.randn(height, width)),
            "slt": (("latitude", "longitude"), np.random.randn(height, width)),
            "lsm": (("latitude", "longitude"), np.random.randn(height, width)),
        },
        coords={"latitude": np.linspace(90, -90, height), "longitude": lon},
    )
    atmos = xr.Dataset(
        data_vars={
            "temperature": (("time", "level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width)),
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width)),
            "v_component_of_wind": (("time", "level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width)),
            "specific_humidity": (("time", "level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width)),
            "geopotential": (("time", "level", "latitude", "longitude"), np.random.randn(4, len(levels), height, width)),
        },
        coords={"time": times, "level": list(levels), "latitude": lat, "longitude": lon},
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


def tiny_hres_t0_config(levels: tuple[int, ...] = (850, 925, 1000)) -> EngineConfig:
    base = DEFAULT_PRESETS.get("hres_t0_finetuned")
    variant = replace(base.variant, resolution=(5, 8), levels=levels)
    return EngineConfig(
        variant=variant,
        source=base.source,
        allow_hub_download=False,
    )


def _reference_prepare(values: np.ndarray, time_index: int = 2) -> torch.Tensor:
    return torch.from_numpy(values[[time_index - 1, time_index]][None][..., ::-1, :].copy())


def test_hres_t0_adapter_builds_notebook_layout(tmp_path: Path) -> None:
    paths = write_hres_t0_fixture(tmp_path)
    config = tiny_hres_t0_config()
    adapter = Wb2HresT0Adapter()
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 12),
        raw_paths=paths,
        time_index=2,
    )

    batch = adapter.build_initial_batch(request, config)

    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)
    assert batch.metadata.time == (datetime(2022, 5, 11, 12),)
    assert batch.metadata.atmos_levels == (850, 925, 1000)
    assert batch.metadata.lat[0] > batch.metadata.lat[-1]
    assert batch.static_vars["z"].shape == (5, 8)


def test_hres_t0_matches_hres_t0_data_prepare(tmp_path: Path) -> None:
    paths = write_hres_t0_fixture(tmp_path)
    config = tiny_hres_t0_config()
    adapter = Wb2HresT0Adapter()
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 12),
        raw_paths=paths,
        time_index=2,
    )
    batch = adapter.build_initial_batch(request, config)

    surf = xr.open_dataset(paths["surface"], engine=NETCDF_ENGINE)
    try:
        expected = _reference_prepare(surf["2m_temperature"].values, time_index=2)
    finally:
        surf.close()

    assert torch.allclose(batch.surf_vars["2t"], expected)


def test_from_source_validates_hres_t0_batch(tmp_path: Path) -> None:
    paths = write_hres_t0_fixture(tmp_path)
    config = tiny_hres_t0_config()
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 12),
        raw_paths=paths,
        time_index=2,
    )
    batch = builder.from_source(request)
    assert batch.surf_vars["msl"].shape == (1, 2, 5, 8)


def test_tc_tracking_preset_uses_hres_t0_adapter(tmp_path: Path) -> None:
    paths = write_hres_t0_fixture(tmp_path)
    config = DEFAULT_PRESETS.get("tc_tracking")
    config = replace(config, variant=replace(config.variant, resolution=(5, 8), levels=(850, 925, 1000)))
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 12),
        raw_paths=paths,
        time_index=2,
    )
    batch = builder.from_source(request)
    assert batch.metadata.time == (datetime(2022, 5, 11, 12),)


def test_hres_t0_default_paths_under_cache_dir(tmp_path: Path) -> None:
    write_hres_t0_fixture(tmp_path / "hres_t0", day=datetime(2022, 5, 11))
    config = tiny_hres_t0_config()
    adapter = Wb2HresT0Adapter()
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 12),
        cache_dir=tmp_path / "hres_t0",
        time_index=2,
    )
    batch = adapter.build_initial_batch(request, config)
    assert batch.metadata.time == (datetime(2022, 5, 11, 12),)
