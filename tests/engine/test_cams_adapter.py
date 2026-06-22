from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from flash_aurora.engine.core.config import CAMS_STATIC, EngineConfig, STANDARD_LEVELS
from flash_aurora.engine.core.netcdf_codec import INGRESS_NETCDF_ENGINE, NETCDF_ENGINE
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.adapters.cams import CamsAdapter
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder


def write_cams_fixture(
    directory: Path,
    *,
    day: datetime = datetime(2022, 6, 11),
    height: int = 5,
    width: int = 8,
    levels: tuple[int, ...] = STANDARD_LEVELS,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(90, -90, height)
    lon = np.linspace(0, 360, width, endpoint=False)
    times = np.array(
        [
            np.datetime64("2022-06-11T00:00:00"),
            np.datetime64("2022-06-11T12:00:00"),
        ],
        dtype="datetime64[s]",
    )
    forecast_period = np.array([0, 6], dtype="timedelta64[h]")

    surf_vars = {
        "t2m": (("forecast_period", "valid_time", "latitude", "longitude"), np.random.randn(2, 2, height, width)),
        "u10": (("forecast_period", "valid_time", "latitude", "longitude"), np.random.randn(2, 2, height, width)),
        "v10": (("forecast_period", "valid_time", "latitude", "longitude"), np.random.randn(2, 2, height, width)),
        "msl": (("forecast_period", "valid_time", "latitude", "longitude"), np.random.randn(2, 2, height, width)),
    }
    for name in ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2"):
        surf_vars[name] = (
            ("forecast_period", "valid_time", "latitude", "longitude"),
            np.random.randn(2, 2, height, width),
        )

    surf = xr.Dataset(
        data_vars=surf_vars,
        coords={
            "forecast_period": forecast_period,
            "valid_time": times,
            "latitude": lat,
            "longitude": lon,
        },
    )

    atmos_vars = {
        name: (("forecast_period", "valid_time", "pressure_level", "latitude", "longitude"), np.random.randn(2, 2, len(levels), height, width))
        for name in ("t", "u", "v", "q", "z", "co", "no", "no2", "go3", "so2")
    }
    atmos = xr.Dataset(
        data_vars=atmos_vars,
        coords={
            "forecast_period": forecast_period,
            "valid_time": times,
            "pressure_level": list(levels),
            "latitude": lat,
            "longitude": lon,
        },
    )

    day_str = day.strftime("%Y-%m-%d")
    paths = {
        "surface": directory / f"{day_str}-cams-surface-level.nc",
        "atmospheric": directory / f"{day_str}-cams-atmospheric.nc",
    }
    surf.to_netcdf(paths["surface"], engine=NETCDF_ENGINE)
    atmos.to_netcdf(paths["atmospheric"], engine=NETCDF_ENGINE)
    return paths


def write_cams_static_pickle(path: Path, height: int, width: int) -> None:
    payload = {name: np.random.randn(height, width) for name in CAMS_STATIC}
    path.write_bytes(pickle.dumps(payload))


def tiny_cams_config(tmp_path: Path, levels: tuple[int, ...] = STANDARD_LEVELS) -> EngineConfig:
    base = DEFAULT_PRESETS.get("cams")
    variant = replace(base.variant, resolution=(5, 8), levels=levels)
    write_cams_static_pickle(tmp_path / "aurora-0.4-air-pollution-static.pickle", 5, 8)
    return EngineConfig(
        variant=variant,
        source=base.source,
        asset_root=tmp_path,
        allow_hub_download=False,
    )


def test_cams_adapter_builds_notebook_layout(tmp_path: Path) -> None:
    paths = write_cams_fixture(tmp_path / "inputs")
    config = tiny_cams_config(tmp_path)
    adapter = CamsAdapter()
    request = IngestRequest(
        valid_time=datetime(2022, 6, 11, 12),
        raw_paths=paths,
    )

    batch = adapter.build_initial_batch(request, config)

    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)
    assert batch.atmos_vars["co"].shape == (1, 2, len(STANDARD_LEVELS), 5, 8)
    assert batch.static_vars["static_co"].shape == (5, 8)
    assert "tcco" in batch.surf_vars
    assert batch.metadata.time == (datetime(2022, 6, 11, 12),)
    assert batch.metadata.atmos_levels == STANDARD_LEVELS


def test_from_source_validates_cams_batch(tmp_path: Path) -> None:
    paths = write_cams_fixture(tmp_path / "inputs")
    config = tiny_cams_config(tmp_path)
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2022, 6, 11, 12),
        raw_paths=paths,
    )
    batch = builder.from_source(request)
    assert batch.surf_vars["tcno2"].shape == (1, 2, 5, 8)


def test_cams_default_paths_under_cache_dir(tmp_path: Path) -> None:
    write_cams_fixture(tmp_path / "cams", day=datetime(2022, 6, 11))
    config = tiny_cams_config(tmp_path)
    adapter = CamsAdapter()
    request = IngestRequest(
        valid_time=datetime(2022, 6, 11, 12),
        cache_dir=tmp_path / "cams",
    )
    batch = adapter.build_initial_batch(request, config)
    assert batch.metadata.time == (datetime(2022, 6, 11, 12),)


def test_cams_adapter_reads_netcdf4_cache(tmp_path: Path) -> None:
    paths = write_cams_fixture(tmp_path / "inputs")
    config = tiny_cams_config(tmp_path)
    netcdf4_surface = tmp_path / "inputs" / "2022-06-11-cams-surface-level-netcdf4.nc"
    with xr.open_dataset(paths["surface"], engine=NETCDF_ENGINE) as src:
        src.to_netcdf(netcdf4_surface, engine=INGRESS_NETCDF_ENGINE)

    adapter = CamsAdapter()
    request = IngestRequest(
        valid_time=datetime(2022, 6, 11, 12),
        raw_paths={**paths, "surface": netcdf4_surface},
    )
    batch = adapter.build_initial_batch(request, config)
    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)


def test_cams_adapter_reorders_descending_pressure_levels(tmp_path: Path) -> None:
    descending = tuple(reversed(STANDARD_LEVELS))
    paths = write_cams_fixture(tmp_path / "inputs", levels=descending)
    config = tiny_cams_config(tmp_path)
    adapter = CamsAdapter()
    request = IngestRequest(
        valid_time=datetime(2022, 6, 11, 12),
        raw_paths=paths,
    )

    batch = adapter.build_initial_batch(request, config)

    assert batch.metadata.atmos_levels == STANDARD_LEVELS


def test_cams_missing_inputs_raises(tmp_path: Path) -> None:
    config = tiny_cams_config(tmp_path)
    adapter = CamsAdapter()
    request = IngestRequest(valid_time=datetime(2022, 6, 11, 12), cache_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        adapter.build_initial_batch(request, config)
