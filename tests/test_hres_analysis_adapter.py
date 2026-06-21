from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from engine.core.config import EngineConfig, STANDARD_LEVELS
from engine.core.netcdf_codec import NETCDF_ENGINE
from engine.core.presets import DEFAULT_PRESETS
from engine.ingress.adapters.hres_analysis import GribHresAnalysisAdapter
from engine.ingress.adapters.request import IngestRequest
from engine.ingress.build_ic import InitialConditionBuilder


def write_hres_analysis_fixture(
    directory: Path,
    *,
    day: datetime = datetime(2023, 1, 1),
    height: int = 9,
    width: int = 16,
    levels: tuple[int, ...] = STANDARD_LEVELS,
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
            "t2m": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "u10": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "v10": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
            "msl": (("time", "latitude", "longitude"), np.random.randn(4, height, width)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )

    def atmos_frame(offset: float) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                name: (("isobaricInhPa", "latitude", "longitude"), np.random.randn(len(levels), height, width) + offset)
                for name in ("t", "u", "v", "q", "z")
            },
            coords={"isobaricInhPa": list(levels), "latitude": lat, "longitude": lon},
        )

    day_str = day.strftime("%Y-%m-%d")
    paths = {
        "surface": directory / f"{day_str}-surface-level.nc",
        "atmospheric_00": directory / f"{day_str}-atmospheric-00.nc",
        "atmospheric_06": directory / f"{day_str}-atmospheric-06.nc",
    }
    surf.to_netcdf(paths["surface"], engine=NETCDF_ENGINE)
    atmos_frame(0.0).to_netcdf(paths["atmospheric_00"], engine=NETCDF_ENGINE)
    atmos_frame(1.0).to_netcdf(paths["atmospheric_06"], engine=NETCDF_ENGINE)
    return paths


def write_static_pickle(path: Path, height: int, width: int) -> None:
    payload = {
        "lsm": np.random.randn(height, width),
        "slt": np.random.randn(height, width),
        "z": np.random.randn(height, width),
    }
    path.write_bytes(pickle.dumps(payload))


def tiny_hres_01_config(
    tmp_path: Path,
    *,
    regrid_res: float = 45.0,
    output_shape: tuple[int, int] = (5, 8),
) -> EngineConfig:
    base = DEFAULT_PRESETS.get("hres_0.1")
    variant = replace(base.variant, resolution=output_shape, levels=STANDARD_LEVELS)
    source = replace(base.source, regrid_res=regrid_res)
    write_static_pickle(tmp_path / "aurora-0.1-static.pickle", *output_shape)
    return EngineConfig(
        variant=variant,
        source=source,
        asset_root=tmp_path,
        allow_hub_download=False,
    )


def test_hres_01_adapter_regrids_and_injects_static(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = tiny_hres_01_config(tmp_path)
    adapter = GribHresAnalysisAdapter()
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
    )

    batch = adapter.build_initial_batch(request, config)

    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)
    assert batch.atmos_vars["t"].shape == (1, 2, len(STANDARD_LEVELS), 5, 8)
    assert batch.static_vars["z"].shape == (5, 8)
    assert batch.metadata.atmos_levels == STANDARD_LEVELS
    assert batch.metadata.time == (datetime(2023, 1, 1, 6),)
    assert batch.metadata.lat[0] > batch.metadata.lat[-1]


def test_from_source_validates_hres_01_batch(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = tiny_hres_01_config(tmp_path)
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
    )
    batch = builder.from_source(request)
    assert batch.surf_vars["msl"].shape == (1, 2, 5, 8)


def test_hres_01_default_paths_under_cache_dir(tmp_path: Path) -> None:
    config = tiny_hres_01_config(tmp_path)
    cache = tmp_path / "inputs"
    write_hres_analysis_fixture(cache, day=datetime(2023, 1, 1))
    adapter = GribHresAnalysisAdapter()
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        cache_dir=cache,
    )
    batch = adapter.build_initial_batch(request, config)
    assert batch.metadata.time == (datetime(2023, 1, 1, 6),)


def test_hres_01_missing_inputs_raises(tmp_path: Path) -> None:
    config = tiny_hres_01_config(tmp_path)
    adapter = GribHresAnalysisAdapter()
    request = IngestRequest(valid_time=datetime(2023, 1, 1, 6), cache_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        adapter.build_initial_batch(request, config)
