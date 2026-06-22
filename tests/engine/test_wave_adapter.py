from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from flash_aurora.engine.core.config import EngineConfig, STANDARD_LEVELS, WAVE_STATIC, WAVE_SURF_WAM
from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.adapters.wave import Wb2WamWaveAdapter
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder


def write_wave_fixture(
    directory: Path,
    *,
    day: datetime = datetime(2022, 5, 11),
    height: int = 5,
    width: int = 8,
    levels: tuple[int, ...] = STANDARD_LEVELS,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(-90, 90, height)
    lon = np.linspace(0, 360, width, endpoint=False)
    times = np.array(
        [
            np.datetime64("2022-05-11T00:00:00"),
            np.datetime64("2022-05-11T06:00:00"),
            np.datetime64("2022-05-11T12:00:00"),
        ],
        dtype="datetime64[s]",
    )

    surf = xr.Dataset(
        data_vars={
            "2m_temperature": (("time", "latitude", "longitude"), np.random.randn(3, height, width)),
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.random.randn(3, height, width)),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.random.randn(3, height, width)),
            "mean_sea_level_pressure": (("time", "latitude", "longitude"), np.random.randn(3, height, width)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    atmos = xr.Dataset(
        data_vars={
            "temperature": (("time", "level", "latitude", "longitude"), np.random.randn(3, len(levels), height, width)),
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), np.random.randn(3, len(levels), height, width)),
            "v_component_of_wind": (("time", "level", "latitude", "longitude"), np.random.randn(3, len(levels), height, width)),
            "specific_humidity": (("time", "level", "latitude", "longitude"), np.random.randn(3, len(levels), height, width)),
            "geopotential": (("time", "level", "latitude", "longitude"), np.random.randn(3, len(levels), height, width)),
        },
        coords={"time": times, "level": list(levels), "latitude": lat, "longitude": lon},
    )

    marker = np.arange(height * width, dtype=np.float64).reshape(height, width)
    wave_vars = {
        name: (("time", "latitude", "longitude"), np.broadcast_to(marker, (2, height, width)).copy())
        for name in WAVE_SURF_WAM
    }
    wave = xr.Dataset(
        data_vars=wave_vars,
        coords={"time": times[:2], "latitude": lat, "longitude": lon},
    )

    day_str = day.strftime("%Y-%m-%d")
    paths = {
        "surface": directory / f"{day_str}-surface-level.nc",
        "atmospheric": directory / f"{day_str}-atmospheric.nc",
        "wave": directory / f"{day_str}-wave.nc",
    }
    surf.to_netcdf(paths["surface"], engine=NETCDF_ENGINE)
    atmos.to_netcdf(paths["atmospheric"], engine=NETCDF_ENGINE)
    wave.to_netcdf(paths["wave"], engine=NETCDF_ENGINE)
    return paths


def write_wave_static_pickle(path: Path, height: int, width: int) -> None:
    payload = {name: np.random.randn(height, width) for name in WAVE_STATIC}
    path.write_bytes(pickle.dumps(payload))


def tiny_wave_config(tmp_path: Path, levels: tuple[int, ...] = STANDARD_LEVELS) -> EngineConfig:
    base = DEFAULT_PRESETS.get("wave")
    variant = replace(base.variant, resolution=(5, 8), levels=levels)
    write_wave_static_pickle(tmp_path / "aurora-0.25-wave-static.pickle", 5, 8)
    return EngineConfig(
        variant=variant,
        source=base.source,
        asset_root=tmp_path,
        allow_hub_download=False,
    )


def test_wave_adapter_builds_notebook_layout(tmp_path: Path) -> None:
    paths = write_wave_fixture(tmp_path / "inputs")
    config = tiny_wave_config(tmp_path)
    adapter = Wb2WamWaveAdapter()
    request = IngestRequest(valid_time=datetime(2022, 5, 11, 6), raw_paths=paths)

    batch = adapter.build_initial_batch(request, config)

    assert batch.surf_vars["2t"].shape == (1, 2, 5, 8)
    assert batch.atmos_vars["t"].shape == (1, 2, len(STANDARD_LEVELS), 5, 8)
    assert set(batch.surf_vars) == {"2t", "10u", "10v", "msl"} | set(WAVE_SURF_WAM)
    assert set(batch.static_vars) == set(WAVE_STATIC)
    assert batch.metadata.time == (datetime(2022, 5, 11, 6),)
    assert batch.metadata.atmos_levels == STANDARD_LEVELS
    assert batch.metadata.lat[0] > batch.metadata.lat[-1]
    assert "dwi" in batch.surf_vars
    assert "10u_wave" not in batch.surf_vars


def test_wave_hres_fields_flip_but_wave_fields_do_not(tmp_path: Path) -> None:
    paths = write_wave_fixture(tmp_path / "inputs")
    config = tiny_wave_config(tmp_path)
    adapter = Wb2WamWaveAdapter()
    request = IngestRequest(valid_time=datetime(2022, 5, 11, 6), raw_paths=paths)

    batch = adapter.build_initial_batch(request, config)

    with xr.open_dataset(paths["surface"], engine=NETCDF_ENGINE) as surf_ds, xr.open_dataset(
        paths["wave"], engine=NETCDF_ENGINE
    ) as wave_ds:
        expected_hres = torch.from_numpy(
            np.ascontiguousarray(surf_ds["2m_temperature"].values[:2])[None][..., ::-1, :].copy()
        )
        expected_wave = torch.from_numpy(np.ascontiguousarray(wave_ds["swh"].values[:2])[None])

    assert torch.equal(batch.surf_vars["2t"], expected_hres)
    assert torch.equal(batch.surf_vars["swh"], expected_wave)


def test_from_source_validates_wave_batch(tmp_path: Path) -> None:
    paths = write_wave_fixture(tmp_path / "inputs")
    config = tiny_wave_config(tmp_path)
    builder = InitialConditionBuilder(config)
    request = IngestRequest(valid_time=datetime(2022, 5, 11, 6), raw_paths=paths)

    batch = builder.from_source(request)
    assert batch.surf_vars["wind"].shape == (1, 2, 5, 8)


def test_wave_default_paths_under_cache_dir(tmp_path: Path) -> None:
    write_wave_fixture(tmp_path / "wave", day=datetime(2022, 5, 11))
    config = tiny_wave_config(tmp_path)
    adapter = Wb2WamWaveAdapter()
    request = IngestRequest(
        valid_time=datetime(2022, 5, 11, 6),
        cache_dir=tmp_path / "wave",
    )

    batch = adapter.build_initial_batch(request, config)
    assert batch.metadata.time == (datetime(2022, 5, 11, 6),)


def test_wave_missing_inputs_raises(tmp_path: Path) -> None:
    config = tiny_wave_config(tmp_path)
    adapter = Wb2WamWaveAdapter()
    request = IngestRequest(valid_time=datetime(2022, 5, 11, 6), cache_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        adapter.build_initial_batch(request, config)
