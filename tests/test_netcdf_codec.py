from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aurora import Batch, Metadata
from engine.core.netcdf_codec import read_batch_netcdf, write_batch_netcdf


def _sample_batch() -> Batch:
    lat = torch.linspace(90, -90, 4)
    lon = torch.arange(0, 360, 360 / 8)
    return Batch(
        surf_vars={"2t": torch.randn(1, 2, 4, 8)},
        static_vars={"lsm": torch.randn(4, 8)},
        atmos_vars={"t": torch.randn(1, 2, 3, 4, 8)},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(np.datetime64("2020-01-01T00:00:00"), np.datetime64("2020-01-01T06:00:00")),
            atmos_levels=(100, 250, 500),
            rollout_step=0,
        ),
    )


def test_netcdf_roundtrip(tmp_path: Path) -> None:
    batch = _sample_batch()
    path = tmp_path / "batch.nc"
    write_batch_netcdf(batch, path)
    loaded = read_batch_netcdf(path)

    for key in batch.surf_vars:
        np.testing.assert_allclose(batch.surf_vars[key], loaded.surf_vars[key])
    for key in batch.static_vars:
        np.testing.assert_allclose(batch.static_vars[key], loaded.static_vars[key])
    for key in batch.atmos_vars:
        np.testing.assert_allclose(batch.atmos_vars[key], loaded.atmos_vars[key])

    np.testing.assert_allclose(batch.metadata.lat, loaded.metadata.lat)
    np.testing.assert_allclose(batch.metadata.lon, loaded.metadata.lon)
    assert batch.metadata.time == loaded.metadata.time
    assert batch.metadata.atmos_levels == loaded.metadata.atmos_levels
    assert batch.metadata.rollout_step == loaded.metadata.rollout_step


def test_netcdf_write_does_not_import_netcdf4(tmp_path: Path) -> None:
    import sys

    sys.modules.pop("netCDF4", None)
    batch = _sample_batch()
    write_batch_netcdf(batch, tmp_path / "batch.nc")
