from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import xarray as xr

from flash_aurora.engine.core.netcdf_codec import NETCDF_ENGINE, write_batch_netcdf
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.ic_cache import ic_cache_paths, netcdf_cache_location

from tests.engine.test_era5_adapter import tiny_era5_config, write_era5_fixture
from tests.engine.test_hres_analysis_adapter import (
    tiny_hres_01_config,
    write_hres_analysis_fixture,
)


def _assert_batches_equal(left, right) -> None:
    assert set(left.surf_vars) == set(right.surf_vars)
    for key in left.surf_vars:
        assert left.surf_vars[key].equal(right.surf_vars[key])
    assert set(left.atmos_vars) == set(right.atmos_vars)
    for key in left.atmos_vars:
        assert left.atmos_vars[key].equal(right.atmos_vars[key])
    assert set(left.static_vars) == set(right.static_vars)
    for key in left.static_vars:
        assert left.static_vars[key].equal(right.static_vars[key])
    assert left.metadata.atmos_levels == right.metadata.atmos_levels
    assert left.metadata.time == right.metadata.time
    assert left.metadata.rollout_step == right.metadata.rollout_step
    assert left.metadata.lat.equal(right.metadata.lat)
    assert left.metadata.lon.equal(right.metadata.lon)


def _with_ic_cache(config):
    return replace(config, ic_cache=True)


def test_hres_ic_cache_hit_is_bitwise_identical(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = _with_ic_cache(tiny_hres_01_config(tmp_path, regrid_res=45.0))
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
        cache_dir=paths["surface"].parent,
    )

    cold = builder.from_source(request)
    warm = builder.from_source(request)
    _assert_batches_equal(cold, warm)

    cache_root, cache_id = paths["surface"].parent, f"{config.variant.name}-2023-01-01-t1"
    cache_pt, cache_meta = ic_cache_paths(cache_root, cache_id)
    assert cache_pt.is_file()
    assert cache_meta.is_file()


def test_era5_ic_cache_hit_is_bitwise_identical(tmp_path: Path) -> None:
    paths = write_era5_fixture(tmp_path / "inputs")
    config = _with_ic_cache(tiny_era5_config())
    config.asset_root = tmp_path
    config.user_cwd = tmp_path
    builder = InitialConditionBuilder(config)
    request = IngestRequest(
        valid_time=datetime(2023, 1, 1, 6),
        raw_paths=paths,
        cache_dir=paths["surface"].parent,
        time_index=1,
    )

    cold = builder.from_source(request)
    warm = builder.from_source(request)
    _assert_batches_equal(cold, warm)


def test_ic_cache_disabled_recomputes_each_time(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = tiny_hres_01_config(tmp_path, regrid_res=45.0)
    assert config.ic_cache is False
    builder = InitialConditionBuilder(config)
    request = IngestRequest(valid_time=datetime(2023, 1, 1, 6), raw_paths=paths)

    builder.from_source(request)
    builder.from_source(request)

    cache_pt, _ = ic_cache_paths(
        paths["surface"].parent,
        f"{config.variant.name}-2023-01-01-t1",
    )
    assert not cache_pt.is_file()


def test_ic_cache_invalidates_when_input_changes(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = _with_ic_cache(tiny_hres_01_config(tmp_path, regrid_res=45.0))
    builder = InitialConditionBuilder(config)
    request = IngestRequest(valid_time=datetime(2023, 1, 1, 6), raw_paths=paths)

    first = builder.from_source(request)

    surface = paths["surface"]
    with xr.open_dataset(surface, engine=NETCDF_ENGINE) as dataset:
        data = dataset["t2m"].values.copy()
        data[0, 0, 0] += 17.0
        dataset = dataset.assign(t2m=(("time", "latitude", "longitude"), data))
        dataset.to_netcdf(surface, engine=NETCDF_ENGINE)

    second = builder.from_source(request)
    assert not first.surf_vars["2t"].equal(second.surf_vars["2t"])


def test_netcdf_path_ic_cache(tmp_path: Path) -> None:
    paths = write_hres_analysis_fixture(tmp_path / "inputs")
    config = tiny_hres_01_config(tmp_path, regrid_res=45.0)
    builder = InitialConditionBuilder(config)
    request = IngestRequest(valid_time=datetime(2023, 1, 1, 6), raw_paths=paths)
    reference = builder.from_source(request)

    ic_path = tmp_path / "user_ic.nc"
    write_batch_netcdf(reference, ic_path)

    cached_config = _with_ic_cache(config)
    cached_builder = InitialConditionBuilder(cached_config)
    cold = cached_builder.from_netcdf_path(ic_path)
    warm = cached_builder.from_netcdf_path(ic_path)
    _assert_batches_equal(cold, warm)

    cache_root, cache_id = netcdf_cache_location(ic_path, cached_config)
    cache_pt, _ = ic_cache_paths(cache_root, cache_id)
    assert cache_pt.is_file()
