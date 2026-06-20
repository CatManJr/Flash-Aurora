from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine import AuroraEngine
from engine.core.netcdf_codec import write_batch_netcdf
from engine.ingress.build_ic import InitialConditionBuilder
from tests.helpers import assert_batches_close


@pytest.mark.integration
@pytest.mark.gpu
def test_run_from_netcdf_predict(
    engine_config_offline,
    asset_root: Path,
    tmp_path: Path,
) -> None:
    engine = AuroraEngine(engine_config_offline)
    engine.load()

    builder = InitialConditionBuilder(engine_config_offline)
    reference_batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")

    netcdf_path = tmp_path / "input.nc"
    write_batch_netcdf(reference_batch, netcdf_path)

    with torch.inference_mode():
        direct = engine.predict(reference_batch)
        from_netcdf = engine.run_from_netcdf(netcdf_path, steps=1)[0]

    assert_batches_close(direct, from_netcdf, atol=1e-4)


def test_run_from_netcdf_requires_existing_file(engine_config_offline, tmp_path: Path) -> None:
    engine = AuroraEngine(engine_config_offline)
    with pytest.raises(FileNotFoundError):
        engine.run_from_netcdf(tmp_path / "missing.nc")
