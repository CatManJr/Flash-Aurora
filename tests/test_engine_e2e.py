from __future__ import annotations

from pathlib import Path

import pytest

from engine import AuroraEngine
from engine.ingress.build_ic import InitialConditionBuilder


@pytest.mark.integration
def test_small_pretrained_rollout_export(
    engine_config_offline,
    asset_root: Path,
    tmp_path: Path,
) -> None:
    engine_config_offline.export_dir = tmp_path / "output"
    engine = AuroraEngine(engine_config_offline)
    engine.load()

    builder = InitialConditionBuilder(engine_config_offline)
    batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")

    paths = list(engine.rollout_and_export(batch, steps=1))
    assert len(paths) == 1
    assert paths[0].name == "prediction-000.nc"
    assert paths[0].is_file()
