from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.presets import DEFAULT_PRESETS
from engine.ingress.build_ic import InitialConditionBuilder


def test_from_pickle_requires_existing_file(tmp_path: Path) -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    config.asset_root = tmp_path
    config.allow_hub_download = False
    builder = InitialConditionBuilder(config)

    with pytest.raises(FileNotFoundError):
        builder.from_pickle("missing.pickle")
