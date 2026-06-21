from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.config import EngineConfig


@pytest.fixture(scope="session")
def asset_root() -> Path:
    default = Path.cwd() / "fetched"
    if default.is_dir() and any(default.iterdir()):
        return default.resolve()
    pytest.skip("Populate ./fetched for integration tests or run unit tests only")


@pytest.fixture(scope="session")
def engine_config_offline(asset_root: Path) -> EngineConfig:
    from engine.core.presets import DEFAULT_PRESETS

    config = DEFAULT_PRESETS.get("small_pretrained")
    config.asset_root = asset_root
    config.allow_hub_download = False
    return config
