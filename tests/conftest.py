from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine.core.config import EngineConfig


@pytest.fixture(scope="session")
def asset_root() -> Path | None:
    env = os.environ.get("AURORA_HF_LOCAL_DIR") or os.environ.get("FLASH_AURORA_ASSET_ROOT")
    if not env:
        return None
    path = Path(env).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"asset root is not a directory: {path}")
    return path


@pytest.fixture(scope="session")
def engine_config_offline(asset_root: Path | None) -> EngineConfig:
    if asset_root is None:
        pytest.skip("Set AURORA_HF_LOCAL_DIR for integration tests")
    from engine.core.presets import DEFAULT_PRESETS

    config = DEFAULT_PRESETS.get("small_pretrained")
    config.asset_root = asset_root
    config.allow_hub_download = False
    return config
