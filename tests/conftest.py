from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.config import EngineConfig


@pytest.fixture(scope="session")
def asset_root() -> Path:
    import os

    env = os.environ.get("AURORA_HF_LOCAL_DIR") or os.environ.get("FLASH_AURORA_ASSET_ROOT")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            return path
    default = Path.cwd() / "fetched"
    if default.is_dir() and any(default.iterdir()):
        return default
    pytest.skip("Set asset_root, AURORA_HF_LOCAL_DIR, or populate ./fetched for integration tests")


@pytest.fixture(scope="session")
def engine_config_offline(asset_root: Path) -> EngineConfig:
    from engine.core.presets import DEFAULT_PRESETS

    config = DEFAULT_PRESETS.get("small_pretrained")
    config.asset_root = asset_root
    config.allow_hub_download = False
    return config
