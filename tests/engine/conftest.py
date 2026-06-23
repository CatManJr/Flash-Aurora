from __future__ import annotations

from pathlib import Path

import pytest

from flash_aurora.engine.core.asset_root import resolve_asset_root
from flash_aurora.engine.core.config import EngineConfig


@pytest.fixture(scope="session")
def asset_root() -> Path:
    root = resolve_asset_root()
    if root is None or not root.is_dir() or not any(root.iterdir()):
        pytest.skip(
            "Set AURORA_ASSET_ROOT to a data-disk directory with checkpoints and "
            "cached ingress (not ./fetched under the repo on the system drive)"
        )
    return root


@pytest.fixture(scope="session")
def engine_config_offline(asset_root: Path) -> EngineConfig:
    from flash_aurora.engine.core.presets import DEFAULT_PRESETS

    config = DEFAULT_PRESETS.get("small_pretrained")
    config.asset_root = asset_root
    config.allow_hub_download = False
    return config
