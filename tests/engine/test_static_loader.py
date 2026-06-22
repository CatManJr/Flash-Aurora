from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from flash_aurora.engine.core.config import CAMS_STATIC
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.static import StaticFieldLoader


def test_static_loader_fetches_missing_pickle_despite_local_checkpoint_policy(tmp_path: Path) -> None:
    config = replace(
        DEFAULT_PRESETS.get("cams"),
        asset_root=tmp_path,
        allow_hub_download=False,
    )
    store = AssetStore(root=tmp_path)
    pickle_path = tmp_path / "aurora-0.4-air-pollution-static.pickle"

    def fake_fetch(self, *args, **kwargs):
        assert kwargs["allow_download"] is True
        payload = {name: np.zeros((3, 4)) for name in CAMS_STATIC}
        pickle_path.write_bytes(pickle.dumps(payload))
        return pickle_path

    loader = StaticFieldLoader(config, store)
    with patch(
        "flash_aurora.engine.ingress.static.AssetStore.fetch_hub_file",
        side_effect=fake_fetch,
    ):
        static = loader.load()

    assert set(static) == set(CAMS_STATIC)
