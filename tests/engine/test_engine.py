from pathlib import Path

import pytest

from flash_aurora.engine.core.engine import AuroraEngine
from flash_aurora.engine.core.presets import DEFAULT_PRESETS


def test_from_preset_returns_engine(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    assert engine.config.variant.model_class == "AuroraPretrained"
    assert engine.config.source.schema == "cds_era5"
    assert engine.fetched_dir == tmp_path.resolve()


def test_engine_captures_user_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets"
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=assets)
    assert engine.config.user_cwd == tmp_path.resolve()
    assert engine.fetched_dir == assets.resolve()


def test_preset_registry_lists_examples() -> None:
    names = set(DEFAULT_PRESETS.names())
    assert {"era5_pretrained", "hres_t0_finetuned", "wave", "tc_tracking"}.issubset(names)
