from pathlib import Path

import pytest

from engine.core.engine import AuroraEngine
from engine.core.presets import DEFAULT_PRESETS


def test_from_preset_returns_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    engine = AuroraEngine.from_preset("era5_pretrained")
    assert engine.config.variant.model_class == "AuroraPretrained"
    assert engine.config.source.schema == "cds_era5"
    assert engine.fetched_dir.name == "fetched"


def test_engine_captures_user_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURORA_HF_LOCAL_DIR", raising=False)
    monkeypatch.delenv("FLASH_AURORA_ASSET_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    engine = AuroraEngine.from_preset("era5_pretrained")
    assert engine.config.user_cwd == tmp_path.resolve()
    assert engine.fetched_dir == (tmp_path / "fetched").resolve()


def test_preset_registry_lists_examples() -> None:
    names = set(DEFAULT_PRESETS.names())
    assert {"era5_pretrained", "hres_t0_finetuned", "wave", "tc_tracking"}.issubset(names)
