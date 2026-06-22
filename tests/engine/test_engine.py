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


def test_set_export_dir_updates_rollout_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=assets)
    target = tmp_path / "exports" / "era5"
    resolved = engine.set_export_dir(target)
    assert resolved == target.resolve()
    assert engine.config.export_dir == target.resolve()


def test_build_model_applies_inference_precision(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        inference_precision="bf16_mixed@fp32",
    )
    model = engine._loader.build_model()
    assert model.inference_config is not None
    assert model.inference_config.config_label == "bf16_mixed@fp32"
    block = model.backbone.encoder_layers[0].blocks[0]
    assert block.attn.use_cute_window_attn is True
