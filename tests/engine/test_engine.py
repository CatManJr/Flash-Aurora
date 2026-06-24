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


def test_default_export_dir_is_under_asset_root(tmp_path: Path) -> None:
    assets = tmp_path / "data"
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=assets)
    assert engine._resolved_export_dir() == (assets / "output").resolve()


def test_engine_close_releases_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    engine._model = object()
    engine._forward_warmed = True
    engine._graph_pool._captured["shape"] = object()
    released_with: bool | None = None

    def release_gpu(*, move_model_to_cpu: bool = True) -> None:
        nonlocal released_with
        released_with = move_model_to_cpu

    monkeypatch.setattr(engine, "release_gpu", release_gpu)

    engine.close()
    engine.close()

    assert released_with is False
    assert engine._model is None
    assert engine._forward_warmed is False
    assert engine._graph_pool._captured == {}


def test_engine_load_failure_releases_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    released = False

    def fail_load():
        raise RuntimeError("load failed")

    def release_gpu(*, move_model_to_cpu: bool = True) -> None:
        nonlocal released
        released = move_model_to_cpu

    monkeypatch.setattr(engine, "_load_model_to_device", fail_load)
    monkeypatch.setattr(engine, "release_gpu", release_gpu)

    with pytest.raises(RuntimeError, match="load failed"):
        engine.load()

    assert released is True


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


def test_build_model_enables_lora_merged_inference_for_finetuned_presets(tmp_path: Path) -> None:
    finetuned = AuroraEngine.from_preset("hres_t0_finetuned", asset_root=tmp_path)
    pretrained = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    finetuned_attn = finetuned._loader.build_model().backbone.encoder_layers[0].blocks[0].attn
    pretrained_attn = pretrained._loader.build_model().backbone.encoder_layers[0].blocks[0].attn
    assert finetuned_attn.use_lora_merged_inference is True
    assert pretrained_attn.use_lora_merged_inference is False
