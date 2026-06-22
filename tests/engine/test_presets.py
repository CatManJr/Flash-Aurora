from __future__ import annotations

from flash_aurora.engine.core.presets import DEFAULT_PRESETS


def test_preset_names() -> None:
    names = DEFAULT_PRESETS.names()
    assert "era5_pretrained" in names
    assert "hres_t0_finetuned" in names
    assert "small_pretrained" in names
    assert "wave" in names


def test_presets_use_relative_asset_names() -> None:
    for name, config in DEFAULT_PRESETS.items():
        variant = config.variant
        assert not variant.checkpoint_filename.startswith("/")
        assert not variant.static_pickle.startswith("/")
        assert variant.hf_repo == "microsoft/aurora"


def test_get_returns_copy() -> None:
    first = DEFAULT_PRESETS.get("era5_pretrained")
    second = DEFAULT_PRESETS.get("era5_pretrained")
    first.device = "cpu"
    assert second.device != "cpu"


def test_unknown_preset_raises() -> None:
    try:
        DEFAULT_PRESETS.get("missing")
        assert False, "expected KeyError"
    except KeyError:
        pass
