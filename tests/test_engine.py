from engine.core.presets import DEFAULT_PRESETS
from engine.core.engine import AuroraEngine


def test_from_preset_returns_engine() -> None:
    engine = AuroraEngine.from_preset("era5_pretrained")
    assert engine.config.variant.model_class == "AuroraPretrained"
    assert engine.config.source.schema == "cds_era5"


def test_preset_registry_lists_examples() -> None:
    names = set(DEFAULT_PRESETS.names())
    assert {"era5_pretrained", "hres_t0_finetuned", "wave", "tc_tracking"}.issubset(names)
