from pathlib import Path

from flash_aurora.engine.core.engine import AuroraEngine


def test_from_preset_accepts_lifecycle_optimizations(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        overlap_ic_load=False,
        async_export=True,
        export_pool_size=4,
        export_max_inflight=2,
        export_use_egress_stream=False,
    )
    assert engine.config.overlap_ic_load is False
    assert engine.config.async_export is True
    assert engine.config.export_pool_size == 4
    assert engine.config.export_max_inflight == 2
    assert engine.config.export_use_egress_stream is False


def test_lifecycle_defaults(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    assert engine.config.overlap_ic_load is True
    assert engine.config.async_export is False
    assert engine.config.export_pool_size == 2
    assert engine.config.export_use_egress_stream is True


def test_prepare_overlap_follows_config(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        overlap_ic_load=False,
    )
    assert engine._resolve_overlap(None) is False
    assert engine._resolve_overlap(True) is True


def test_rollout_async_export_follows_config(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        async_export=True,
    )
    assert engine._resolve_async_export(None) is True
    assert engine._resolve_async_export(False) is False
