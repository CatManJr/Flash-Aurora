from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.engine.core.engine import AuroraEngine


def test_forward_warmup_iters_default(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    assert engine.config.forward_warmup_iters == 2


def test_from_preset_accepts_forward_warmup_iters(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        forward_warmup_iters=0,
    )
    assert engine.config.forward_warmup_iters == 0


def test_maybe_warmup_skips_when_disabled(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        forward_warmup_iters=0,
    )
    engine._model = MagicMock()
    with patch.object(engine._graph_pool, "warmup") as warmup:
        engine._maybe_warmup(MagicMock())
        warmup.assert_not_called()


def test_maybe_warmup_runs_once(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    engine._model = MagicMock()
    batch = MagicMock()
    with patch.object(engine._graph_pool, "warmup") as warmup:
        engine._maybe_warmup(batch)
        engine._maybe_warmup(batch)
        warmup.assert_called_once()


def test_warmup_explicit_iters(tmp_path: Path) -> None:
    engine = AuroraEngine.from_preset(
        "era5_pretrained",
        asset_root=tmp_path,
        forward_warmup_iters=0,
    )
    engine._model = MagicMock()
    batch = MagicMock()
    with patch.object(engine._graph_pool, "warmup") as warmup:
        engine.warmup(batch, forward_iters=3)
        warmup.assert_called_once()
        assert engine._forward_warmed is True
