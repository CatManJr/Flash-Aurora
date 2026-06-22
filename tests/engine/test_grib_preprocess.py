from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.adapters.hres_analysis import GribHresAnalysisAdapter
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.download import grib_preprocess


def test_require_cfgrib_raises_clear_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cfgrib":
            raise ImportError("no cfgrib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="uv pip install cfgrib"):
        grib_preprocess.require_cfgrib()


def test_adapter_materializes_netcdf_before_ingest(tmp_path: Path) -> None:
    config = EngineConfig(
        variant=DEFAULT_PRESETS.get("hres_0.1").variant,
        source=DEFAULT_PRESETS.get("hres_0.1").source,
        asset_root=tmp_path,
        allow_hub_download=False,
    )
    cache = tmp_path / "hres_0.1"
    cache.mkdir()
    day = "2022-05-11"
    (cache / f"surf_2t_{day}.grib").write_bytes(b"grib")
    (cache / f"atmos_t_{day}_00.grib").write_bytes(b"grib")

    adapter = GribHresAnalysisAdapter()
    request = IngestRequest(valid_time=datetime(2022, 5, 11, 6), cache_dir=cache)

    with patch(
        "flash_aurora.engine.ingress.adapters.hres_analysis.materialize_hres_01_netcdf"
    ) as materialize, patch.object(adapter, "_build_from_netcdf") as from_nc, patch.object(
        adapter, "_has_netcdf_cache", side_effect=[False, True]
    ), patch(
        "flash_aurora.engine.ingress.adapters.hres_analysis.StaticFieldLoader"
    ) as loader_cls:
        mock_batch = MagicMock()
        mock_batch.regrid.return_value = mock_batch
        from_nc.return_value = mock_batch
        loader_cls.return_value.load.return_value = {}
        batch = adapter.build_initial_batch(request, config)

    materialize.assert_called_once_with(cache, day, levels=config.variant.levels)
    from_nc.assert_called_once()
    assert batch is mock_batch
