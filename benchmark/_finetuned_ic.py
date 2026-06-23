"""Load real ingress ``Batch`` objects for finetuned Aurora presets."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.download import DataDownloader
from flash_aurora.engine.ingress.download.layout import cache_subdir

# Finetuned presets with LoRA (excludes ``wave``: MARS archive often unavailable).
FINETUNED_LORA_PRESETS: tuple[str, ...] = (
    "hres_t0_finetuned",
    "hres_0.1",
    "cams",
    "tc_tracking",
)

_DEFAULT_VALID_TIME: dict[str, datetime] = {
    "hres_t0_finetuned": datetime(2022, 5, 11, 6),
    "hres_0.1": datetime(2022, 5, 11, 6),
    "cams": datetime(2022, 6, 11, 12),
    # ``tc_tracking`` shares the 0.25° finetuned ckpt with ``hres_t0_finetuned``.
    "tc_tracking": datetime(2022, 9, 17, 12),
}

_DEFAULT_TIME_INDEX: dict[str, int] = {
    "hres_t0_finetuned": 1,
    "hres_0.1": 1,
    "cams": 1,
    "tc_tracking": 1,
}


def preset_engine_config(preset_name: str, asset_root: Path) -> EngineConfig:
    try:
        base = DEFAULT_PRESETS.get(preset_name)
    except KeyError as exc:
        raise KeyError(
            f"Unknown preset {preset_name!r}. Available: {', '.join(DEFAULT_PRESETS.names())}"
        ) from exc
    return replace(
        base,
        asset_root=asset_root.expanduser().resolve(),
        allow_hub_download=False,
    )


def load_preset_batch(
    preset_name: str,
    asset_root: Path,
    *,
    valid_time: datetime | None = None,
    time_index: int | None = None,
) -> tuple[Any, EngineConfig]:
    """Build IC from cached ingress NetCDF (no download)."""
    config = preset_engine_config(preset_name, asset_root)
    vt = valid_time or _DEFAULT_VALID_TIME[preset_name]
    ti = _DEFAULT_TIME_INDEX[preset_name] if time_index is None else time_index
    cache = asset_root.expanduser().resolve() / cache_subdir(config.source)
    downloader = DataDownloader(config)
    missing = downloader.missing(vt, cache_dir=cache)
    if missing:
        raise FileNotFoundError(
            f"Incomplete ingress cache for preset {preset_name!r} at {cache}: missing {missing}"
        )
    request = downloader.ingest_request(vt, cache_dir=cache, time_index=ti, download=False)
    batch = InitialConditionBuilder(config).from_source(request)
    return batch, config


def checkpoint_path(config: EngineConfig, asset_root: Path) -> Path:
    root = asset_root.expanduser().resolve()
    if config.checkpoint_path is not None:
        return config.checkpoint_path.expanduser().resolve()
    return root / config.variant.checkpoint_filename
