"""Load real ingress ``Batch`` objects for any engine preset (except ``wave``)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from flash_aurora.engine.core.config import (
    CAMS_ATMOS_POLLUTION,
    CAMS_SURF_POLLUTION,
    EngineConfig,
)
from flash_aurora.engine.core.presets import DEFAULT_PRESETS
from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder
from flash_aurora.engine.ingress.download import DataDownloader
from flash_aurora.engine.ingress.download.layout import cache_subdir

# All presets except ``wave`` (MARS/WAM ingress often unavailable).
PRECISION_PRESETS: tuple[str, ...] = tuple(
    name for name in DEFAULT_PRESETS.names() if name != "wave"
)

_DEFAULT_VALID_TIME: dict[str, datetime] = {
    "era5_pretrained": datetime(2023, 1, 1, 6),
    "small_pretrained": datetime(2023, 1, 1, 6),
    "hres_t0_finetuned": datetime(2022, 5, 11, 6),
    "hres_0.1": datetime(2022, 5, 11, 6),
    "cams": datetime(2022, 6, 11, 12),
    "tc_tracking": datetime(2022, 9, 17, 12),
}

_DEFAULT_TIME_INDEX: dict[str, int] = {name: 1 for name in PRECISION_PRESETS}


def preset_engine_config(preset_name: str, asset_root: Path) -> EngineConfig:
    try:
        base = DEFAULT_PRESETS.get(preset_name)
    except KeyError as exc:
        raise KeyError(
            f"Unknown preset {preset_name!r}. Available: {', '.join(PRECISION_PRESETS)}"
        ) from exc
    return replace(
        base,
        asset_root=asset_root.expanduser().resolve(),
        allow_hub_download=False,
    )


def load_small_pretrained_batch(asset_root: Path) -> tuple[Any, EngineConfig]:
    """HF/Microsoft test pickle (400x800, 7 levels) — not compatible with full ERA5 cache."""
    import os
    import pickle
    import warnings

    from flash_aurora.aurora import Batch, Metadata
    from flash_aurora.aurora.batch import interpolate_numpy

    root = asset_root.expanduser().resolve()
    input_path = root / "aurora-0.25-small-pretrained-test-input.pickle"
    static_path = root / "aurora-0.25-static.pickle"
    if not input_path.is_file():
        raise FileNotFoundError(f"missing small test input: {input_path}")
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static vars: {static_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(input_path, "rb") as f:
            test_input = pickle.load(f)
        with open(static_path, "rb") as f:
            static_vars = pickle.load(f)

    if os.name == "nt":

        class PatchedDateTime(datetime):
            def timestamp(self) -> float:
                return -631134000.0

        test_input["metadata"]["time"] = [PatchedDateTime(1950, 1, 1, 6, 0)]

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    import torch

    batch = Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
            rollout_step=0,
        ),
    )
    return batch, preset_engine_config("small_pretrained", asset_root)


def load_preset_batch(
    preset_name: str,
    asset_root: Path,
    *,
    valid_time: datetime | None = None,
    time_index: int | None = None,
) -> tuple[Any, EngineConfig]:
    """Build IC from cached ingress NetCDF (no download)."""
    if preset_name not in PRECISION_PRESETS:
        raise KeyError(f"Preset {preset_name!r} not in PRECISION_PRESETS")
    if preset_name == "small_pretrained":
        return load_small_pretrained_batch(asset_root)
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


def output_var_tolerances(config: EngineConfig) -> tuple[tuple[str, str, float], ...]:
    """Per-output-variable official or heuristic tolerances for mean-relative error."""
    from _pretrained_era5 import _OFFICIAL_TOLERANCES  # noqa: PLC0415

    standard = dict(_OFFICIAL_TOLERANCES)
    pollution_tol = 5e-3
    specs: list[tuple[str, str, float]] = []
    for name in config.variant.surf_vars:
        tol = standard.get(name, pollution_tol)
        specs.append(("surf_vars", name, tol))
    for name in config.variant.atmos_vars:
        tol = standard.get(name, pollution_tol)
        specs.append(("atmos_vars", name, tol))
    return tuple(specs)


def pollution_var_names() -> frozenset[str]:
    return frozenset(CAMS_SURF_POLLUTION + CAMS_ATMOS_POLLUTION)
