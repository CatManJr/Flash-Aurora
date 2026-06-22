#!/usr/bin/env python3
"""Generate flash-aurora docs example notebooks (no cell outputs)."""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Default asset root for tutorial notebooks (team data disk; outputs are checked in).
TUTORIAL_ASSET_ROOT = "/root/autodl-tmp/aurora"

TUTORIAL_DISK_BLURB = (
    "> **Asset root:** default is `./assets` under the working directory. "
    f"To reuse a team data disk, uncomment `ASSET_ROOT = Path(\"{TUTORIAL_ASSET_ROOT}\")` in the setup cell. "
    "Saved notebook outputs may show whichever path was active when the tutorial was run.\n"
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def nb(*cells) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": list(cells),
    }


def write_nb(name: str, cells) -> None:
    path = DOCS / name
    path.write_text(json.dumps(nb(*cells), indent=1, ensure_ascii=False) + "\n")
    print("wrote", path)


SETUP_PREAMBLE = """from datetime import datetime
from pathlib import Path

from flash_aurora.engine import (
    DEFAULT_PRESETS,
    DataDownloader,
    HF_MIRROR_ENDPOINT,
)
from flash_aurora.engine.core.redaction import safe_path

PRESET = "{preset}"
VALID_TIME = datetime({valid_time})
TIME_INDEX = {time_index}
ROLLOUT_STEPS = {rollout_steps}

# Named tier or combo: backbone@encoder_decoder (see README).
INFERENCE_PRECISION = "{inference_precision}"

# Default: ./assets under the notebook working directory (created if missing).
ASSET_ROOT: Path | str | None = None

# Optional — absolute path to a mounted data disk with checkpoints/cache (uncomment to use):
# ASSET_ROOT = Path("{tutorial_asset_root}")

if ASSET_ROOT is not None:
    root = Path(ASSET_ROOT).expanduser()
    if not root.is_absolute():
        raise ValueError("ASSET_ROOT must be an absolute path")
    ASSET_ROOT = root.resolve()
else:
    ASSET_ROOT = (Path.cwd() / "assets").resolve()
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

variant = DEFAULT_PRESETS.get(PRESET).variant
CHECKPOINT_PATH = ASSET_ROOT / variant.checkpoint_filename
USE_HF_MIRROR = False  # True -> https://hf-mirror.com when huggingface.co is blocked

if CHECKPOINT_PATH.is_file():
    CHECKPOINT_ARG = CHECKPOINT_PATH
    ALLOW_HUB_DOWNLOAD = False
    HF_ENDPOINT = None
    print("checkpoint: local", safe_path(CHECKPOINT_PATH))
else:
    CHECKPOINT_ARG = None
    ALLOW_HUB_DOWNLOAD = True
    HF_ENDPOINT = HF_MIRROR_ENDPOINT if USE_HF_MIRROR else None
    print("checkpoint: missing locally; will download from Hugging Face")
    print("  target dir:", safe_path(ASSET_ROOT))
    print("  filename:", variant.checkpoint_filename)
    print("  hf_endpoint:", HF_ENDPOINT or "https://huggingface.co (default)")

downloader = DataDownloader.from_preset(PRESET, asset_root=ASSET_ROOT)
cache_dir = downloader.resolve_cache_dir()

print("cache_dir:", safe_path(cache_dir))
print("asset_root:", safe_path(ASSET_ROOT))
print("allow_hub_download:", ALLOW_HUB_DOWNLOAD)"""

SETUP_STATIC_PICKLE = """
from flash_aurora.engine.core.hub import HubDownloadOptions
from flash_aurora.engine.core.paths import AssetStore

# Static fields from Hugging Face (file lives in ASSET_ROOT, not under the ingress cache).
STATIC_PICKLE_PATH = ASSET_ROOT / variant.static_pickle
HF_OPTIONS = HubDownloadOptions(
    endpoint=HF_MIRROR_ENDPOINT if USE_HF_MIRROR else HF_ENDPOINT,
)
if STATIC_PICKLE_PATH.is_file():
    print("static_pickle: local", safe_path(STATIC_PICKLE_PATH))
else:
    print("static_pickle: missing locally; will download from Hugging Face")
    print("  target dir:", safe_path(ASSET_ROOT))
    print("  filename:", variant.static_pickle)
    print("  hf_endpoint:", HF_OPTIONS.endpoint or "https://huggingface.co (default)")
    STATIC_PICKLE_PATH = AssetStore(root=ASSET_ROOT).fetch_hub_file(
        variant.static_pickle,
        repo=variant.hf_repo,
        allow_download=True,
        explicit=ASSET_ROOT,
        hub=HF_OPTIONS,
    )
    print("static_pickle: ready", safe_path(STATIC_PICKLE_PATH))"""


def setup_cell(**fmt: str) -> str:
    return SETUP_PREAMBLE.format(**fmt) + "\n" + SETUP_STATIC_PICKLE

LOAD_ROLLOUT = """import torch

from flash_aurora.engine import AuroraEngine, InitialConditionBuilder
from flash_aurora.aurora.model.inference_precision import describe_inference_config

engine = AuroraEngine.from_preset(
    PRESET,
    asset_root=ASSET_ROOT,
    checkpoint_path=CHECKPOINT_ARG,
    allow_hub_download=ALLOW_HUB_DOWNLOAD,
    hf_mirror=USE_HF_MIRROR,
    hf_endpoint=None if USE_HF_MIRROR else HF_ENDPOINT,
)

# Must be set before load() — see README "Inference precision tiers".
engine.config.inference_precision = INFERENCE_PRECISION
engine.config.gpu_rollout_steps = ROLLOUT_STEPS

request = downloader.ingest_request(
    VALID_TIME,
    time_index=TIME_INDEX,
    download=False,
)
builder = InitialConditionBuilder(engine.config)
batch = builder.from_source(request)
print("IC time:", batch.metadata.time)
print("spatial:", batch.spatial_shape)

engine.load()
print("device:", next(engine.model.parameters()).device)

cfg = engine.model.inference_config
if cfg is not None:
    print("inference tier:", cfg.config_label)
    print(describe_inference_config(cfg))

preds = []
with torch.inference_mode():
    for pred in engine.rollout_stream(batch, ROLLOUT_STEPS):
        preds.append(pred.to("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("predictions:", [str(p.metadata.time[0]) for p in preds])
engine.release_gpu()"""


def rollout_cell() -> str:
    return LOAD_ROLLOUT


EXPORT_OPTIONAL = """# Model already on GPU from section 3.
EXPORT_DIR = ASSET_ROOT / "output" / PRESET
paths = list(engine.rollout_and_export(batch, steps=ROLLOUT_STEPS, export_dir=EXPORT_DIR))
for p in paths:
    print(safe_path(p), p.is_file())"""


def example_hres_0_1() -> None:
    write_nb(
        "example_hres_0.1.ipynb",
        [
            md(
                """# HRES 0.1° Analysis — flash-aurora Engine

Same setup as the upstream [Microsoft Aurora HRES 0.1° example](microsoft-aurora/docs/example_hres_0.1.ipynb):

- **Date:** 2022-05-11
- **Initial condition:** 06:00 UTC (hours 00:00 + 06:00)
- **Rollout:** 2 steps → 12:00 and 18:00 UTC
- **Model:** `AuroraHighRes` / `aurora-0.1-finetuned` via preset `hres_0.1`

The Engine **`GribHresAnalysisAdapter`** accepts either:

1. **GRIB files** downloaded from UCAR RDA (same URLs as upstream), or
2. **NetCDF cache** under `<ASSET_ROOT>/hres_0.1/` (`{day}-surface-level.nc`, `{day}-atmospheric-00.nc`, `{day}-atmospheric-06.nc`).

It then **`batch.regrid(res=0.1)`** and injects static fields from `aurora-0.1-static.pickle` in `ASSET_ROOT`.

## Prerequisites

1. **Extra packages:** `uv pip install cfgrib` (GRIB ingress; `requests` is used by the downloader).
2. **Checkpoint** and **`aurora-0.1-static.pickle`** under `ASSET_ROOT`, or Hugging Face download in the setup cell.
3. **GPU with sufficient VRAM** — 0.1° global grid is ~1801×3600; `AuroraEngine` uses a cross-process **GPU guard** (queue / share) under `<ASSET_ROOT>/.flash-aurora/gpu_guard/`."""
            ),
            code(
                setup_cell(
                    tutorial_asset_root=TUTORIAL_ASSET_ROOT,
                    preset="hres_0.1",
                    valid_time="2022, 5, 11, 6",
                    time_index=1,
                    rollout_steps=2,
                    inference_precision="bf16_mixed@fp32",
                )
            ),
            md(
                """## 1. Download IFS HRES 0.1° analysis (UCAR RDA)

`DataDownloader.ensure()` fetches the upstream UCAR RDA GRIB set into `<ASSET_ROOT>/hres_0.1/`:

- Surface/static: `surf_{var}_YYYY-MM-DD.grib`
- Atmospheric: `atmos_{var}_YYYY-MM-DD_HH.grib` for HH ∈ {00, 06, 12, 18}

Re-run is safe: existing files are skipped."""
            ),
            code(
                """from pathlib import Path

from flash_aurora.engine import DownloadResult
from flash_aurora.engine.core.redaction import safe_path

missing = downloader.missing(VALID_TIME)

if missing:
    print("Missing GRIB files:", missing)
    result = downloader.ensure(VALID_TIME)
else:
    print("HRES 0.1 GRIB cache already complete under", safe_path(cache_dir))
    result = DownloadResult(
        cache_dir=cache_dir,
        paths=downloader.expected_paths(VALID_TIME),
        downloaded=(),
        skipped=tuple(downloader.expected_paths(VALID_TIME)),
    )

print("downloaded:", result.downloaded)
print("skipped:", result.skipped)
for key, path in result.paths.items():
    print(f"  {key}: {safe_path(path)}")"""
            ),
            md(
                """## 2. Build initial condition

`InitialConditionBuilder` detects the GRIB cache, loads 00:00 + 06:00 fields, regrids to 0.1°, and attaches HF static pickle fields."""
            ),
            md(
                """## 3. Load model and rollout

0.1° inference is memory-intensive. `engine.load()` acquires a GPU lease via the cross-process guard (small presets may share; `hres_0.1` queues until ~48 GiB is free). Call `engine.release_gpu()` when finished. Start with `bf16_mixed@fp32`."""
            ),
            code(rollout_cell()),
            md(
                """## 4. Visualize: Aurora vs HRES analysis

Compare predicted 2 m temperature against the downloaded `surf_2t` GRIB at forecast valid times."""
            ),
            code(
                """import matplotlib.pyplot as plt
import xarray as xr

truth = xr.open_dataset(
    cache_dir / VALID_TIME.strftime("surf_2t_%Y-%m-%d.grib"),
    engine="cfgrib",
)

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction (flash-aurora Engine)")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ref = truth["t2m"][2 + i].values
    ax[i, 1].imshow(ref - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("HRES Analysis")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.show()"""
            ),
            md(
                """## 5. (Optional) Export rollout to NetCDF

Writes one file per step under `<ASSET_ROOT>/output/hres_0.1/`."""
            ),
            code(EXPORT_OPTIONAL),
        ],
    )


def example_cams() -> None:
    write_nb(
        "example_cams.ipynb",
        [
            md(
                f"""# CAMS Air Pollution — flash-aurora Engine

Same setup as the upstream [Microsoft Aurora CAMS example](microsoft-aurora/docs/example_cams.ipynb):

- **Date:** 2022-06-11 (analysis at UTC 12:00 from 00:00 + 12:00 inputs)
- **Rollout:** 4 steps → 12 Jun 00/12 and 13 Jun 00/12 UTC
- **Model:** `AuroraAirPollution` via preset `cams`

`CamsAdapter` reads cached NetCDF under `<ASSET_ROOT>/cams/` and loads pollution static fields from `aurora-0.4-air-pollution-static.pickle` in `ASSET_ROOT`.

> **ADS endpoint:** CAMS uses the [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/). `DataDownloader` always talks to the ADS API URL for you—the same Copernicus UID key as CDS works; you do **not** need to edit `url` in `~/.cdsapirc`.

{TUTORIAL_DISK_BLURB}
## Prerequisites

1. **ADS account** and accepted dataset terms for [CAMS global composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts).
2. **ADS credentials** (any one): `ADSAPI_KEY`, `CDSAPI_KEY` (same UID key), `~/.cdsapirc` (`key:` line only), or interactive `getpass` below.
3. **Checkpoint** and **`aurora-0.4-air-pollution-static.pickle`** under `ASSET_ROOT`, or Hugging Face download in the setup cell (independent of ADS ingress in section 1)."""
            ),
            code(
                setup_cell(
                    tutorial_asset_root=TUTORIAL_ASSET_ROOT,
                    preset="cams",
                    valid_time="2022, 6, 11, 12",
                    time_index=1,
                    rollout_steps=4,
                    inference_precision="bf16_mixed@fp32",
                )
            ),
            md(
                f"""## 1. Download CAMS (ADS API)

`DataDownloader.ensure()` retrieves the upstream CAMS zip from ADS, unpacks surface and atmospheric NetCDF files under `<ASSET_ROOT>/cams/`, and skips the API call when the cache is already complete."""
            ),
            code(
                """import getpass
import os
from pathlib import Path

from flash_aurora.engine import DownloadResult
from flash_aurora.engine.core.redaction import safe_path
from flash_aurora.engine.ingress.download.paths import read_cdsapirc_key

missing = downloader.missing(VALID_TIME)

if missing:
    # ADS credentials only when a download is actually needed.
    # 1) Environment: export ADSAPI_KEY=your-key  (or CDSAPI_KEY — same Copernicus UID)
    # 2) Inline assignment for local testing only — do not commit keys to git
    # 3) Interactive getpass — paste into the hidden input when prompted
    ADS_API_KEY = (
        os.environ.get("ADSAPI_KEY", "").strip()
        or os.environ.get("CDSAPI_KEY", "").strip()
        or read_cdsapirc_key()
        or None
    )
    # ADS_API_KEY = "paste-your-ads-key-here"
    if ADS_API_KEY is None:
        ADS_API_KEY = getpass.getpass("ADS API key (Copernicus UID): ").strip() or None

    if not ADS_API_KEY:
        raise ValueError(
            "No ADS credentials found. Set ADSAPI_KEY, CDSAPI_KEY, add key: to ~/.cdsapirc, "
            "or paste your key when prompted by getpass."
        )

    print("Missing CAMS files:", missing)
    result = downloader.ensure(VALID_TIME, ads_api_key=ADS_API_KEY)
else:
    print("CAMS cache already complete under", safe_path(cache_dir))
    result = DownloadResult(
        cache_dir=cache_dir,
        paths=downloader.expected_paths(VALID_TIME),
        downloaded=(),
        skipped=tuple(downloader.expected_paths(VALID_TIME)),
    )

print("downloaded:", result.downloaded)
print("skipped:", result.skipped)
for key, path in result.paths.items():
    print(f"  {key}: {safe_path(path)}")"""
            ),
            md(
                """## 2–3. Build IC, load model, rollout

`CamsAdapter` selects `forecast_period=0` (analysis). The batch uses both CAMS times on **2022-06-11** (UTC 00:00 + 12:00); IC metadata is **2022-06-11 UTC 12:00** (`TIME_INDEX=1`), same as upstream."""
            ),
            code(rollout_cell()),
            md(
                """## 4. Visualize pollution fields

Same layout as the upstream example: 2×2 panels of **Aurora predictions** for each rollout step (12 Jun 00/12 and 13 Jun 00/12 UTC). Units scaled like upstream; no ground-truth comparison."""
            ),
            code(
                """import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for i in range(4):
    ax = axs[i // 2, i % 2]
    pred = preds[i]
    ax.imshow(pred.surf_vars["tcno2"][0, 0].numpy() / 1e-6, vmin=0, vmax=10, cmap="Blues")
    ax.set_title(f"TC NO$_2$ {pred.metadata.time[0]}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for i in range(4):
    ax = axs[i // 2, i % 2]
    pred = preds[i]
    ax.imshow(pred.surf_vars["pm10"][0, 0].numpy() / 1e-9, vmin=0, vmax=400, cmap="Blues")
    ax.set_title(f"PM$_{{10}}$ {pred.metadata.time[0]}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()"""
            ),
            md("""## 5. (Optional) Export rollout to NetCDF"""),
            code(EXPORT_OPTIONAL),
        ],
    )


def example_wave() -> None:
    write_nb(
        "example_wave.ipynb",
        [
            md(
                """# Ocean Waves (HRES-WAM) — flash-aurora Engine

Same setup as the upstream [Microsoft Aurora Wave example](microsoft-aurora/docs/example_wave.ipynb):

- **Date:** 2022-09-16
- **Initial condition:** 06:00 UTC (00:00 + 06:00 history for met; same for WAM)
- **Rollout:** 2 steps → 12:00 and 18:00 UTC
- **Model:** `AuroraWave` via preset `wave`

Inputs combine **HRES-WAM** wave fields (MARS) and **HRES T0** meteorology (WeatherBench2). `DataDownloader` orchestrates both into `<ASSET_ROOT>/wave/`.

## Prerequisites

1. **Extra packages:** `uv pip install ecmwf-api-client gcsfs zarr cfgrib` (wave GRIB uses cfgrib for visualization).
2. **ECMWF MARS credentials** in `~/.ecmwfapirc` (see https://api.ecmwf.int/v1/key) **or** env vars `ECMWF_API_KEY`, `ECMWF_API_EMAIL`.
3. **Checkpoint** and **`aurora-0.25-wave-static.pickle`** under `ASSET_ROOT`, or Hugging Face download in the setup cell (not the `.nc` variant)."""
            ),
            code(
                setup_cell(
                    tutorial_asset_root=TUTORIAL_ASSET_ROOT,
                    preset="wave",
                    valid_time="2022, 9, 16, 6",
                    time_index=1,
                    rollout_steps=2,
                    inference_precision="bf16_mixed@fp32",
                )
            ),
            md(
                """## 1. Download HRES-WAM + HRES T0

`downloader.ensure()` writes:

| File | Source |
|------|--------|
| `{day}-wave.grib` | ECMWF MARS (`stream=wave`) |
| `{day}-surface-level.nc`, `{day}-atmospheric.nc` | WeatherBench2 HRES T0 |

Skip network calls when files already exist on the data disk."""
            ),
            code(
                """import getpass
import os
from pathlib import Path

from flash_aurora.engine import DownloadResult
from flash_aurora.engine.core.redaction import safe_path

missing = downloader.missing(VALID_TIME)

if missing:
    ECMWF_API_KEY = os.environ.get("ECMWF_API_KEY", "").strip() or None
    ECMWF_API_EMAIL = os.environ.get("ECMWF_API_EMAIL", "").strip() or None
    if ECMWF_API_KEY is None and not Path.home().joinpath(".ecmwfapirc").is_file():
        ECMWF_API_KEY = getpass.getpass("ECMWF API key (MARS): ").strip() or None
    if ECMWF_API_EMAIL is None and not Path.home().joinpath(".ecmwfapirc").is_file():
        ECMWF_API_EMAIL = getpass.getpass("ECMWF account email: ").strip() or None

    print("Missing cache files:", missing)
    result = downloader.ensure(
        VALID_TIME,
        ecmwf_api_key=ECMWF_API_KEY,
        ecmwf_email=ECMWF_API_EMAIL,
    )
else:
    print("Wave cache already complete under", safe_path(cache_dir))
    result = DownloadResult(
        cache_dir=cache_dir,
        paths=downloader.expected_paths(VALID_TIME),
        downloaded=(),
        skipped=tuple(downloader.expected_paths(VALID_TIME)),
    )

print("downloaded:", result.downloaded)
print("skipped:", result.skipped)
for key, path in result.paths.items():
    print(f"  {key}: {safe_path(path)}")"""
            ),
            md(
                """## 2–3. Build IC, load, rollout

`Wb2WamWaveAdapter` merges flipped HRES T0 met fields with WAM wave variables. Post-processing for near-zero wave height is applied at plot time (upstream Supplementary Information §C.5)."""
            ),
            code(rollout_cell()),
            md(
                """## 4. Visualize mean wave direction (MWD)

Left: Aurora forecast. Right: HRES-WAM reference. Directions masked when SWH < 1e-4 m."""
            ),
            code(
                """import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

wave_vars_ds = xr.open_dataset(
    result.paths["wave"],
    engine="cfgrib",
    backend_kwargs={"indexpath": ""},
)

fig, axs = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(axs.shape[0]):
    pred = preds[i]

    ax = axs[i, 0]
    ax.imshow(pred.surf_vars["mwd"][0, 0].numpy(), vmin=0, vmax=360, cmap="twilight")
    ax.set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax.set_title("Aurora Prediction (flash-aurora Engine)")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[i, 1]
    ref = wave_vars_ds["mwd"][2 + i].values
    ref[wave_vars_ds["swh"][2 + i].values < 1e-4] = np.nan
    ax.imshow(ref, vmin=0, vmax=360, cmap="twilight")
    if i == 0:
        ax.set_title("HRES-WAM")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()"""
            ),
            md("""## 5. (Optional) Export rollout to NetCDF"""),
            code(EXPORT_OPTIONAL),
        ],
    )


def example_tc_tracking() -> None:
    write_nb(
        "example_tc_tracking.ipynb",
        [
            md(
                """# Typhoon Nanmadol Track — flash-aurora Engine

Same setup as the upstream [Microsoft Aurora TC tracking example](microsoft-aurora/docs/example_tc_tracking.ipynb):

- **Date:** 2022-09-17
- **Initial condition:** 12:00 UTC (history 06:00 + 12:00 via `TIME_INDEX=2`)
- **Rollout:** 8 steps (~48 h)
- **Model:** `aurora-0.25-finetuned` via preset `tc_tracking`

Uses the same HRES T0 ingress as `hres_t0_finetuned`, but **`TIME_INDEX=2`** selects the third synoptic time as IC (matching upstream `_prepare` with `x[[1,2]]`).

After each step we call **`Tracker.step(pred)`** from `flash_aurora.aurora` to estimate typhoon center from MSL pressure."""
            ),
            code(
                SETUP_PREAMBLE.format(
                    tutorial_asset_root=TUTORIAL_ASSET_ROOT,
                    preset="tc_tracking",
                    valid_time="2022, 9, 17, 12",
                    time_index=2,
                    rollout_steps=8,
                    inference_precision="bf16_mixed@fp32",
                )
            ),
            md(
                """## 1. Download HRES T0 + ERA5 static

Identical cache layout to [example_hres_t0.ipynb](example_hres_t0.ipynb) under `<ASSET_ROOT>/hres_t0/`."""
            ),
            code(
                """import getpass
import os
from pathlib import Path

from flash_aurora.engine import DownloadResult
from flash_aurora.engine.core.redaction import safe_path

missing = downloader.missing(VALID_TIME)

if missing:
    CDS_API_KEY = os.environ.get("CDSAPI_KEY", "").strip() or None
    if CDS_API_KEY is None and not Path.home().joinpath(".cdsapirc").is_file():
        CDS_API_KEY = getpass.getpass("CDS API key (ERA5 static): ").strip() or None
    if not CDS_API_KEY and not Path.home().joinpath(".cdsapirc").is_file():
        raise ValueError("CDS credentials required for static.nc")
    print("Missing cache files:", missing)
    result = downloader.ensure(VALID_TIME, cds_api_key=CDS_API_KEY)
else:
    print("HRES T0 cache already complete under", safe_path(cache_dir))
    result = DownloadResult(
        cache_dir=cache_dir,
        paths=downloader.expected_paths(VALID_TIME),
        downloaded=(),
        skipped=tuple(downloader.expected_paths(VALID_TIME)),
    )

print("downloaded:", result.downloaded)
print("skipped:", result.skipped)"""
            ),
            md(
                """## 2–3. Build IC, load model, rollout with Tracker

Nanmadol position at 2022-09-17 12:00 UTC from IBTrACS (upstream notebook). We stream rollout steps and update the tracker after each prediction — same control flow as upstream, but via `engine.rollout_stream()`."""
            ),
            code(
                """import torch

from flash_aurora.aurora import Tracker
from flash_aurora.engine import AuroraEngine, InitialConditionBuilder
from flash_aurora.aurora.model.inference_precision import describe_inference_config

tracker = Tracker(
    init_lat=27.50,
    init_lon=132,
    init_time=datetime(2022, 9, 17, 12, 0),
)

engine = AuroraEngine.from_preset(
    PRESET,
    asset_root=ASSET_ROOT,
    checkpoint_path=CHECKPOINT_ARG,
    allow_hub_download=ALLOW_HUB_DOWNLOAD,
    hf_mirror=USE_HF_MIRROR,
    hf_endpoint=None if USE_HF_MIRROR else HF_ENDPOINT,
)
engine.config.inference_precision = INFERENCE_PRECISION

request = downloader.ingest_request(VALID_TIME, time_index=TIME_INDEX, download=False)
builder = InitialConditionBuilder(engine.config)
batch = builder.from_source(request)
print("IC time:", batch.metadata.time)

engine.load()
print("device:", next(engine.model.parameters()).device)
cfg = engine.model.inference_config
if cfg is not None:
    print("inference tier:", cfg.config_label)
    print(describe_inference_config(cfg))

preds = []
with torch.inference_mode():
    for pred in engine.rollout_stream(batch, ROLLOUT_STEPS):
        pred = pred.to("cpu")
        preds.append(pred)
        tracker.step(pred)

print("track points:", len(tracker.results()))"""
            ),
            md(
                """## 4. Visualize MSL and track

Eight panels around the West Pacific; black dots = full track, red dot = current step (upstream layout)."""
            ),
            code(
                """import matplotlib.pyplot as plt

track = tracker.results()

fig, axs = plt.subplots(2, 4, figsize=(10, 7))

for i in range(8):
    pred = preds[i]
    ax = axs[i // 4, i % 4]

    lat_mask = (pred.metadata.lat >= 20) & (pred.metadata.lat <= 45)
    lon_mask = (pred.metadata.lon >= 120) & (pred.metadata.lon <= 140)

    ax.imshow(
        pred.surf_vars["msl"][0, 0][lat_mask][:, lon_mask].numpy() / 100,
        vmin=970,
        vmax=1020,
        extent=(120, 140, 20, 45),
    )
    ax.set_title(str(pred.metadata.time[0]))
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(track.lon, track.lat, c="k", marker=".", markersize=8)
    this_step = track[track.time == pred.metadata.time[0]]
    ax.plot(this_step.lon, this_step.lat, c="r", marker=".", markersize=10)
    ax.text(
        0.05,
        0.95,
        f"Lat.: {this_step.lat.iloc[0]:.1f}$^\\circ$",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.875,
        f"Lon.: {this_step.lon.iloc[0]:.1f}$^\\circ$",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

plt.tight_layout()
plt.show()"""
            ),
        ],
    )


def fix_example_era5() -> None:
    path = DOCS / "example_era5.ipynb"
    data = json.loads(path.read_text())
    for cell in data["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "PRESET = \"era5_pretrained\"" in src and "ASSET_ROOT" in src:
            cell["source"] = [
                line + "\n"
                for line in (
                    """from datetime import datetime
from pathlib import Path

from flash_aurora.engine import (
    DEFAULT_PRESETS,
    DataDownloader,
    HF_MIRROR_ENDPOINT,
)
from flash_aurora.engine.core.redaction import safe_path

PRESET = "era5_pretrained"
DAY = "2023-01-01"
VALID_TIME = datetime(2023, 1, 1, 6)
TIME_INDEX = 1
ROLLOUT_STEPS = 2

# Named tier or combo: backbone@encoder_decoder (see README).
INFERENCE_PRECISION = "bf16_mixed@fp32"  # e.g. "tf32@tf32", "fp32", "bf16@fp32"

# Default: ./assets under the notebook working directory (created if missing).
ASSET_ROOT: Path | str | None = None

# Optional — absolute path to a mounted data disk with checkpoints/cache (uncomment to use):
# ASSET_ROOT = Path("{tutorial_asset_root}")

if ASSET_ROOT is not None:
    root = Path(ASSET_ROOT).expanduser()
    if not root.is_absolute():
        raise ValueError("ASSET_ROOT must be an absolute path")
    ASSET_ROOT = root.resolve()
else:
    ASSET_ROOT = (Path.cwd() / "assets").resolve()
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

variant = DEFAULT_PRESETS.get(PRESET).variant
CHECKPOINT_PATH = ASSET_ROOT / variant.checkpoint_filename
USE_HF_MIRROR = False  # True -> https://hf-mirror.com

if CHECKPOINT_PATH.is_file():
    # Local checkpoint on disk: load weights from ASSET_ROOT, skip Hub download.
    CHECKPOINT_ARG = CHECKPOINT_PATH
    ALLOW_HUB_DOWNLOAD = False
    HF_ENDPOINT = None
    print("checkpoint: local", safe_path(CHECKPOINT_PATH))
else:
    # No local checkpoint: engine.load() will fetch from Hugging Face into ASSET_ROOT.
    CHECKPOINT_ARG = None
    ALLOW_HUB_DOWNLOAD = True
    HF_ENDPOINT = HF_MIRROR_ENDPOINT if USE_HF_MIRROR else None
    print("checkpoint: missing locally; will download from Hugging Face")
    print("  target dir:", safe_path(ASSET_ROOT))
    print("  filename:", variant.checkpoint_filename)
    print("  hf_endpoint:", HF_ENDPOINT or "https://huggingface.co (default)")

downloader = DataDownloader.from_preset(PRESET, asset_root=ASSET_ROOT)
cache_dir = downloader.resolve_cache_dir()

print("cache_dir:", safe_path(cache_dir))
print("asset_root:", safe_path(ASSET_ROOT))
print("allow_hub_download:", ALLOW_HUB_DOWNLOAD)"""
                ).format(tutorial_asset_root=TUTORIAL_ASSET_ROOT).split("\n")
            ]
        if "engine.config.inference_precision = INFERENCE_PRECISION" in src and "describe_inference_config" in src:
            cell["source"] = [
                line + "\n"
                for line in (
                    """import torch

from flash_aurora.aurora.model.inference_precision import describe_inference_config

# Must be set before load() — see README "Inference precision tiers".
engine.config.inference_precision = INFERENCE_PRECISION

engine.load()
print("device:", next(engine.model.parameters()).device)

cfg = engine.model.inference_config
if cfg is not None:
    print("inference tier:", cfg.config_label)
    print(describe_inference_config(cfg))

with torch.inference_mode():
    preds = engine.run_from_adapter(request, steps=ROLLOUT_STEPS)

preds = [pred.to("cpu") for pred in preds]
print("predictions:", [str(p.metadata.time[0]) for p in preds])"""
                ).split("\n")
            ]
        if "request = downloader.ingest_request" in src and "InitialConditionBuilder" in src:
            cell["source"] = [
                line + "\n"
                for line in (
                    """from flash_aurora.engine import AuroraEngine, InitialConditionBuilder

engine = AuroraEngine.from_preset(
    PRESET,
    asset_root=ASSET_ROOT,
    checkpoint_path=CHECKPOINT_ARG,
    allow_hub_download=ALLOW_HUB_DOWNLOAD,
    hf_mirror=USE_HF_MIRROR,
    hf_endpoint=None if USE_HF_MIRROR else HF_ENDPOINT,
)

request = downloader.ingest_request(VALID_TIME, time_index=TIME_INDEX, download=False)

builder = InitialConditionBuilder(engine.config)
batch = builder.from_source(request)
print("IC time:", batch.metadata.time)
print("spatial:", batch.spatial_shape)"""
                ).split("\n")
            ]
        # Preserve committed cell outputs when refreshing setup paths.
    for cell in data["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "Prerequisites" in src and "ERA5 Pretrained" in src:
                cell["source"] = [
                    line + "\n"
                    for line in (
                        f"""# ERA5 Pretrained — flash-aurora Engine

Same data and forecast setup as the upstream [Microsoft Aurora example](microsoft-aurora/docs/example_era5.ipynb) (2023-01-01, 2-step rollout → 12:00 / 18:00 UTC), but uses the **flash-aurora Engine** `DataDownloader` for ERA5 fetch and inference.

{TUTORIAL_DISK_BLURB}
## Prerequisites

1. **CDS credentials** (any one): environment variable `CDSAPI_KEY`, `~/.cdsapirc`, or interactive `getpass` in the notebook below.
2. **Download dependencies**: `pip install cdsapi netcdf4` (or `uv pip install cdsapi netcdf4`).
3. **Checkpoint / data root:** defaults to `./assets`. Uncomment `ASSET_ROOT` in the setup cell to point at an absolute data-disk path.
4. **GPU** recommended for 0.25° global inference.
5. **Network**: CDS uses the official Copernicus API only; for Hugging Face, set `USE_HF_MIRROR = True` in the setup cell when `huggingface.co` is unreachable."""
                    ).split("\n")
                ]
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print("fixed", path)


def example_hres_t0() -> None:
    write_nb(
        "example_hres_t0.ipynb",
        [
            md(
                f"""# HRES T0 Finetuned — flash-aurora Engine

Same forecast setup as the upstream [Microsoft Aurora HRES T0 example](microsoft-aurora/docs/example_hres_t0.ipynb):

- **Date:** 2022-05-11
- **Initial condition:** 06:00 UTC (two-step history 00:00 + 06:00)
- **Rollout:** 2 steps → valid times 12:00 and 18:00 UTC
- **Model:** `aurora-0.25-finetuned` via preset `hres_t0_finetuned`

This notebook replaces manual `Batch` assembly and `aurora.rollout()` with **`DataDownloader`** (WeatherBench2 + ERA5 static) and **`AuroraEngine`**.

{TUTORIAL_DISK_BLURB}
## Prerequisites

1. **Extra packages** (optional download deps): `uv pip install gcsfs zarr` or `pip install gcsfs zarr`.
2. **CDS credentials** for ERA5 static fields (`geopotential`, `land_sea_mask`, `soil_type`): `CDSAPI_KEY`, `~/.cdsapirc`, or interactive `getpass` below.
3. **Checkpoint:** place `aurora-0.25-finetuned.ckpt` under `ASSET_ROOT`, or allow Hugging Face download.
4. **GPU** recommended for 0.25° global inference (~721×1440).
5. **Network:** WeatherBench2 reads from Google Cloud (`gs://weatherbench2/...`); CDS uses the official Copernicus API."""
            ),
            code(
                SETUP_PREAMBLE.format(
                    tutorial_asset_root=TUTORIAL_ASSET_ROOT,
                    preset="hres_t0_finetuned",
                    valid_time="2022, 5, 11, 6",
                    time_index=1,
                    rollout_steps=2,
                    inference_precision="bf16_mixed@fp32",
                )
            ),
            md(
                """## 1. Download HRES T0 + ERA5 static

`DataDownloader.ensure()` fetches:

| File | Source |
|------|--------|
| `{day}-surface-level.nc`, `{day}-atmospheric.nc` | WeatherBench2 HRES T0 zarr |
| `static.nc` | CDS ERA5 single-level (same as upstream notebook) |

Files land in `<ASSET_ROOT>/hres_t0/`. If everything is already cached, CDS/WB2 are **not** contacted again."""
            ),
            code(
                """import getpass
import os
from pathlib import Path

from flash_aurora.engine import DownloadResult
from flash_aurora.engine.core.redaction import safe_path

missing = downloader.missing(VALID_TIME)

if missing:
    CDS_API_KEY = os.environ.get("CDSAPI_KEY", "").strip() or None
    if CDS_API_KEY is None and not Path.home().joinpath(".cdsapirc").is_file():
        CDS_API_KEY = getpass.getpass("CDS API key (for ERA5 static only): ").strip() or None
    if not CDS_API_KEY and not Path.home().joinpath(".cdsapirc").is_file():
        raise ValueError(
            "No CDS credentials found. Static fields come from ERA5; set CDSAPI_KEY or ~/.cdsapirc."
        )
    print("Missing cache files:", missing)
    result = downloader.ensure(VALID_TIME, cds_api_key=CDS_API_KEY)
else:
    print("HRES T0 cache already complete under", safe_path(cache_dir))
    result = DownloadResult(
        cache_dir=cache_dir,
        paths=downloader.expected_paths(VALID_TIME),
        downloaded=(),
        skipped=tuple(downloader.expected_paths(VALID_TIME)),
    )

print("downloaded:", result.downloaded)
print("skipped:", result.skipped)
for key, path in result.paths.items():
    print(f"  {key}: {safe_path(path)}")"""
            ),
            md(
                """## 2. Build initial condition

`Wb2HresT0Adapter` (preset source `wb2_hres`) mirrors upstream `_prepare()`:

- Selects the **pair** of times ending at `TIME_INDEX` (default `1` → 00:00 + 06:00, IC at 06:00).
- **Flips latitude** so latitudes decrease (Aurora convention).
- Loads ERA5 static fields without flipping.

`InitialConditionBuilder.from_source()` returns a validated `Batch` — no manual tensor wiring."""
            ),
            md(
                """## 3. Load model and rollout

Set `INFERENCE_PRECISION` in the setup cell **before** `engine.load()`. The finetuned 0.25° model uses the same Swin backbone as pretrained; `bf16_mixed@fp32` is a good default on recent NVIDIA GPUs."""
            ),
            code(rollout_cell()),
            md(
                """## 4. Visualize: Aurora vs HRES T0 ground truth

Left: model 2 m temperature (K → °C). Right: HRES T0 `2m_temperature` at the matching valid time (with latitude flip applied to match Aurora's grid)."""
            ),
            code(
                """import matplotlib.pyplot as plt
import xarray as xr

surf_vars_ds = xr.open_dataset(result.paths["surface"], engine="netcdf4")

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction (flash-aurora Engine)")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ref = surf_vars_ds["2m_temperature"][2 + i].values[::-1, :]
    ax[i, 1].imshow(ref - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("HRES T0")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.show()"""
            ),
            md(
                """## 5. (Optional) Export rollout to NetCDF

Writes one NetCDF per step under `<ASSET_ROOT>/output/hres_t0_finetuned/`. Re-runs inference from the same `batch`; skip if you only need in-memory `preds`."""
            ),
            code(EXPORT_OPTIONAL),
        ],
    )


def main() -> None:
    example_hres_t0()
    example_hres_0_1()
    example_cams()
    example_wave()
    example_tc_tracking()
    fix_example_era5()


if __name__ == "__main__":
    main()
