# Flash-Aurora

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21860697.svg)](https://doi.org/10.5281/zenodo.21860697)

Inference and serving for [Microsoft Aurora](https://github.com/microsoft/aurora) and other geospatial foundation models. A request takes an analysis cube in and writes NetCDF or GeoTIFF out. The tensors keep a lat/lon layout, so the engine uses shape-specialized kernels, named mixed precision (`bf16_mixed@fp32` by default), and a job-level GPU scheduler rather than an LLM serving loop.

[Walkthrough slides](https://catmanjr.github.io/Flash-Aurora-walkthrough/) · [Tutorial](docs/tutorial.md) · [Benchmark tables](docs/benchmarks.md)

## Install

```bash
pip install flash-aurora
```

From source:

```bash
git clone https://github.com/CatManJr/Flash-Aurora.git
cd Flash-Aurora
uv sync
```

Set `CUTE_DSL_ARCH` when the CuTe kernels need an explicit GPU architecture (`sm_89` on RTX 4090, `sm_120a` on Blackwell). If PyPI is slow, point `uv` at a mirror with `UV_DEFAULT_INDEX` or a local `uv.toml`; keep the committed `uv.lock` on the official index.

## Quick start

```python
from datetime import datetime
from pathlib import Path

from flash_aurora import AuroraEngine, DataDownloader

engine = AuroraEngine.from_preset(
    "era5_pretrained",
    asset_root=Path("/path/to/assets"),
    inference_precision="bf16_mixed@fp32",
)
downloader = DataDownloader.from_preset(
    "era5_pretrained",
    asset_root=engine.config.asset_root,
)
request = downloader.ingest_request(
    datetime(2023, 1, 1, 6),
    time_index=1,
    download=True,
)
batch = engine.prepare(request, rollout_steps=4)
forecasts = list(engine.rollout_stream(batch, steps=4))
engine.release_gpu(move_model_to_cpu=True)
engine.close()
```

Scheduler loopback, two-GPU placement, and the notebook index: [docs/tutorial.md](docs/tutorial.md).

## What it does

Production mixed precision (`bf16_mixed@fp32`) is faster than unfused FP32 and stays closer to that twin than framework autocast. One forward step on RTX PRO 6000 Blackwell is about $570$--$680\,\mathrm{ms}$ on the $0.25^{\circ}$ weather presets, versus about $1.7$--$2.1\,\mathrm{s}$ for unfused FP32. Each bar is a separate process (`--isolate-tiers`).

<p align="center">
  <img src="https://raw.githubusercontent.com/CatManJr/Flash-Aurora/master/docs/image/e2e_latency_by_tier_all_presets.svg" alt="One-step end-to-end forward latency by precision tier" width="95%"/>
</p>

Recommended tiers stay within per-variable tolerances versus the unfused FP32 reference (seed 42). `bf16@*` is not a production path.

<p align="center">
  <img src="https://raw.githubusercontent.com/CatManJr/Flash-Aurora/master/docs/image/precision_mean_rel_stacked_by_model.svg" alt="Stacked mean relative error by precision tier and preset" width="95%"/>
</p>

Window-attention kernels are short-window CuTe DSL (`N = 144` on the default $0.25^{\circ}$ encoder), not a generic LLM attention stack.

<p align="center">
  <img src="https://raw.githubusercontent.com/CatManJr/Flash-Aurora/master/docs/image/window_attn_cute_vs_sdpa_blackwell.svg" alt="CuTe DSL window attention versus PyTorch SDPA on Blackwell" width="95%"/>
</p>

Serving is one GPU per job. A ZeroMQ coordinator fills idle workers; it does not batch tokens inside one forward. [Scheduler notebooks](docs/example_scheduler_distributed_workers.ipynb).

<table width="100%">
  <tr>
    <th width="50%" align="center">One job per worker</th>
    <th width="50%" align="center">Refill while <code>hres_0.1</code> is pending</th>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="https://raw.githubusercontent.com/CatManJr/Flash-Aurora/master/docs/image/4_workers.png" width="100%"/></td>
    <td width="50%" valign="top"><img src="https://raw.githubusercontent.com/CatManJr/Flash-Aurora/master/docs/image/4_workers_refill.png" width="100%"/></td>
  </tr>
</table>

A preset that does not fit one GPU can run encoder / backbone / decoder on two devices in the same process (`DistributedConfig`). [ROI export](docs/example_roi_export.ipynb) clips on the egress path so a region of interest does not require a global dump.

## Presets

| Preset | Grid | Source |
| ------ | ---- | ------ |
| `era5_pretrained` | $721 \times 1440$ | CDS ERA5 |
| `aurora_v1p5` / `aurora_v1p5_ensemble` | $721 \times 1440$ | CDS ERA5 (extended) |
| `small_pretrained` | $400 \times 800$ | CDS ERA5 |
| `hres_t0_finetuned` / `tc_tracking` | $721 \times 1440$ | WeatherBench2 HRES |
| `hres_0.1` | $1801 \times 3600$ | IFS analysis |
| `cams` | $451 \times 900$ | CAMS |
| `wave` | $721 \times 1440$ | WB2 + MARS |

`wave` usually needs a hand-placed MARS cache; see [example_wave.ipynb](docs/example_wave.ipynb). Notebooks for every preset are listed in the [tutorial](docs/tutorial.md#tutorial-notebooks).

## License

[MIT](LICENSE). Aurora code is derived from [Microsoft Aurora](https://github.com/microsoft/aurora) (MIT), frozen at **v1.8.0**, with Aurora 1.5 from tag `v2.0.1`. Kernel files under `flash_aurora.models.ops` follow their per-file headers (including NVIDIA BSD-3-Clause where noted).

Bodnar et al., *A Foundation Model for the Earth System*, Nature (2025). [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y). Upstream docs: [microsoft.github.io/aurora](https://microsoft.github.io/aurora).
