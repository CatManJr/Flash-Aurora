# Flash-Aurora tutorials and API sketches

This page collects CLI and Python API sketches for running Flash-Aurora in process and via the ZeroMQ forecast scheduler. Interactive walkthroughs live in the notebooks under this directory (see [Tutorial notebooks](#tutorial-notebooks)). Conceptual architecture, presets, and precision-tier design remain in the [project README](../README.md). Full latency and drift tables are in [benchmarks.md](benchmarks.md).

## Quick start

Install from the repository root:

```bash
git clone <repository-url>
cd flash-aurora
uv sync
```

Dependencies are declared in `pyproject.toml` and pinned by `uv.lock`. CuTe DSL kernels JIT-compile for the local GPU architecture. Set `CUTE_DSL_ARCH` when needed, for example `sm_89` on RTX 4090 or `sm_120a` on Blackwell GeForce.

Minimal in-process forecast sketch:

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

For a fuller argument reference, see [Core API](#core-api) below. For Aurora 1.5, open [`example_aurora_v1p5.ipynb`](example_aurora_v1p5.ipynb).

## Forecast scheduler deployment

For how this ZMQ layout compares to vLLM's frontend ↔ EngineCore path, see
[design/vllm_zmq_communication.md](design/vllm_zmq_communication.md).

### Localhost loopback (ready + optional preload)

Bind a worker on `ipc://` under the asset root (or `tcp://127.0.0.1` with an
ephemeral port), preload weights, emit `ready`, then serve:

```bash
export AURORA_ASSET_ROOT=/path/to/data/aurora

uv run python -m flash_aurora.scheduler.localhost \
  --preset era5_pretrained \
  --asset-root "$AURORA_ASSET_ROOT" \
  --transport ipc \
  --inference-precision bf16_mixed@fp32
```

The process prints resolved `command_addr` / `event_addr` (including the real TCP
port when using `--transport tcp`). Clients should call
`ForecastClient.wait_for_ready()` before submitting jobs. Late joiners miss the
startup `ready` PUSH and instead observe `health` with `message="ready"` after
preload.

### Persistent worker

Start one worker in its own terminal:

```bash
export AURORA_ASSET_ROOT=/path/to/data/aurora

uv run python -m flash_aurora.scheduler \
  --preset era5_pretrained \
  --asset-root "$AURORA_ASSET_ROOT" \
  --worker-id gpu0-era5 \
  --device cuda:0 \
  --inference-precision bf16_mixed@fp32 \
  --command-addr tcp://127.0.0.1:9755 \
  --event-addr tcp://127.0.0.1:9756
```

Point `ForecastClientConfig` at the same addresses. Send `shutdown` through the client (or SIGINT to the worker process) when tearing down the service.

### Coordinator over more than one GPU

Start multiple workers with different sockets and devices, then start the coordinator:

```bash
export AURORA_ASSET_ROOT=/path/to/data/aurora

uv run python -m flash_aurora.scheduler \
  --preset era5_pretrained \
  --asset-root "$AURORA_ASSET_ROOT" \
  --worker-id gpu0-era5 \
  --device cuda:0 \
  --inference-precision bf16_mixed@fp32 \
  --command-addr tcp://127.0.0.1:9755 \
  --event-addr tcp://127.0.0.1:9756

uv run python -m flash_aurora.scheduler \
  --preset era5_pretrained \
  --asset-root "$AURORA_ASSET_ROOT" \
  --worker-id gpu1-era5 \
  --device cuda:1 \
  --inference-precision bf16_mixed@fp32 \
  --command-addr tcp://127.0.0.1:9765 \
  --event-addr tcp://127.0.0.1:9766

uv run python -m flash_aurora.scheduler.coordinator_main \
  --command-addr tcp://127.0.0.1:9855 \
  --event-addr tcp://127.0.0.1:9856 \
  --worker gpu0-era5,era5_pretrained,tcp://127.0.0.1:9755,tcp://127.0.0.1:9756,cuda:0,1 \
  --worker gpu1-era5,era5_pretrained,tcp://127.0.0.1:9765,tcp://127.0.0.1:9766,cuda:1,1
```

Clients connect to `tcp://127.0.0.1:9855` and `tcp://127.0.0.1:9856`. Two concurrent `era5_pretrained` jobs can then occupy the two workers at the same time. `ForecastRequest.sticky_key` is optional; when set, the coordinator keeps requests with the same key on the same worker when capacity permits.

For Aurora 1.5 workers (`aurora_v1p5` / `aurora_v1p5_ensemble`), `ForecastRequest` also accepts optional `fine_lead_times` (hourly substeps within each main AR step; last entry must match the model timestep), `use_noise_accumulation`, `ensemble_members`, and `noise_seed`. Multi-member runs call `reset_noise()` between members and tag `ForecastEvent.ensemble_member`. Fine-lead and ensemble knobs use the single-GPU rollout path; distributed workers still run standard timestep AR only. CUDA graphs remain unsupported for Aurora 1.5.

### Client sketch

```python
from flash_aurora.scheduler import ForecastClient, ForecastClientConfig, ForecastRequest

client = ForecastClient(
    ForecastClientConfig(
        command_addr="tcp://127.0.0.1:9755",
        event_addr="tcp://127.0.0.1:9756",
    )
)
for event in client.forecast(
    ForecastRequest(
        request_id="job-1",
        preset="era5_pretrained",
        steps=4,
        valid_time="2023-01-01T06:00:00",
        time_index=1,
        download=False,
        export_dir="/path/to/output",
    )
):
    print(event.kind, event.export_path or event.valid_time or "")

client.shutdown_worker()  # stop a worker this client owns
client.close()
```

For incremental progress UI, call `client.submit(request)` and iterate `client.events(request.request_id)` instead of the blocking `forecast()` helper.

## Tutorial notebooks

| Notebook | Topic |
| -------- | ----- |
| [`example_era5.ipynb`](example_era5.ipynb) | Baseline in-process `era5_pretrained`; populate cache and checkpoints. |
| [`example_aurora_v1p5.ipynb`](example_aurora_v1p5.ipynb) | Aurora 1.5: extended ERA5 IC, 6 h / hourly leads, optional ensemble, viz vs ERA5. |
| [`example_hres_t0.ipynb`](example_hres_t0.ipynb) | WeatherBench2 HRES T0 finetuned preset. |
| [`example_hres_0.1.ipynb`](example_hres_0.1.ipynb) | $0.1^{\circ}$ high-resolution Aurora (`hres_0.1`). |
| [`example_cams.ipynb`](example_cams.ipynb) | CAMS air-pollution preset. |
| [`example_wave.ipynb`](example_wave.ipynb) | Wave preset; manual MARS GRIB cache placement when needed. |
| [`example_tc_tracking.ipynb`](example_tc_tracking.ipynb) | Tropical-cyclone tracking LoRA preset. |
| [`example_roi_export.ipynb`](example_roi_export.ipynb) | Batched ROI mask export (NetCDF / GeoTIFF) via `RoiBatch`. |
| [`example_scheduler_single_worker.ipynb`](example_scheduler_single_worker.ipynb) | Single-GPU ZMQ queue; scoped preflight cleanup; graceful shutdown. |
| [`example_scheduler_distributed_workers.ipynb`](example_scheduler_distributed_workers.ipynb) | Heterogeneous four-GPU dispatch; cold-start timing; device status; `last_step_array` previews; refill while a slow job is pending. |
| [`example_scheduler.py`](example_scheduler.py) | Command-line version of the single-worker tutorial. |

All scheduler tutorials require an absolute `AURORA_ASSET_ROOT` with checkpoints and cached ingress. They start real subprocesses, set `download=False`, and call `shutdown_scheduler_subprocess(es)` at the end.

## Supervisor CLI

```bash
# one-shot cleanup (typical notebook usage)
python -m flash_aurora.scheduler.supervisor

# report without killing (dry run)
python -m flash_aurora.scheduler.supervisor --dry-run

# persistent daemon, scan every 30 s
python -m flash_aurora.scheduler.supervisor --daemon --interval 30
```

Unit tests live under [`../tests/scheduler/`](../tests/scheduler/). Integration tests require a GPU and cached assets (`pytest tests/scheduler/ -m "not integration and not gpu"` from the repository root for protocol and mock worker tests).

## Core API

Engine lifecycle.

```python
from datetime import datetime
from pathlib import Path

from flash_aurora import AuroraEngine, DataDownloader

# 1. Create engine (checkpoint loads on first prepare)
engine = AuroraEngine.from_preset(
    "era5_pretrained",  # preset: era5_pretrained, hres_0.1, cams, ...
    asset_root=Path("/path/to/assets"),  # local root for checkpoints and downloads
    checkpoint_path=None,  # explicit checkpoint file; preset default when None
    allow_hub_download=True,  # fetch checkpoint from Hugging Face when missing locally
    hf_endpoint=None,  # Hugging Face API base URL
    hf_mirror=False,  # True uses built-in HF mirror endpoint
    hf_revision=None,  # Hugging Face revision or tag
    hf_token=None,  # token for private or gated models
    export_dir=None,  # default NetCDF export directory for rollout_and_export
    inference_precision="bf16_mixed@fp32",  # tier label backbone@encoder_decoder
    overlap_ic_load=True,  # default: IC build overlaps model load on first prepare
    async_export=False,  # async NetCDF writes in rollout_and_export
    export_pool_size=2,  # thread pool size when async_export is True
    export_max_inflight=None,  # max queued async exports; default pool_size - 1
    export_use_egress_stream=True,  # dedicated CUDA stream for async export D2H
    ic_cache=False,  # disk cache for repeated same-day IC builds
    forward_warmup_iters=2,  # untimed forwards after prepare; 0 disables
    distributed=None,  # DistributedConfig for pipeline parallelism; None is single-GPU
    presets=None,  # optional custom PresetRegistry
)
# EngineConfig-only (set after from_preset):
engine.config.cuda_graph = False  # experimental on legacy; unsupported for Aurora 1.5 (variable lead_times / ensemble noise)
engine.config.device = "cuda:0"  # inference device
engine.config.gpu_guard = True  # queue large jobs when VRAM is saturated
engine.config.gpu_guard_timeout = 3600.0  # seconds to wait for GPU slot
engine.config.gpu_rollout_steps = 1  # rollout depth used for VRAM estimate

# 2. Downloader (ensure runs only when cache files are missing)
downloader = DataDownloader.from_preset(
    "era5_pretrained",  # must match engine preset
    asset_root=engine.config.asset_root,
    user_cwd=None,  # base for relative paths; current working directory when None
    presets=None,  # optional custom PresetRegistry
    credentials=None,  # optional DownloadCredentials bundle
    cds_api_key=None,  # CDS API key for ERA5
    cds_api_url=None,  # CDS API URL
    ads_api_key=None,  # ADS API key for CAMS
    ads_api_url=None,  # ADS API URL
    ecmwf_api_key=None,  # ECMWF Open Data key for HRES
    ecmwf_api_url=None,  # ECMWF Open Data URL
    ecmwf_email=None,  # ECMWF account email for HRES
    workers=None,  # parallel download workers; preset default when None
)

# 3. Build ingest request (download=True calls ensure when cache is missing)
request = downloader.ingest_request(
    datetime(2023, 1, 1, 6),  # any valid UTC time for the initial condition
    cache_dir=None,  # override cache directory; preset layout when None
    time_index=1,  # 0..3 for 00/06/12/18 UTC cycle
    download=True,  # False skips auto-download even if cache is missing
    credentials=None,  # optional per-call credential override
    cds_api_key=None,
    cds_api_url=None,
    ads_api_key=None,
    ads_api_url=None,
    ecmwf_api_key=None,
    ecmwf_api_url=None,
    ecmwf_email=None,
    prompt=False,  # prompt for credentials when missing
    workers=None,  # parallel download workers for this ensure call
)

# 4. Prepare IC batch on CPU, load model, run forward warmup
batch = engine.prepare(
    request,
    rollout_steps=4,  # rollout depth for GPU guard VRAM estimate
    overlap=None,  # None uses engine.config.overlap_ic_load; True or False to override
)

# 5. Autoregressive rollout
forecasts = list(
    engine.rollout_stream(
        batch,
        steps=4,  # number of 6-hour lead steps
        observers=None,  # optional RolloutObserver hooks per step
    )
)

# 6. Optional NetCDF export (export while autoregressing)
# paths = list(
#     engine.rollout_and_export(
#         batch,
#         steps=4,
#         export_dir=None,  # None uses engine.config.export_dir
#         async_export=None,  # None uses engine.config.async_export
#     )
# )

# 7. Release GPU reservation between jobs; optionally move weights to CPU
engine.release_gpu(move_model_to_cpu=True)

# 8. Terminal teardown (for example at process exit)
engine.close()  # releases GpuGuard lease without CPU migration; clears model state
```

Download and ingest.

```python
from datetime import datetime
from flash_aurora import AuroraEngine, DataDownloader
from flash_aurora.engine import InitialConditionBuilder

engine = AuroraEngine.from_preset("era5_pretrained", asset_root="/path/to/assets")
dl = DataDownloader.from_preset("era5_pretrained", asset_root="/path/to/assets")
dl.ensure(valid_time=datetime(2023, 1, 1, 6))

request = dl.ingest_request(datetime(2023, 1, 1, 6), time_index=1, download=False)
batch = InitialConditionBuilder(engine.config).from_source(request)
forecasts = list(engine.rollout_stream(batch, steps=4))
paths = list(engine.rollout_and_export(batch, steps=4))
```

## Lifecycle optimizations

Cold-start time is usually dominated by CPU ingress (`build_ic`), model initialization (`build_model` / CuTe JIT), and NetCDF export, not GPU forward. Two optional optimizations address that; both are configured on `EngineConfig` / `from_preset()` and can be overridden per call.

| Field                      | Default | What it does                                                                                                                                         |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `overlap_ic_load`          | `True`  | `prepare()` / `prepare_from_netcdf()` runs IC build on a background thread while the model is built and checkpoint-loaded.                           |
| `async_export`             | `False` | `rollout_and_export()` pipelines egress-stream GPU to CPU copy and background NetCDF writes (Earth-2 `AsyncZarrBackend` pattern, one file per step). |
| `export_pool_size`         | `2`     | Thread-pool size for async NetCDF writes.                                                                                                            |
| `export_max_inflight`      | `None`  | Max queued writes before back-pressure (`pool_size - 1` when unset).                                                                                 |
| `export_use_egress_stream` | `True`  | Use a dedicated CUDA stream for D2H during async export.                                                                                             |
| `ic_cache`                 | `False` | Cache the post-processed `Batch` on disk so repeated prepares with the same inputs replay bit-for-bit (see below).                                   |
| `forward_warmup_iters`     | `2`     | Untimed forward passes after `prepare()` / before first rollout step to compile CuTe kernels (set `0` to disable).                                   |

**Per-call overrides.** `engine.prepare(request, overlap=False)` and `engine.rollout_and_export(batch, steps=K, async_export=True)` take precedence over the config for that invocation only.

## IC cache (`ic_cache`)

Optional disk cache for repeat runs on the same initial field. Applies to all presets via `InitialConditionBuilder`. Keys for `from_source(request)` include preset, calendar day, `time_index`, and input file hashes (CDS/GRIB/NetCDF cache layout or `raw_paths`). Keys for `from_netcdf_path(path)` and `prepare_from_netcdf(path)` include preset and user NetCDF content hash.

Cache files live under `{cache_dir}/.ic-cache/` (download workflow) or beside the user NetCDF. Enable when benchmarking or serving the same analysis day repeatedly:

```python
engine = AuroraEngine.from_preset(
    "hres_0.1",
    asset_root="/path/to/assets",
    ic_cache=True,
)
batch = engine.prepare(request)  # cold once, then fast on cache hit
```

Default is `False` so arbitrary one-off user NetCDF paths do not write multi-GB cache files unless opted in.

## Which presets benefit

Measured on RTX PRO 6000 (`bf16_mixed@fp32`, forward warmup $2$). Forward latency is ${\sim}680$ ms per step on $0.25^{\circ}$ and $0.1^{\circ}$ presets regardless of these flags.

| Preset                                                | `overlap_ic_load` | `async_export`                   | Rationale                                                                                                                                  |
| ----------------------------------------------------- | ----------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `hres_0.1`                                            | **On** (default)  | **On** for `K \gtrsim 4`         | IC build (GRIB/regrid) dominates prepare (about 2 min); each export step is about 5 s vs about 0.7 s forward. Largest win from both flags. |
| `era5_pretrained`, `hres_t0_finetuned`, `tc_tracking` | On (default)      | Optional                         | Modest prepare overlap (about 4-8 s saved). Export is smaller per step; async helps mainly on longer rollouts.                             |
| `cams`                                                | On (default)      | Optional for long rollouts       | Medium grid; export cost grows with step count.                                                                                            |
| `wave`                                                | On (default)      | Optional if exporting many steps | Ingress can be heavy when GRIB cache is cold.                                                                                              |
| `small_pretrained`                                    | Off               | Off                              | Fast IC and tiny NetCDF; overhead of threading not worth it.                                                                               |

Turn overlap off when debugging ingress, profiling serial prepare stages, or when the initial condition is already a pre-built NetCDF and `load()` alone is sufficient. Turn async export off for single-step smoke tests, very short rollouts ($K=1$ or $K=2$), or when debugging export correctness.

### Recommended service path ($0.1^{\circ}$ example)

```python
from datetime import datetime

from flash_aurora import AuroraEngine, DataDownloader

engine = AuroraEngine.from_preset(
    "hres_0.1",
    asset_root="/path/to/assets",
    inference_precision="bf16_mixed@fp32",
    overlap_ic_load=True,
    async_export=True,
    export_pool_size=2,
)
dl = DataDownloader.from_preset("hres_0.1", asset_root=engine.config.asset_root)
request = dl.ingest_request(
    datetime(2022, 5, 11, 6),
    time_index=1,
    download=True,
)

batch = engine.prepare(request, rollout_steps=4)
paths = list(engine.rollout_and_export(batch, steps=4))
engine.release_gpu()
```

### Debug / baseline settings

```python
engine = AuroraEngine.from_preset(
    "era5_pretrained",
    asset_root="/path/to/assets",
    overlap_ic_load=False,
    async_export=False,
)
engine.load()
batch = InitialConditionBuilder(engine.config).from_source(request)
paths = list(engine.rollout_and_export(batch, steps=4))
```

**Configuration arguments.** Key fields on `EngineConfig`: `variant`, `source`, `asset_root`, `checkpoint_path`, `inference_precision`, `cuda_graph`, `device`, `export_dir`, `allow_hub_download`, `gpu_guard`, `gpu_rollout_steps`, `overlap_ic_load`, `async_export`, `export_pool_size`, `export_max_inflight`, `export_use_egress_stream`, `ic_cache`, `forward_warmup_iters`. Inspect registered names with `DEFAULT_PRESETS.names()`.

**Utilities.** `ecmwf_credential_status()` reports ECMWF API readiness before MARS requests; `normalize_user_path()` and `AssetStore` constrain file access to allowed roots under `asset_root`.

