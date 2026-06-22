# Flash-Aurora: One Small Step Toward Efficient Geospatial Foundation Models Inference Service

High-performance inference stack for the [Microsoft Aurora](https://github.com/microsoft/aurora) Earth-system foundation model. The repository packages the upstream model under `flash_aurora.aurora`, custom CUDA kernels (Triton and CuTeDSL), precision routing, and `flash_aurora.engine` for data ingress, checkpoints, rollout, and export.

## Install

```bash
git clone <repository-url>
cd flash-aurora
uv sync
```

Dependencies include PyTorch, `nvidia-cutlass-dsl[cu13]`, and `quack-kernels` (see `pyproject.toml`). CuTe DSL JIT-compiles for the local GPU by default. Other dependencies are the same as [Microsoft Aurora](https://github.com/microsoft/aurora).

## Repository layout

| Path | Role |
|------|------|
| `flash_aurora/aurora/` | Aurora model fork (upstream README preserved in place). See `NOTICE.md` and `LICENSE.txt`. |
| `flash_aurora/engine/` | `AuroraEngine`, presets, ingress/download, rollout, export, GPU guard. |
| `flash_aurora/aurora/ops/triton/` | Fused Swin layout and AdaLN kernels. |
| `flash_aurora/aurora/ops/cute/` | CuTeDSL window self-attention kernels. |
| `benchmark/` | Kernel- and model-level timing scripts. |
| `tests/` | `tests/aurora`, `tests/kernels`, `tests/engine`. |

Run tests: `./scripts/run_tests.sh`

## Inference precision tiers

Presets are labeled `backbone@encoder_decoder`, for example `bf16_mixed@fp32`.

**Left token (backbone):** matmul and window-attention mode for Swin3D.

| Token | Backbone |
|-------|----------|
| `fp32` | Strict FP32 matmul; PyTorch SDPA for attention unless a higher tier replaces it. |
| `tf32` | TF32 tensor-core matmul; CuTe window attention (FP32 I/O). |
| `bf16_mixed` | Hybrid BF16 on attention QKV/proj and MLP; TF32 elsewhere; CuTe window attention (BF16). |
| `bf16` | Full backbone BF16 matmul with fused CuTe attention chain. |

**Right token (encoder/decoder):** Perceiver matmul precision, `fp32` (strict) or `tf32` (tensor cores).

Set tiers on the model (`inference_precision=...`) or on `EngineConfig` when using `AuroraEngine`.

### Triton fusion on every custom tier

All custom `inference_precision` tiers (`fp32@*`, `tf32@*`, `bf16_mixed@*`, `bf16@*`) enable the same Triton fusion base:

- **Fused window layout** (`use_triton_layout`): roll, pad, partition, and reverse layout in fused kernels instead of many small eager allocations.
- **Fused AdaLN and residual** (`use_triton_adaln`): LayerNorm and FiLM modulation fused with the residual add on the Swin hot path.

These kernels are active regardless of whether the backbone runs FP32, TF32, or BF16 matmul. They lower peak activation memory and improve bandwidth efficiency relative to the decomposed PyTorch Swin path. CuTe window attention and backbone matmul precision are layered on top of this Triton base for `tf32` and BF16 tiers. The pure PyTorch reference in benchmarks (`pytorch_backbone_fp32_encoder_decoder_fp32`) deliberately disables Triton and CuTe for accuracy baselining only.

Optional `InferenceWorkspacePool` reuses a scratch buffer for the backbone decoder concat to avoid repeated large allocations on fixed-shape inference.

## Window attention kernel performance

Measured with `benchmark/bench_window_attn.py` on an **NVIDIA RTX PRO 6000 Blackwell Server Edition** (trimmed mean of 200 runs per shape).

**0.25-degree ERA5 encoder stages** (unmasked, N=144 per window):

| Stage | Bwin | Heads | BF16 CuTe (ms) | BF16 SDPA (ms) | Speedup |
|-------|------|-------|----------------|----------------|---------|
| 1 | 1800 | 8 | 0.727 | 0.785 | 1.08x |
| 2 | 450 | 16 | 0.373 | 0.406 | 1.09x |
| 3 | 128 | 32 | 0.221 | 0.241 | 1.09x |

| Stage | Bwin | Heads | TF32 CuTe (ms) | FP32 SDPA (ms) | Speedup |
|-------|------|-------|----------------|----------------|---------|
| 1 | 1800 | 8 | 1.612 | 2.612 | 1.62x |
| 2 | 450 | 16 | 0.817 | 1.315 | 1.61x |
| 3 | 128 | 32 | 0.473 | 0.765 | 1.62x |

**Shifted-window mask** (Swin bias -100):

| Mode | Stage-1 (Bwin=1800, H=8) | Speedup vs SDPA |
|------|--------------------------|-----------------|
| BF16 CuTe | 0.829 ms vs 1.031 ms | 1.24x |
| TF32 CuTe | 1.936 ms vs 3.042 ms | 1.57x |

Production inference uses N=144 windows on the default 0.25-degree grid. BF16 CuTe attention is not supported for N below 32; use `tf32` or PyTorch SDPA on downsampled stages with very small windows.

## End-to-end forward performance

Measured with `benchmark/bench_aurora_pretrained.py` on **AuroraPretrained** at **721 x 1440**, batch size 1, ERA5 initial conditions (2023-01-01 06:00 UTC). All custom tiers below include the Triton fusion base described above.

| Tier | Forward (ms) | Speedup vs PyTorch FP32 ref | Mean abs error vs ref | Cosine sim vs ref |
|------|--------------|----------------------------|------------------------|-------------------|
| `bf16_mixed@fp32` | 681.9 | 3.15x | 0.115 | 1.000 |
| `bf16_mixed@tf32` | 681.4 | 3.16x | 0.115 | 1.000 |
| `bf16@fp32` | 680.5 | 3.16x | 0.191 | 1.000 |
| `tf32@tf32` | 931.0 | 2.31x | 0.060 | 1.000 |
| `tf32@fp32` | 1093.3 | 1.97x | 0.018 | 1.000 |
| `fp32@fp32` | 1988.9 | 1.08x | 5.4e-05 | 1.000 |
| PyTorch autocast (backbone) | 1013.7 | 2.12x | 0.140 | 1.000 |
| PyTorch FP32 ref | 2151.0 | base | 0 | 1.000 |

The PyTorch FP32 reference uses no custom kernels. Every other custom tier uses Triton layout and AdaLN fusion; `tf32` and BF16 tiers additionally use CuTe window attention and the corresponding backbone matmul mode. Cosine similarity is computed over the flattened output tensor (all surface and atmospheric variables) relative to the PyTorch FP32 reference. All custom tiers pass per-variable mean relative-error tolerances from the upstream golden tests on this ERA5 sample. Recommended production preset on this hardware: `bf16_mixed@fp32` or `bf16_mixed@tf32` (about 3x speedup with bounded drift).

## Engine (`flash_aurora.engine`)

`flash_aurora.engine` is the inference service layer. It binds Aurora variants, upstream data profiles, checkpoint resolution, batch validation, multi-step rollout, and NetCDF export behind a preset-driven API. Tutorial notebooks under `docs/example_*.ipynb` exercise each preset end to end.

### Architecture

The engine is organized in four layers. Data flows from download and adapters into a validated `Batch`, through the loaded model, and optionally to disk as forecast NetCDF.

| Layer | Path | Role |
|-------|------|------|
| Core | `engine/core/` | `EngineConfig`, `PresetRegistry`, `AuroraEngine`, checkpoint load, `RolloutSession`. |
| Ingress | `engine/ingress/` | `DataDownloader`, source adapters, `InitialConditionBuilder`, `BatchValidator`, static fields. |
| Egress | `engine/egress/` | `RolloutExporter`, CPU offload, step-wise NetCDF naming. |
| Runtime | `engine/runtime/` | CUDA Graph warmup (`GraphPool`), cross-process `GpuGuard`, VRAM budget estimates. |

A **preset** pairs a `ModelVariantSpec` (checkpoint, variable lists, grid shape $(H, W)$, timestep $\Delta t$) with a `SourceProfile` (schema, latitude convention, cache layout). `DataDownloader.ensure()` fills the preset cache. `InitialConditionBuilder` reads cached files or adapter requests and attaches Hugging Face static fields. `BatchValidator` checks tensor shapes and variable names against the variant. `AuroraEngine.load()` resolves checkpoints, applies `inference_precision`, and optionally acquires a `GpuGuard` lease from estimated VRAM. `predict()` runs one forward step; `rollout_stream()` chains $K$ steps with model-internal history, advancing valid time by $\Delta t$ per step. `rollout_and_export()` writes CPU-side NetCDF under `export_dir`.

### Presets and data sources

| Preset | Model | Grid $(H \times W)$ | Source | Download backend |
|--------|-------|---------------------|--------|------------------|
| `era5_pretrained` | AuroraPretrained | $721 \times 1440$ | CDS ERA5 | CDS |
| `hres_t0_finetuned` | Aurora (LoRA) | $721 \times 1440$ | WeatherBench2 HRES | WB2 + ERA5 static |
| `small_pretrained` | AuroraSmallPretrained | $400 \times 800$ | CDS ERA5 | CDS |
| `hres_0.1` | AuroraHighRes | $1801 \times 3600$ | IFS GRIB analysis | ECMWF Open Data / GRIB |
| `cams` | AuroraAirPollution | $451 \times 900$ | CAMS reanalysis | ADS |
| `wave` | AuroraWave | $721 \times 1440$ | WB2 met + MARS wave GRIB | WB2 + MARS |
| `tc_tracking` | Aurora (LoRA) | $721 \times 1440$ | WeatherBench2 HRES | WB2 + ERA5 static |

Personal ECMWF accounts typically lack MARS archive access. For `wave`, stage `{day}-wave.grib` under the cache manually or use an institutional MARS credential; see `docs/example_wave.ipynb`.

### Capabilities

- **Checkpoint and static assets.** Local `asset_root` with optional Hugging Face Hub download (`allow_hub_download`, mirror via `HF_MIRROR_ENDPOINT`).
- **Precision wiring.** `EngineConfig.inference_precision` selects the Triton fusion base and, when set, TF32/BF16 matmul and CuTe window attention (see above).
- **Automated ingress.** CDS (ERA5), ADS (CAMS), WeatherBench2 (HRES met), ECMWF Open Data (0.1-degree GRIB), and MARS (wave GRIB when permitted). Credentials merge from environment variables, `~/.cdsapirc`, `~/.ecmwfapirc`, and optional constructor kwargs.
- **Multi-step rollout.** `rollout_stream(batch, K)` and `run_from_netcdf(..., steps=K)`; optional `RolloutObserver` hooks per step.
- **NetCDF export.** `rollout_and_export()` writes forecast steps to `export_dir`.
- **CUDA Graph warmup.** `warmup()` captures fixed-shape backbone graphs when `cuda_graph=True`.
- **GPU scheduling.** `GpuGuard` (default on) estimates VRAM from variant, precision tier, and rollout depth; large jobs queue when memory is saturated. Disable with `gpu_guard=False` or `FLASH_AURORA_GPU_GUARD=0`.

### Core API

**Engine lifecycle.**

```python
from flash_aurora import AuroraEngine

engine = AuroraEngine.from_preset(
    "era5_pretrained",
    asset_root="/path/to/assets",
    inference_precision="bf16_mixed@fp32",
)
engine.load()
engine.warmup()
pred = engine.run_from_netcdf("/path/to/era5.nc", steps=1)[0]
engine.release_gpu()
```

**Download and ingest.**

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

**Configuration surface.** Key fields on `EngineConfig`: `variant`, `source`, `asset_root`, `checkpoint_path`, `inference_precision`, `cuda_graph`, `device`, `export_dir`, `allow_hub_download`, `gpu_guard`, `gpu_rollout_steps`. Inspect registered names with `DEFAULT_PRESETS.names()`.

**Utilities.** `ecmwf_credential_status()` reports ECMWF API readiness before MARS requests; `normalize_user_path()` and `AssetStore` constrain file access to allowed roots under `asset_root`.

## Testing notes

`test_aurora_small` compares FP64 forward outputs to Microsoft Hugging Face reference pickles. On recent PyTorch builds (for example 2.12.x), a small drift on a few surface variables can appear even with the official `microsoft-aurora` wheel. The test passes and emits a `UserWarning` when drift exceeds upstream tolerances. Use the [vanilla microsoft-aurora library](https://microsoft.github.io/aurora) to verify on your stack.

## License

This repository is licensed under the [MIT License](LICENSE).

Third-party components bundled in the library:

- `flash_aurora.aurora` is derived from [Microsoft Aurora](https://github.com/microsoft/aurora) (MIT). See [`flash_aurora/aurora/LICENSE.txt`](flash_aurora/aurora/LICENSE.txt) and [`flash_aurora/aurora/NOTICE.md`](flash_aurora/aurora/NOTICE.md).
- Some source files include additional notices (for example NVIDIA BSD-3-Clause in `flash_aurora/aurora/ops/cute/_dense_gemm_sm120.py`). See per-file headers.

## Reference

**Aurora model.** Bodnar et al., *A Foundation Model for the Earth System*, Nature (2025). [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y). Upstream documentation: [microsoft.github.io/aurora](https://microsoft.github.io/aurora).

**CUTLASS / CuTe DSL.** CuTe window-attention and dense GEMM kernels under `flash_aurora/aurora/ops/cute/` adapt layout, TMA, and GEMM patterns from [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) CuTe DSL examples (BSD-3-Clause; see file headers such as `ops/cute/_dense_gemm_sm120.py`). Runtime dependency: `nvidia-cutlass-dsl`.

**Flash Attention.** FMHA mainloop, online softmax, and dispatch structure follow [flash-attn](https://github.com/Dao-AILab/flash-attention) (`flash_attn/cute/`; Tri Dao).
