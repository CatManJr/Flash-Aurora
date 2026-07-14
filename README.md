# Flash-Aurora: Toward Efficient Inference for Geospatial Foundation Models

Flash-Aurora is an inference stack for the [Microsoft Aurora](https://github.com/microsoft/aurora) Earth-system foundation model. It provides Triton and CuTe DSL kernels, mixed-precision routing, data ingress, checkpoint loading, autoregressive rollout, NetCDF export, and a ZeroMQ scheduler for out-of-process serving.

Companion documents:

- [docs/tutorial.md](docs/tutorial.md): install sketches, Engine and scheduler API examples, notebook index.
- [docs/benchmarks.md](docs/benchmarks.md): full latency, precision-drift, window-attention, and distributed-rollout tables.

## Highlights

### Extensible inference stack: Aurora 1.5 on day one

New model families plug into the same Engine without rewriting the hot path. Aurora 1.5 (Microsoft Aurora tag `v2.0.0`) is the first proof point: a side-path package beside the frozen legacy family (upstream **v1.8.0**), registered through the same preset, registry, and adapter surface.

- **Fixed composition root.** Presets `aurora_v1p5` and `aurora_v1p5_ensemble`, `ModelFactory`, ingress `cds_era5_v1p5`, and `RolloutSession` dispatch connect download, validate, load, rollout, and export. Clients continue to call `AuroraEngine.from_preset(...)`.
- **Shared acceleration.** `Batch`, `models/ops` (Triton / CuTe), and `inference_precision` accelerate the 1.5 Swin backbone. Variable `fine_lead_times`, prescribed insolation, and ensemble `reset_noise()` loops stay in the side path.
- **Day-one surface.** Deterministic and ensemble checkpoints, extended ERA5 initial conditions, hourly fine leads, and `bf16_mixed@fp32` end-to-end latency on par with `era5_pretrained` (about $3.1\times$ versus PyTorch FP32 on a $721 \times 1440$ grid). See [docs/example_aurora_v1p5.ipynb](docs/example_aurora_v1p5.ipynb).

The same pattern (model package, preset, adapter; reuse Engine and kernels) applies to later Aurora generations or other geospatial foundation models with a compatible contract.

### Triton fusion for lower backbone memory traffic

PyTorch Swin3D materializes many short-lived tensors for window layout and AdaLN boundaries. Flash-Aurora fuses these on the backbone hot path:

- **Fused window layout** (`triton_swin3d_layout.py`): cyclic shift, pad, 3D partition, and inverse merge in fused kernels; optional `InferenceWorkspacePool` reuse for fixed shapes.
- **Fused AdaLN and residual** (`triton_adaln.py`): LayerNorm, FiLM, and residual add without a full-width AdaLN intermediate.

On `bf16_mixed@*` and `tf32@*` tiers, AdaLN can keep FP32 activations between Swin blocks (`output_fp32`), reducing global-memory traffic while preserving inter-block precision. Details: [Precision tiers](#precision-tiers).

### CuTe DSL window attention

Aurora Swin windows are short ($N = 144$ for window size $(2, 6, 12)$). The CuTe kernels load the full window $K$ and $V$ into shared memory in a single stage ($tile_n \ge N$), run QK and PV MMAs with FP32 online softmax, and never materialize an $N \times N$ attention matrix in global memory. BF16 and TF32-simulated FP32 variants target these fixed shapes. On Blackwell (`sm_120`) microbenchmarks report about $1.07$--$1.22\times$ versus BF16 SDPA and about $1.59$--$1.69\times$ versus FP32 SDPA; on RTX 4090 (`sm_89`) about $1.04$--$1.05\times$ versus the fastest SDPA backend and about $2.2\times$ versus FP32 SDPA. Full tables: [docs/benchmarks.md](docs/benchmarks.md#window-attention-microbenchmarks).

### Mixed-precision inference

Tiers use the label `backbone@encoder_decoder` (default production tier `bf16_mixed@fp32`):

- **`bf16_mixed` backbone:** BF16 CuTe attention and BF16 MLP; TF32 Tensor Core GEMM elsewhere; FP32 activations between blocks via Triton fusion.
- **`tf32` backbone:** TF32 GEMM throughout Swin plus CuTe attention with FP32 I/O.
- **`fp32` backbone:** Strict FP32 GEMM and PyTorch SDPA; accuracy baseline with Triton fusion still enabled.

Encoder and Perceiver decoder default to `@fp32` because errors map directly into output fields, while the Swin backbone dominates runtime (about $63\%$ on `era5_pretrained`). Headline end-to-end latency for `bf16_mixed@fp32` is about $3.2\times$ versus PyTorch FP32 on `era5_pretrained` (about $680\,\mathrm{ms}$/step), about $3.1\times$ on `aurora_v1p5` (about $701\,\mathrm{ms}$/step), and about $3\times$ on finetuned presets with merged LoRA. Full tables: [docs/benchmarks.md](docs/benchmarks.md).

### Asynchronous multi-GPU serving

One ZeroMQ worker per GPU binds a preset. A coordinator routes jobs to idle workers and streams events to clients. This is job-level scheduling across heterogeneous models, not tensor parallelism inside one rollout. CLI and client sketches: [docs/tutorial.md](docs/tutorial.md#forecast-scheduler-deployment).

<table width="100%">
  <tr>
    <th width="50%" align="center">Single 1-step task distribution</th>
    <th width="50%" align="center">Refill while <code>hres_0.1</code> is pending</th>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/image/4_workers.png" width="100%"/></td>
    <td width="50%" valign="top"><img src="docs/image/4_workers_refill.png" width="100%"/></td>
  </tr>
</table>

Left: one job per worker and preset. Right: faster workers take follow-up jobs while `hres_0.1` remains pending. Traces from `docs/example_scheduler_distributed_workers.ipynb` on $4\times$ RTX PRO 6000 Blackwell.

### Pipeline parallelism for GPUs under $40\,\mathrm{GiB}$

`DistributedConfig` splits encoder, backbone, and spatial decoder across two GPUs in one process (not `torchrun`). Spatial decoder split cuts peak decoder VRAM from about $28\,\mathrm{GiB}$ to about $14\,\mathrm{GiB}$ per card on ERA5. On $2\times$ RTX 5090, a $4$-step run reaches about $1.3\,\mathrm{s}$/step on `era5_pretrained` and about $3.6\,\mathrm{s}$/step on `hres_0.1` (export-bound). Placement and timing tables: [docs/benchmarks.md](docs/benchmarks.md#distributed-pipeline).

<table width="100%">
  <tr>
    <th width="50%" align="center"><code>era5_pretrained</code> (5090)</th>
    <th width="50%" align="center"><code>hres_0.1</code> (5090)</th>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/image/distributed_rollout_utilization_5090_era5_pretrained_2gpu.png" width="100%"/></td>
    <td width="50%" valign="top"><img src="docs/image/distributed_rollout_utilization_5090_hres_0.1_2gpu.png" width="100%"/></td>
  </tr>
</table>

## Install

```bash
git clone <repository-url>
cd flash-aurora
uv sync
```

Dependencies are declared in `pyproject.toml` and pinned by `uv.lock`. Set `CUTE_DSL_ARCH` when needed (for example `sm_89` or `sm_120a`). API sketches and a minimal forecast script: [docs/tutorial.md](docs/tutorial.md#quick-start).

## Repository layout

| Path | Role |
| ---- | ---- |
| `flash_aurora/models/aurora/` | Legacy optimized Aurora family (upstream freeze **v1.8.0**). |
| `flash_aurora/models/aurora_v1p5/` | Aurora 1.5 side path (`v2.0.0` semantics); shares ops and `inference_precision`. |
| `flash_aurora/models/ops/` | Shared Triton and CuTe kernels. |
| `flash_aurora/models/inference_precision.py` | Named precision presets. |
| `flash_aurora/engine/` | Preset Engine: core, ingress, egress, runtime, distributed. |
| `flash_aurora/scheduler/` | ZeroMQ worker, coordinator, client, supervisor. |
| `docs/` | Notebooks, [tutorial.md](docs/tutorial.md), [benchmarks.md](docs/benchmarks.md). |
| `benchmark/` | Latency, precision, kernel, and distributed harnesses. |
| `tests/` | Model, kernel, engine, and scheduler tests (`./scripts/run_tests.sh`). |

## Reading guide

1. Run forecasts in process: [Engine](#engine) then [docs/tutorial.md](docs/tutorial.md) or `docs/example_*.ipynb`.
2. Fit a preset that exceeds one GPU: [Distributed pipeline](#distributed-pipeline) and [docs/benchmarks.md](docs/benchmarks.md#distributed-pipeline).
3. Serve outside the notebook: [Forecast scheduler](#forecast-scheduler-zmq) and [docs/tutorial.md](docs/tutorial.md#forecast-scheduler-deployment).
4. Compare precision or latency: [Precision tiers](#precision-tiers) and [docs/benchmarks.md](docs/benchmarks.md).

## Engine

`flash_aurora.engine` is the preset-driven inference layer: model variants, data profiles, checkpoints, batch validation, rollout, NetCDF export, and optional single-process multi-GPU pipeline.

### Architecture

| Layer | Path | Role |
| ----- | ---- | ---- |
| Core | `engine/core/` | `EngineConfig`, presets, `AuroraEngine`, checkpoints, `RolloutSession`, lifecycle. |
| Ingress | `engine/ingress/` | Downloaders, adapters, `InitialConditionBuilder`, validator, optional IC cache. |
| Egress | `engine/egress/` | Offload, step NetCDF naming, optional async export. |
| Runtime | `engine/runtime/` | Warmup / graph pool, `GpuGuard`, VRAM estimates, resource monitor. |
| Distributed | `engine/distributed/` | Placement plan and pipeline forward for multi-GPU rollout. |

A preset pairs a `ModelVariantSpec` with a `SourceProfile`. `DataDownloader.ensure()` fills the cache; `InitialConditionBuilder` builds a validated `Batch`; `AuroraEngine.load()` applies `inference_precision` and optional pipeline placement; `rollout_stream` advances $K$ steps by $\Delta t$; `rollout_and_export` can overlap D2H and NetCDF writes with the next forward when `async_export=True`.

### Presets and data sources

| Preset | Model | Grid $H \times W$ | Source | Backend |
| ------ | ----- | ----------------- | ------ | ------- |
| `era5_pretrained` | AuroraPretrained | $721 \times 1440$ | CDS ERA5 | CDS |
| `aurora_v1p5` | AuroraV1p5 | $721 \times 1440$ | CDS ERA5 (extended) | CDS |
| `aurora_v1p5_ensemble` | AuroraV1p5Ensemble | $721 \times 1440$ | CDS ERA5 (extended) | CDS |
| `hres_t0_finetuned` | Aurora (LoRA) | $721 \times 1440$ | WeatherBench2 HRES | WB2 + ERA5 static |
| `small_pretrained` | AuroraSmallPretrained | $400 \times 800$ | CDS ERA5 | CDS |
| `hres_0.1` | AuroraHighRes | $1801 \times 3600$ | IFS analysis | ECMWF Open Data / GRIB |
| `cams` | AuroraAirPollution | $451 \times 900$ | CAMS | ADS |
| `wave` | AuroraWave | $721 \times 1440$ | WB2 met + MARS wave | WB2 + MARS |
| `tc_tracking` | Aurora (LoRA) | $721 \times 1440$ | WeatherBench2 HRES | WB2 + ERA5 static |

Personal ECMWF accounts typically lack MARS access; see `docs/example_wave.ipynb` for manual wave GRIB cache setup.

### Capabilities (summary)

Local checkpoints with optional Hub download (`allow_hub_download`, `HF_MIRROR_ENDPOINT`); `inference_precision` for Triton / CuTe / BF16 / TF32 routing on the legacy family and Aurora 1.5 backbone; `DistributedConfig` for legacy multi-GPU pipeline (not enabled for Aurora 1.5); ingress for CDS (including Aurora 1.5 extended surface), ADS, WeatherBench2, Open Data GRIB, and MARS when permitted; `fine_lead_times` on Aurora 1.5 rollouts; optional async export, IC load overlap, disk `ic_cache`, and `GpuGuard`. CUDA graph capture remains experimental and is forced off for Aurora 1.5.

API sketches, lifecycle notes, and IC-cache examples: [docs/tutorial.md](docs/tutorial.md).

### Distributed pipeline

Single-process pipeline parallelism for presets that exceed one GPU. Pass `distributed=DistributedConfig(devices=("cuda:0", "cuda:1"), ...)` to `from_preset` or set `engine.config.distributed` before `load()`. Benchmarks: [docs/benchmarks.md](docs/benchmarks.md#distributed-pipeline) and `benchmark/bench_distributed_rollout.py`.

### Forecast scheduler (ZMQ)

Long-lived workers each own one GPU and one preset. Clients send JSON commands and receive step events (`export_paths`, `metadata_only`, or `last_step_array`). Deployment commands, client sketches, notebook list, and supervisor CLI: [docs/tutorial.md](docs/tutorial.md#forecast-scheduler-deployment).

## Precision tiers

Tiers are labeled `backbone@encoder_decoder` (for example `bf16_mixed@fp32`).

| Backbone token | Meaning |
| -------------- | ------- |
| `fp32` | Strict FP32 GEMM; PyTorch SDPA unless replaced. |
| `tf32` | TF32 Tensor Core GEMM; CuTe window attention (FP32 I/O). |
| `bf16_mixed` | BF16 attention QKV/proj and MLP; TF32 elsewhere; CuTe BF16 attention. |
| `bf16` | Full backbone BF16 GEMM with fused CuTe attention (not recommended for production). |

Encoder/decoder tokens are `fp32` or `tf32`. Set via `inference_precision=` on `from_preset` or `EngineConfig`.

All custom tiers enable the same Triton fusion base (layout and AdaLN). CuTe attention and GEMM precision layer on top. The PyTorch reference tier `pytorch_backbone_fp32_encoder_decoder_fp32` disables Triton and CuTe for drift baselines.

Official per-variable tolerances $\tau_v$ and full drift tables: [docs/benchmarks.md](docs/benchmarks.md#official-per-variable-tolerances).

## Window attention kernels

Flash-Aurora replaces PyTorch SDPA on Swin windows with CuTe DSL kernels under `flash_aurora/models/ops/cute/`. For production $0.25^{\circ}$ stages with $N = 144$, a single shared-memory tile holds the full window $K$ and $V$ (`tile_n \ge N`). Logits use FP32 online softmax; the full $N \times N$ map is not written to global memory. Modes `BF16_MIXED` and `TF32_ACC_FP32` trade throughput against FP32 fidelity. Microbenchmark tables: [docs/benchmarks.md](docs/benchmarks.md#window-attention-microbenchmarks).

## Benchmarks (summary)

End-to-end numbers use NVIDIA RTX PRO 6000 Blackwell, PyTorch $2.12.1+\mathrm{cu}130$, `CUTE_DSL_ARCH=sm_120a`, warmup $2$, repeat $5$, and `--isolate-tiers` for fair speedups. The wave preset is omitted (MARS). Tier `bf16@*` is excluded from latency tables (no speed gain over `bf16_mixed@*`, larger drift).

**`aurora_v1p5` latency** ($721 \times 1440$, 2026-07-14):

| Tier | forward ($\mathrm{ms}$) | vs PyTorch FP32 |
| ---- | ----------------------: | --------------: |
| `bf16_mixed@fp32` | $700.8$ | $3.12\times$ |
| `tf32@tf32` | $946.0$ | $2.31\times$ |
| PyTorch FP32 ref | $2185.8$ | base |

**`aurora_v1p5` precision** (seed $42$, baseline PyTorch FP32): recommended tiers (`bf16_mixed@*`, `tf32@*`, `fp32@fp32`) pass $31/31$ output variables; `bf16@fp32` and plain PyTorch autocast fail selected extended surface fields.

Full latency grids, precision-drift tables for all presets, reproduce commands, and distributed rollout numbers: [docs/benchmarks.md](docs/benchmarks.md).

## Testing notes

`test_aurora_small` compares FP64 forwards to Microsoft Hugging Face reference pickles. On recent PyTorch builds a small drift on a few surface variables can appear even with the official `microsoft-aurora` wheel; the test passes and emits a `UserWarning` when drift exceeds upstream tolerances.

## License

This repository is licensed under the [MIT License](LICENSE).

- `flash_aurora.models.aurora` is derived from [Microsoft Aurora](https://github.com/microsoft/aurora) (MIT), frozen at upstream **v1.8.0**. See `flash_aurora/models/aurora/LICENSE.txt` and `NOTICE.md`.
- `flash_aurora.models.aurora_v1p5` is the Aurora 1.5 side path from tag `v2.0.0` (MIT). Presets: `aurora_v1p5` / `aurora_v1p5_ensemble`. See `flash_aurora/models/aurora_v1p5/LICENSE.txt` and `NOTICE.md`.
- Shared kernels live under `flash_aurora.models.ops` (see per-file headers, including NVIDIA BSD-3-Clause where noted).

## Reference

**Aurora model.** Bodnar et al., *A Foundation Model for the Earth System*, Nature (2025). [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y). Upstream: [microsoft.github.io/aurora](https://microsoft.github.io/aurora).

**CUTLASS / CuTe DSL.** Window-attention and dense GEMM kernels under `flash_aurora/models/ops/cute/` adapt patterns from [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) (BSD-3-Clause). Runtime package: `nvidia-cutlass-dsl`.

**Flash Attention.** FMHA mainloop and online softmax structure follow [flash-attn](https://github.com/Dao-AILab/flash-attention) (Tri Dao).
