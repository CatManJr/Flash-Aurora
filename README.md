# Flash-Aurora: One More Step Toward Efficient Geospatial FOundation Models Inference Service

High-performance inference stack for the [Microsoft Aurora](https://github.com/microsoft/aurora) Earth-system foundation model. The repository packages the upstream model under `flash_aurora.aurora`, custom CUDA kernels (Triton and CuTeDSL), precision routing, and `flash_aurora.engine` for data ingress, checkpoints, rollout, and export.

## Install

```bash
git clone <repository-url>
cd flash-aurora
uv sync
```

Dependencies include PyTorch, `nvidia-cutlass-dsl[cu13]`, and `quack-kernels` (see `pyproject.toml`). CuTe DSL JIT-compiles for the local GPU by default.

## Repository layout

| Path | Role |
|------|------|
| `flash_aurora/aurora/` | Aurora model fork (upstream README preserved in place). See `NOTICE.md` and `LICENSE.txt`. |
| `flash_aurora/engine/` | `AuroraEngine`, presets, NetCDF ingress, rollout, export. |
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

| Tier | Forward (ms) | Speedup vs PyTorch FP32 ref | Mean abs error vs ref |
|------|--------------|----------------------------|------------------------|
| `bf16_mixed@fp32` | 681.9 | 3.15x | 0.115 |
| `bf16_mixed@tf32` | 681.4 | 3.16x | 0.115 |
| `bf16@fp32` | 680.5 | 3.16x | 0.191 |
| `tf32@tf32` | 931.0 | 2.31x | 0.060 |
| `tf32@fp32` | 1093.3 | 1.97x | 0.018 |
| `fp32@fp32` | 1988.9 | 1.08x | 5.4e-05 |
| PyTorch FP32 ref (no Triton/CuTe) | 2151.0 | 1.00x | 0 |
| PyTorch autocast BF16 backbone | 1013.7 | 2.12x | 0.140 |

The PyTorch FP32 reference uses no custom kernels. Every other custom tier uses Triton layout and AdaLN fusion; `tf32` and BF16 tiers additionally use CuTe window attention and the corresponding backbone matmul mode. All custom tiers pass per-variable mean relative-error tolerances from the upstream golden tests on this ERA5 sample when compared to the PyTorch FP32 reference. Recommended production preset on this hardware: `bf16_mixed@fp32` or `bf16_mixed@tf32` (about 3x speedup with bounded drift).

## AuroraEngine

`AuroraEngine` integrates the model, precision presets, and I/O into one API:

1. **Preset registry** -- Named bundles (for example `era5_pretrained`) pair a model variant, checkpoint, and data source profile (CDS ERA5, WeatherBench2 HRES, CAMS, wave).
2. **Checkpoint resolution** -- Local `asset_root` with optional Hugging Face Hub download and mirror support.
3. **Precision wiring** -- `EngineConfig.inference_precision` enables the Triton fusion base and, when selected, TF32/BF16 matmul and CuTe attention consistently across the forward path.
4. **Ingress and validation** -- `InitialConditionBuilder` and `BatchValidator` for NetCDF and adapter inputs.
5. **Rollout and export** -- Multi-step `rollout_stream` and optional `cuda_graph` warmup for fixed-shape backbone replay.

```python
from flash_aurora import AuroraEngine

engine = AuroraEngine.from_preset("era5_pretrained", asset_root="/path/to/assets")
engine.config.inference_precision = "bf16_mixed@fp32"
engine.load()
engine.warmup()
predictions = engine.run_from_netcdf("/path/to/era5.nc", steps=1)
```

## Testing notes

`test_aurora_small` compares FP64 forward outputs to Microsoft Hugging Face reference pickles. On recent PyTorch builds (for example 2.12.x), a small drift on a few surface variables can appear even with the official `microsoft-aurora` wheel. The test passes and emits a `UserWarning` when drift exceeds upstream tolerances. Use the [vanilla microsoft-aurora library](https://microsoft.github.io/aurora) to verify on your stack.

## Reference

Bodnar et al., *A Foundation Model for the Earth System*, Nature (2025). [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)

Upstream model documentation: [microsoft.github.io/aurora](https://microsoft.github.io/aurora)
