# Flash-Aurora: Toward Efficient Inference for Geospatial Foundation Models

Flash-Aurora is a high-performance inference stack for the [Microsoft Aurora](https://github.com/microsoft/aurora) Earth-system foundation model. The repository packages the upstream model in `flash_aurora.aurora`, custom GPU kernels (Triton and CuTe DSL), precision routing, and `flash_aurora.engine` for data ingress, checkpoint loading, rollout, and NetCDF export.

## Install

```bash
git clone <repository-url>
cd flash-aurora
uv sync
```

Dependencies include PyTorch, `nvidia-cutlass-dsl[cu13]`, and `quack-kernels` (see `pyproject.toml`). CuTe DSL kernels JIT-compile for the local GPU architecture (set `CUTE_DSL_ARCH`, for example `sm_120a` on NVIDIA Blackwell). Remaining Python dependencies follow [Microsoft Aurora](https://github.com/microsoft/aurora).

## Repository layout


| Path                              | Role                                                                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `flash_aurora/aurora/`            | Aurora model fork (upstream README preserved in place). See `NOTICE.md` and `LICENSE.txt`.            |
| `flash_aurora/engine/`            | `AuroraEngine`, presets, ingress/download, rollout, export, GPU guard.                                |
| `flash_aurora/aurora/ops/triton/` | Fused Swin layout and AdaLN kernels.                                                                  |
| `flash_aurora/aurora/ops/cute/`   | CuTe DSL window self-attention kernels.                                                               |
| `benchmark/`                      | Kernel- and model-level timing scripts (`bench_aurora_latency_all.py`, `bench_window_attn.py`, etc.). |
| `tests/`                          | `tests/aurora`, `tests/kernels`, `tests/engine`.                                                      |


Run tests: `./scripts/run_tests.sh`

## Inference precision tiers

Tiers use the label `backbone@encoder_decoder`, for example `bf16_mixed@fp32`.

**Backbone token (left):** matrix-multiply and window-attention mode for the Swin3D backbone.


| Token        | Backbone                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `fp32`       | Strict FP32 GEMM; PyTorch scaled dot-product attention (SDPA) unless a higher tier replaces it.   |
| `tf32`       | TF32 Tensor Core GEMM; CuTe DSL window attention (FP32 I/O).                                      |
| `bf16_mixed` | Hybrid BF16 on attention QKV/proj and MLP, TF32 elsewhere; CuTe DSL window attention (BF16).    |
| `bf16`       | Full backbone BF16 GEMM with fused CuTe DSL attention.                                              |


**Encoder/decoder token (right):** Perceiver GEMM precision, either `fp32` (strict) or `tf32` (Tensor Cores).

Set a tier on the model (`inference_precision=...`) or on `EngineConfig` when using `AuroraEngine`.

### Triton fusion on every custom tier

All custom `inference_precision` tiers (`fp32@*`, `tf32@*`, `bf16_mixed@*`, `bf16@*`) enable the same Triton fusion base:

- **Fused window layout** (`use_triton_layout`): roll, pad, partition, and reverse in fused kernels instead of many small eager allocations.
- **Fused AdaLN and residual** (`use_triton_adaln`): adaptive layer normalization and FiLM modulation fused with the residual add on the Swin hot path.

These kernels run regardless of whether the backbone uses FP32, TF32, or BF16 GEMM. They reduce peak activation memory and improve memory bandwidth relative to the decomposed PyTorch Swin path. CuTe DSL window attention and backbone GEMM precision stack on this Triton base for `tf32` and BF16 tiers. The pure PyTorch reference in benchmarks (`pytorch_backbone_fp32_encoder_decoder_fp32`) disables Triton and CuTe DSL for accuracy baselining only.

`InferenceWorkspacePool` optionally reuses a scratch buffer for the backbone-decoder concat on fixed-shape inference, avoiding repeated large allocations.

## Window attention kernel performance

Swin window self-attention is the dominant cost in the Aurora backbone. Flash-Aurora replaces PyTorch `scaled_dot_product_attention` on this path with hand-written **CuTe DSL** kernels (`flash_aurora/aurora/ops/cute/`), following the tiled fused multi-head attention (FMHA) structure used in FlashAttention: load $Q$, $K$, $V$ tiles into shared memory, form logits $S = \mathrm{scale}\, Q K^\top$ with warp MMA, apply the Swin mask, run **row-wise online softmax** in FP32 registers, then accumulate $O \leftarrow \mathrm{softmax}(S)\, V$ without materializing the full $N \times N$ attention matrix.

**Tensor layout.** Inputs are $(B_{\mathrm{win}}, H, N, D_h)$, where $B_{\mathrm{win}} = B \cdot n_W$ folds batch and window index, $N$ is tokens per window (144 on the default $0.25^{\circ}$ encoder), and $D_h$ is head dimension (64). Shifted-window masks are FP32 additive biases of $-100$ in PyTorch; the CuTe path packs them once to a compact `uint8` mask and applies the equivalent unscaled bias inside the kernel so logits match SDPA.

**Two precision modes** (`WinAttnPrecision`) trade Tensor Core throughput against fidelity to strict FP32. The model selects the mode from activation dtype at the callsite (`swin3d.WindowAttention`).

| Mode | Activations | $QK^\top$ MMA | Softmax / $PV$ | FP32 fidelity |
|------|-------------|---------------|----------------|---------------|
| `TF32_ACC_FP32` | FP32 in/out | TF32 Tensor Cores (`mma.sync…tf32.tf32.f32`) | FP32 online softmax; $P$ cast to BF16 for the $PV$ tile; $V$ may be converted on load | Matches **strict FP32** SDPA within $\sim 10^{-3}$ relative error (kernel tests vs `allow_tf32=False` reference). Used by `tf32@*` tiers. |
| `BF16_MIXED` | BF16 in/out | BF16 Tensor Cores with **FP32 accumulators** (`mma.sync…bf16.bf16.f32`) | Same FP32 softmax; $PV$ stays in the BF16 MMA path | Matches **BF16 SDPA** within $\sim 2\%$ relative error. End-to-end `bf16_mixed@*` runs attention in BF16 but keeps FP32 activations between Swin blocks so the rest of the backbone stays numerically close to FP32. |

In both modes the numerically sensitive steps—logit scaling, masked softmax normalization, and row sums—stay in **FP32**. Lower precision is confined to the two GEMMs ($QK^\top$ and $PV$), which is the standard mixed-precision recipe for attention: approximate the matmuls, keep the exponential normalization exact. `TF32_ACC_FP32` therefore tracks FP32 SDPA (not cuDNN TF32 SDPA) to about three significant figures; `BF16_MIXED` accepts the larger BF16 matmul error but remains within upstream per-variable drift tolerances on full-model rollouts (see **Precision drift** below).

**Kernel variants.** Tile sizes $(tile_m, tile_n)$ are chosen from $N$ and $D_h$ (`_smem_utils.py`). When $tile_n \ge N$ (production $N=144$), the attention fits in a **single KV tile**; the default BF16 kernel uses a 128-thread `cp.async` mainloop. When $tile_n < N$ (coarser downsampled stages), a **TMA stream** kernel double-buffers $K$ and $V$. TF32 uses an analogous tiled loop; on single-pass paths $V$ is cast FP32$\to$BF16 inside the kernel to fuse away a host-side cast. A **QKV-packed** entry point avoids separate $Q$, $K$, $V$ tensors on the fused BF16 attention chain. Kernels JIT-compile per $(D_h, N, \mathrm{has\_bias}, tile)$ via CuTe DSL and are cached for the process.

### Microbenchmarks

Measured with `benchmark/bench_window_attn.py` on an **NVIDIA RTX PRO 6000 Blackwell Server Edition**, PyTorch **2.12.1**, `CUTE_DSL_ARCH=sm_120a` (trimmed mean of 200 runs per shape). $B_{\mathrm{win}}$ is the number of spatial tokens per window; $H$ is the head count.

**0.25-degree ERA5 encoder stages** (unmasked, $N=144$ tokens per window):


| Stage | $B_{\mathrm{win}}$ | $H$ | BF16 CuTe DSL (ms) | BF16 SDPA (ms) | Speedup |
| ----- | ------------------ | --- | ------------------ | -------------- | ------- |
| 1     | 1800               | 8   | 0.727              | 0.780          | 1.07x   |
| 2     | 450                | 16  | 0.374              | 0.407          | 1.09x   |
| 3     | 128                | 32  | 0.220              | 0.239          | 1.09x   |



| Stage | $B_{\mathrm{win}}$ | $H$ | TF32 CuTe DSL (ms) | FP32 SDPA (ms) | Speedup |
| ----- | ------------------ | --- | ------------------ | -------------- | ------- |
| 1     | 1800               | 8   | 1.613              | 2.582          | 1.60x   |
| 2     | 450                | 16  | 0.819              | 1.308          | 1.60x   |
| 3     | 128                | 32  | 0.477              | 0.760          | 1.59x   |


**Shifted-window mask** (Swin relative position bias $-100$):


| Mode           | Stage 1 ($B_{\mathrm{win}}=1800$, $H=8$) | Speedup vs SDPA |
| -------------- | --------------------------------------- | --------------- |
| BF16 CuTe DSL  | 0.829 ms vs 1.014 ms                    | 1.22x           |
| TF32 CuTe DSL  | 1.906 ms vs 3.221 ms                    | 1.69x           |


Production inference on the default $0.25^{\circ}$ grid uses $N=144$ windows per stage. BF16 CuTe DSL attention requires at least 32 tokens per window; on coarser downsampled stages with smaller $N$, use `tf32` or PyTorch SDPA.

## End-to-end benchmarks

**Hardware and software:** NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch **2.12.1+cu130**, CUDA 13.0, `CUTE_DSL_ARCH=sm_120a`, batch size 1, real cached ingress (no download). Custom tiers include the Triton fusion base (layout and AdaLN). The PyTorch FP32 reference (`pytorch_backbone_fp32_encoder_decoder_fp32`) disables Triton and CuTe DSL. Finetuned presets report `lora_eager` versus `lora_merged`; pretrained presets report a single forward latency (no LoRA).

The `wave` preset is omitted from every benchmark table. Its ingress requires MARS wave GRIB from the ECMWF archive; personal API accounts typically lack MARS access, so reproducible end-to-end runs are not available in this environment (see `docs/example_wave.ipynb` for manual cache setup).

**Note on `bf16@*`:** excluded from latency tables (no speed win over `bf16_mixed@*`; worse precision drift). See the precision section below.

### Forward latency (all presets)

Two harness modes are reported side by side.

| Mode | Flag | Use |
|------|------|-----|
| Fair speedup | `--isolate-tiers` (default) | Each preset-by-tier pair in a fresh subprocess; use for vs-ref ratios and headline numbers. |
| Single-process | `--no-isolate-tiers` | All tiers in one process; illustrates how cuDNN autotune warms across tiers and can deflate the PyTorch FP32 reference when it is timed after custom kernels. Custom-tier absolute latency is stable; only vs ref is misleading. |

Both modes use warmup 2 and repeat 5. Speedup vs ref uses `lora_merged` on finetuned presets and forward latency on pretrained presets, each relative to `pytorch_backbone_fp32_encoder_decoder_fp32`. Machine-readable reports: `benchmark/latency_all_isolated.md` (fair) and `benchmark/latency_all_single_process.md` (artifact).

**Single-process reference deflation:** On `era5_pretrained`, the PyTorch FP32 reference is ${\sim}2128$ ms when isolated but ${\sim}1135$ ms when timed after Triton/CuTe DSL tiers in the same process (${\sim}1.9\times$ faster). `bf16_mixed@fp32` remains ${\sim}676$ ms in both runs, so the headline ratio shifts from ${\sim}3.15\times$ to ${\sim}1.68\times$ even though production custom latency is unchanged.

**Finetuned insight:** On finetuned models, custom-precision tier gaps are narrower than the headline pretrained-vs-ref ratio suggests once encoder and decoder time (${\sim}190$ ms on $721 \times 1440$) and backbone copy/cast overhead are included (see `bench_aurora_finetuned_stage_timing.py`). LoRA eager adds a second low-rank GEMM ($1.28\times$--$1.38\times$ eager/merged on weather presets). LoRA merge is orthogonal to precision tier choice. On CAMS, fair isolated `bf16_mixed@*` shows ${\sim}1.31\times$ eager/merged; `lora_merged` (${\sim}571$ ms) is the production end-to-end metric.

#### Fair speedup (`--isolate-tiers`)

Generated 2026-06-23; full tables: `benchmark/latency_all_isolated.md`.

##### `era5_pretrained` ($721 \times 1440$)


| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| `bf16_mixed@fp32` | 676.4 | 3.15x |
| `bf16_mixed@tf32` | 676.8 | 3.14x |
| `tf32@fp32` | 1077.5 | 1.98x |
| `tf32@tf32` | 919.2 | 2.32x |
| `fp32@fp32` | 1945.0 | 1.09x |
| PyTorch autocast | 1004.4 | 2.12x |
| PyTorch FP32 ref | 2128.2 | base |


##### `small_pretrained` ($400 \times 800$)


| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| `bf16_mixed@fp32` | 42.4 | 2.40x |
| `bf16_mixed@tf32` | 42.4 | 2.40x |
| `tf32@fp32` | 64.1 | 1.59x |
| `tf32@tf32` | 57.3 | 1.78x |
| `fp32@fp32` | 94.7 | 1.08x |
| PyTorch autocast | 56.3 | 1.81x |
| PyTorch FP32 ref | 101.9 | base |


##### `hres_t0_finetuned` ($721 \times 1440$, LoRA)


| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| `bf16_mixed@fp32` | 881.7 | 638.7 | 1.38x | 3.23x |
| `bf16_mixed@tf32` | 881.6 | 638.4 | 1.38x | 3.23x |
| `tf32@fp32` | 1249.6 | 1006.3 | 1.24x | 2.05x |
| `tf32@tf32` | 1091.5 | 846.5 | 1.29x | 2.44x |
| `fp32@fp32` | 2115.5 | 1890.4 | 1.12x | 1.09x |
| PyTorch autocast | 1104.4 | 967.7 | 1.14x | 2.13x |
| PyTorch FP32 ref | 2307.9 | 2061.9 | 1.12x | base |


##### `hres_0.1` ($1801 \times 3600$, LoRA)


| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| `bf16_mixed@fp32` | 898.0 | 672.0 | 1.34x | 2.97x |
| `bf16_mixed@tf32` | 898.6 | 672.4 | 1.34x | 2.97x |
| `tf32@fp32` | 1247.7 | 1019.9 | 1.22x | 1.96x |
| `tf32@tf32` | 1091.1 | 861.3 | 1.27x | 2.32x |
| `fp32@fp32` | 2051.0 | 1838.0 | 1.12x | 1.09x |
| PyTorch autocast | 1112.1 | 986.2 | 1.13x | 2.02x |
| PyTorch FP32 ref | 2227.5 | 1994.6 | 1.12x | base |


##### `cams` ($451 \times 900$, LoRA)


| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| `bf16_mixed@fp32` | 747.3 | 571.0 | 1.31x | 2.96x |
| `bf16_mixed@tf32` | 747.5 | 571.9 | 1.31x | 2.96x |
| `tf32@fp32` | 1096.1 | 916.5 | 1.20x | 1.85x |
| `tf32@tf32` | 898.1 | 718.3 | 1.25x | 2.35x |
| `fp32@fp32` | 1734.6 | 1562.3 | 1.11x | 1.08x |
| PyTorch autocast | 985.7 | 888.6 | 1.11x | 1.90x |
| PyTorch FP32 ref | 1874.5 | 1691.6 | 1.11x | base |


##### `tc_tracking` ($721 \times 1440$, LoRA)


| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| `bf16_mixed@fp32` | 881.7 | 638.5 | 1.38x | 3.23x |
| `bf16_mixed@tf32` | 881.7 | 638.3 | 1.38x | 3.23x |
| `tf32@fp32` | 1249.6 | 1006.1 | 1.24x | 2.05x |
| `tf32@tf32` | 1092.0 | 847.0 | 1.29x | 2.43x |
| `fp32@fp32` | 2115.1 | 1890.9 | 1.12x | 1.09x |
| PyTorch autocast | 1104.2 | 967.4 | 1.14x | 2.13x |
| PyTorch FP32 ref | 2307.9 | 2059.9 | 1.12x | base |

#### Single-process artifact (`--no-isolate-tiers`)

Custom tiers run first in one process; the PyTorch FP32 reference is timed last, so cuDNN state from Triton/CuTe DSL is already warm. Custom-tier absolute latency matches the isolated run; vs ref is understated.

##### `era5_pretrained` ($721 \times 1440$)


| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| `bf16_mixed@fp32` | 676.7 | 1.68x |
| `bf16_mixed@tf32` | 677.1 | 1.68x |
| `tf32@fp32` | 920.7 | 1.23x |
| `tf32@tf32` | 921.3 | 1.23x |
| `fp32@fp32` | 944.9 | 1.20x |
| PyTorch autocast | 846.5 | 1.34x |
| PyTorch FP32 ref | 1135.5 | base |


##### `small_pretrained` ($400 \times 800$)


| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| `bf16_mixed@fp32` | 41.9 | 1.59x |
| `bf16_mixed@tf32` | 41.8 | 1.59x |
| `tf32@fp32` | 57.7 | 1.15x |
| `tf32@tf32` | 57.1 | 1.17x |
| `fp32@fp32` | 59.6 | 1.12x |
| PyTorch autocast | 49.5 | 1.34x |
| PyTorch FP32 ref | 66.5 | base |


##### `hres_t0_finetuned` ($721 \times 1440$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 882.4           | 639.1            | 1.38x        | 1.66x               |
| `bf16_mixed@tf32` | 882.2           | 639.1            | 1.38x        | 1.66x               |
| `tf32@fp32`       | 1091.7          | 847.6            | 1.29x        | 1.25x               |
| `tf32@tf32`       | 1092.5          | 848.2            | 1.29x        | 1.25x               |
| `fp32@fp32`       | 1115.6          | 874.2            | 1.28x        | 1.21x               |
| PyTorch autocast  | 946.0           | 808.8            | 1.17x        | 1.31x               |
| PyTorch FP32 ref  | 1308.5          | 1059.9           | 1.23x        | base                |


##### `hres_0.1` ($1801 \times 3600$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 897.2           | 672.4            | 1.33x        | 1.58x               |
| `bf16_mixed@tf32` | 896.8           | 671.8            | 1.33x        | 1.58x               |
| `tf32@fp32`       | 1089.6          | 862.5            | 1.26x        | 1.23x               |
| `tf32@tf32`       | 1089.8          | 863.6            | 1.26x        | 1.23x               |
| `fp32@fp32`       | 1111.9          | 889.4            | 1.25x        | 1.19x               |
| PyTorch autocast  | 955.5           | 829.4            | 1.15x        | 1.28x               |
| PyTorch FP32 ref  | 1289.8          | 1060.3           | 1.22x        | base                |


##### `cams` ($451 \times 900$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 747.4 | 571.4 | 1.31x | 1.53x |
| `bf16_mixed@tf32` | 747.6 | 571.2 | 1.31x | 1.53x |
| `tf32@fp32` | 897.9 | 719.0 | 1.25x | 1.22x |
| `tf32@tf32` | 898.0 | 719.8 | 1.25x | 1.21x |
| `fp32@fp32` | 915.3 | 738.8 | 1.24x | 1.18x |
| PyTorch autocast | 788.0 | 690.4 | 1.14x | 1.27x |
| PyTorch FP32 ref | 1054.5 | 874.5 | 1.21x | base |


On CAMS in single-process mode, the PyTorch FP32 reference merged latency (${\sim}875$ ms) is still deflated relative to fair isolated (${\sim}1692$ ms), so vs-ref speedup (${\sim}1.53\times$) understates the fair ratio (${\sim}2.96\times$). Custom-tier absolute ms matches the isolated run.

##### `tc_tracking` ($721 \times 1440$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 881.7           | 639.2            | 1.38x        | 1.66x               |
| `bf16_mixed@tf32` | 882.1           | 638.6            | 1.38x        | 1.66x               |
| `tf32@fp32`       | 1091.8          | 847.8            | 1.29x        | 1.25x               |
| `tf32@tf32`       | 1093.2          | 848.8            | 1.29x        | 1.25x               |
| `fp32@fp32`       | 1115.6          | 874.5            | 1.28x        | 1.21x               |
| PyTorch autocast  | 945.6           | 808.7            | 1.17x        | 1.31x               |
| PyTorch FP32 ref  | 1308.5          | 1059.3           | 1.24x        | base                |


Recommended production tiers: `bf16_mixed@fp32` or `bf16_mixed@tf32` for weather presets (always `lora_merged`); on CAMS use `lora_merged` with `bf16_mixed@*` for speed, or `tf32@fp32` if strict `pm10` tolerance is required (see precision tables).

### Official per-variable tolerances

Benchmarks compare each tier to the PyTorch FP32 reference using the mean relative error
$\bar{e}_v = \mathrm{mean}(|y_v - \hat{y}_v|) / \mathrm{mean}(|\hat{y}_v|)$
per output variable $v$. A tier **passes** variable $v$ when $\bar{e}_v \le \tau_v$. Tolerances $\tau_v$ follow `tests/aurora/test_model.py` (Microsoft upstream golden tests):


| Variable | $\tau_v$         | Variable | $\tau_v$         |
| -------- | ---------------- | -------- | ---------------- |
| `2t`     | $10^{-4}$        | `u`      | $5\times10^{-3}$ |
| `10u`    | $5\times10^{-3}$ | `v`      | $5\times10^{-3}$ |
| `10v`    | $5\times10^{-3}$ | `q`      | $5\times10^{-3}$ |
| `msl`    | $10^{-4}$        | `t`      | $10^{-4}$        |
| `z`      | $5\times10^{-3}$ |          |                  |


CAMS pollution outputs (`pm1`, `pm2p5`, `pm10`, `tcco`, `tc_no`, `tcno2`, `gtco3`, `tcso2`, `co`, `no`, `no2`, `go3`, `so2`) use a heuristic $\tau_v = 5\times10^{-3}$ (same order as wind and humidity). Upstream does not publish golden tolerances for these channels.

### Precision drift (seed 42, `lora_merged` on finetuned presets)

Measured with `benchmark/bench_aurora_precision_all.py`, seed 42, baseline `pytorch_backbone_fp32_encoder_decoder_fp32`. Entries are $\bar{e}_v$; values above $\tau_v$ are **bold**.

#### `era5_pretrained` ($721 \times 1440$, 9 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `t`      | `u`      | `v`      | `q`      | `z`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 3.84e-05 | 8.23e-04 | 1.01e-03 | 5.53e-06 | 1.05e-05 | 5.00e-04 | 9.35e-04 | 3.78e-04 | 4.40e-06 |
| `bf16_mixed@tf32` | 3.84e-05 | 8.23e-04 | 1.01e-03 | 5.53e-06 | 1.05e-05 | 5.00e-04 | 9.35e-04 | 3.78e-04 | 4.40e-06 |
| `tf32@fp32`       | 3.02e-06 | 1.56e-04 | 2.07e-04 | 7.19e-07 | 3.11e-06 | 1.41e-04 | 2.55e-04 | 1.07e-04 | 1.55e-06 |
| `tf32@tf32`       | 3.02e-06 | 1.56e-04 | 2.07e-04 | 7.19e-07 | 3.11e-06 | 1.41e-04 | 2.55e-04 | 1.07e-04 | 1.55e-06 |
| `fp32@fp32`       | 1.32e-06 | 8.68e-05 | 1.14e-04 | 3.13e-07 | 2.17e-06 | 9.32e-05 | 1.62e-04 | 7.81e-05 | 1.11e-06 |
| PyTorch autocast  | 4.36e-05 | 1.40e-03 | 1.77e-03 | 7.40e-06 | 1.75e-05 | 9.42e-04 | 1.76e-03 | 6.18e-04 | 7.31e-06 |


All tiers pass on every variable.

#### `small_pretrained` ($400 \times 800$, 8 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `u`      | `v`      | `t`      | `q`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 2.63e-05 | 1.61e-03 | 1.87e-03 | 5.83e-06 | 1.09e-03 | 1.76e-03 | 2.20e-05 | 8.12e-04 |
| `bf16_mixed@tf32` | 2.65e-05 | 1.63e-03 | 1.88e-03 | 5.86e-06 | 1.10e-03 | 1.77e-03 | 2.20e-05 | 8.26e-04 |
| `tf32@fp32`       | 1.22e-05 | 4.21e-04 | 4.64e-04 | 2.18e-06 | 3.34e-04 | 5.15e-04 | 7.60e-06 | 2.40e-04 |
| `tf32@tf32`       | 1.22e-05 | 4.21e-04 | 4.64e-04 | 2.18e-06 | 3.34e-04 | 5.15e-04 | 7.60e-06 | 2.40e-04 |
| `fp32@fp32`       | 1.19e-05 | 3.59e-04 | 3.89e-04 | 2.00e-06 | 2.95e-04 | 4.36e-04 | 7.11e-06 | 2.17e-04 |
| PyTorch autocast  | 3.55e-05 | 2.58e-03 | 2.93e-03 | 8.29e-06 | 1.65e-03 | 2.65e-03 | 2.86e-05 | 1.16e-03 |


All tiers pass on every variable.

#### `hres_t0_finetuned` ($721 \times 1440$, LoRA merged, 9 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `t`      | `u`      | `v`      | `q`      | `z`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 2.82e-05 | 9.24e-04 | 1.11e-03 | 4.37e-06 | 1.16e-05 | 5.65e-04 | 1.11e-03 | 4.10e-04 | 4.76e-06 |
| `bf16_mixed@tf32` | 2.82e-05 | 9.24e-04 | 1.11e-03 | 4.37e-06 | 1.16e-05 | 5.65e-04 | 1.11e-03 | 4.10e-04 | 4.76e-06 |
| `tf32@fp32`       | 3.09e-06 | 1.82e-04 | 2.31e-04 | 7.46e-07 | 3.30e-06 | 1.53e-04 | 2.85e-04 | 1.13e-04 | 1.63e-06 |
| `tf32@tf32`       | 3.09e-06 | 1.82e-04 | 2.31e-04 | 7.46e-07 | 3.30e-06 | 1.53e-04 | 2.85e-04 | 1.13e-04 | 1.63e-06 |
| `fp32@fp32`       | 1.46e-06 | 1.02e-04 | 1.28e-04 | 3.50e-07 | 2.26e-06 | 9.77e-05 | 1.75e-04 | 8.02e-05 | 1.14e-06 |
| PyTorch autocast  | 3.32e-05 | 1.51e-03 | 1.87e-03 | 6.41e-06 | 1.84e-05 | 1.00e-03 | 1.90e-03 | 6.49e-04 | 7.51e-06 |


All tiers pass on every variable.

#### `hres_0.1` ($1801 \times 3600$, LoRA merged, 9 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `t`      | `u`      | `v`      | `q`      | `z`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 3.20e-05 | 9.35e-04 | 1.11e-03 | 4.64e-06 | 1.18e-05 | 5.97e-04 | 1.13e-03 | 3.98e-04 | 4.50e-06 |
| `bf16_mixed@tf32` | 3.20e-05 | 9.35e-04 | 1.11e-03 | 4.64e-06 | 1.18e-05 | 5.97e-04 | 1.13e-03 | 3.98e-04 | 4.50e-06 |
| `tf32@fp32`       | 3.39e-06 | 1.92e-04 | 2.45e-04 | 7.81e-07 | 3.51e-06 | 1.70e-04 | 3.17e-04 | 1.13e-04 | 1.63e-06 |
| `tf32@tf32`       | 3.39e-06 | 1.92e-04 | 2.45e-04 | 7.81e-07 | 3.51e-06 | 1.70e-04 | 3.17e-04 | 1.13e-04 | 1.63e-06 |
| `fp32@fp32`       | 1.54e-06 | 1.04e-04 | 1.31e-04 | 3.47e-07 | 2.29e-06 | 1.03e-04 | 1.82e-04 | 7.65e-05 | 1.11e-06 |
| PyTorch autocast  | 3.72e-05 | 1.49e-03 | 1.84e-03 | 6.51e-06 | 1.81e-05 | 9.95e-04 | 1.88e-03 | 6.16e-04 | 6.88e-06 |


All tiers pass on every variable.

#### `tc_tracking` ($721 \times 1440$, LoRA merged, 9 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `t`      | `u`      | `v`      | `q`      | `z`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 2.85e-05 | 9.29e-04 | 1.15e-03 | 4.70e-06 | 1.18e-05 | 5.84e-04 | 1.09e-03 | 3.99e-04 | 5.06e-06 |
| `bf16_mixed@tf32` | 2.85e-05 | 9.29e-04 | 1.15e-03 | 4.70e-06 | 1.18e-05 | 5.84e-04 | 1.09e-03 | 3.99e-04 | 5.06e-06 |
| `tf32@fp32`       | 3.03e-06 | 1.77e-04 | 2.35e-04 | 7.64e-07 | 3.33e-06 | 1.56e-04 | 2.77e-04 | 1.08e-04 | 1.69e-06 |
| `tf32@tf32`       | 3.03e-06 | 1.77e-04 | 2.35e-04 | 7.64e-07 | 3.33e-06 | 1.56e-04 | 2.77e-04 | 1.08e-04 | 1.69e-06 |
| `fp32@fp32`       | 1.39e-06 | 9.82e-05 | 1.27e-04 | 3.46e-07 | 2.27e-06 | 9.97e-05 | 1.69e-04 | 7.65e-05 | 1.18e-06 |
| PyTorch autocast  | 3.33e-05 | 1.49e-03 | 1.90e-03 | 6.78e-06 | 1.87e-05 | 1.02e-03 | 1.87e-03 | 6.27e-04 | 7.87e-06 |


All tiers pass on every variable.

#### `cams` ($451 \times 900$, LoRA merged, 22 vars)


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `pm1`    | `pm2p5`      | `pm10`       | `tcco`   | `tc_no`  | `tcno2`  | `gtco3`  | `tcso2`  | `t`      | `u`      | `v`      | `q`      | `z`      | `co`     | `no`     | `no2`    | `go3`    | `so2`    |
| ----------------- | -------- | -------- | -------- | -------- | -------- | ------------ | ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 5.50e-05 | 1.52e-03 | 1.69e-03 | 8.71e-06 | 3.44e-03 | 4.24e-03     | **5.21e-03** | 1.98e-04 | 5.54e-04 | 7.04e-04 | 8.93e-05 | 3.49e-03 | 2.21e-05 | 8.80e-04 | 1.53e-03 | 6.56e-04 | 1.08e-05 | 3.25e-04 | 6.87e-06 | 2.09e-05 | 2.42e-04 | 4.07e-05 |
| `bf16_mixed@tf32` | 5.50e-05 | 1.53e-03 | 1.69e-03 | 8.84e-06 | 3.45e-03 | 4.25e-03     | **5.21e-03** | 2.00e-04 | 5.56e-04 | 7.07e-04 | 8.92e-05 | 3.50e-03 | 2.23e-05 | 8.87e-04 | 1.53e-03 | 6.61e-04 | 1.09e-05 | 3.34e-04 | 6.84e-06 | 2.08e-05 | 2.45e-04 | 4.09e-05 |
| `tf32@fp32`       | 1.07e-05 | 4.71e-04 | 5.19e-04 | 2.37e-06 | 7.73e-04 | 9.99e-04     | 1.27e-03     | 5.82e-05 | 1.28e-04 | 1.49e-04 | 2.62e-05 | 7.62e-04 | 1.12e-05 | 3.87e-04 | 7.33e-04 | 2.58e-04 | 8.23e-06 | 1.54e-04 | 4.21e-06 | 9.13e-06 | 1.01e-04 | 2.57e-05 |
| `tf32@tf32`       | 1.07e-05 | 4.71e-04 | 5.19e-04 | 2.37e-06 | 7.73e-04 | 9.99e-04     | 1.27e-03     | 5.82e-05 | 1.28e-04 | 1.49e-04 | 2.62e-05 | 7.62e-04 | 1.12e-05 | 3.87e-04 | 7.33e-04 | 2.58e-04 | 8.23e-06 | 1.54e-04 | 4.21e-06 | 9.13e-06 | 1.01e-04 | 2.57e-05 |
| `fp32@fp32`       | 1.03e-05 | 4.25e-04 | 4.56e-04 | 2.17e-06 | 7.63e-04 | 9.84e-04     | 1.24e-03     | 4.69e-05 | 1.16e-04 | 1.33e-04 | 2.25e-05 | 6.53e-04 | 1.09e-05 | 3.55e-04 | 6.86e-04 | 2.37e-04 | 8.12e-06 | 1.44e-04 | 4.03e-06 | 8.34e-06 | 9.09e-05 | 2.44e-05 |
| PyTorch autocast  | 6.99e-05 | 4.25e-03 | 4.14e-03 | 1.63e-05 | 4.03e-03 | **5.23e-03** | **6.23e-03** | 3.92e-04 | 8.51e-04 | 1.14e-03 | 1.71e-04 | 4.70e-03 | 3.09e-05 | 1.69e-03 | 2.78e-03 | 9.99e-04 | 1.45e-05 | 5.05e-04 | 8.76e-06 | 2.99e-05 | 3.90e-04 | 5.28e-05 |


On CAMS, `bf16_mixed@*` exceeds $\tau_{\mathrm{pm10}} = 5\times10^{-3}$ by about 4% (meteorological channels remain within tolerance). `tf32@*` and `fp32@fp32` pass all 22 variables.

**Excluded tier `bf16@fp32` (not recommended):** on CAMS, $\bar{e}_{\mathrm{pm2p5}} = 5.32\times10^{-3}$ and $\bar{e}_{\mathrm{pm10}} = 6.35\times10^{-3}$; on `era5_pretrained`, $\bar{e}_{10u} = 1.53\times10^{-3}$ and $\bar{e}_v = 1.96\times10^{-3}$ (within $\tau$ but $2$--$3\times$ higher than `bf16_mixed@fp32`). Latency matches `bf16_mixed@*` on pretrained ERA5 (680.5 ms vs 681.9 ms) and offers no benefit.

### Reproducing the benchmarks

All commands assume PyTorch **2.12.1** from `uv.lock`, CUDA 13.0, and `CUTE_DSL_ARCH=sm_120a` on Blackwell. Populate `asset_root` with checkpoints and cached ingress NetCDF before running.

**End-to-end latency** (all presets except `wave`, every tier, `lora_eager` vs `lora_merged` where applicable):

Fair speedup (default):

```bash
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_latency_all.py \
  --asset-root /path/to/assets --warmup 2 --repeat 5 \
  --isolate-tiers --report-out benchmark/latency_all_isolated.md
```

Single-process artifact (cuDNN cross-tier warmup demo; ref timed after custom tiers):

```bash
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_latency_all.py \
  --asset-root /path/to/assets --warmup 2 --repeat 5 \
  --no-isolate-tiers --defer-ref \
  --report-out benchmark/latency_all_single_process.md
```

`--defer-ref` times the PyTorch FP32 reference after all custom tiers in the same process (reproduces the understated vs-ref column). Omit `--defer-ref` for a quicker single-process run with the reference timed first.

Finetuned-only shortcut (delegates to the same harness): `benchmark/bench_aurora_finetuned_lora.py`.

**Window attention** (CuTe DSL vs PyTorch SDPA micro-benchmark):

```bash
CUTE_DSL_ARCH=sm_120a BENCH_MEASURED=200 uv run python benchmark/bench_window_attn.py
```

**Precision drift** (seed 42, `lora_merged` on finetuned models):

```bash
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_precision_all.py \
  --asset-root /path/to/assets --seed 42
```

Report: `benchmark/precision_all_seed42.md`.

**Stage timing** (encoder / backbone / decoder breakdown; optional cast profiling):

```bash
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_finetuned_stage_timing.py \
  --asset-root /path/to/assets --profile-casts
```

Legacy single-preset timing: `benchmark/bench_aurora_pretrained.py` (subset of `bench_aurora_latency_all.py` for `era5_pretrained` only).

## Engine (`flash_aurora.engine`)

`flash_aurora.engine` is the inference service layer. It binds Aurora variants, upstream data profiles, checkpoint resolution, batch validation, multi-step rollout, and NetCDF export behind a preset-driven API. Tutorial notebooks under `docs/example_*.ipynb` exercise each preset end to end.

### Architecture

The engine has four layers. Data flows from download and adapters into a validated `Batch`, through the loaded model, and optionally to disk as forecast NetCDF.


| Layer   | Path              | Role                                                                                           |
| ------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| Core    | `engine/core/`    | `EngineConfig`, `PresetRegistry`, `AuroraEngine`, checkpoint load, `RolloutSession`.           |
| Ingress | `engine/ingress/` | `DataDownloader`, source adapters, `InitialConditionBuilder`, `BatchValidator`, static fields. |
| Egress  | `engine/egress/`  | `RolloutExporter`, CPU offload, step-wise NetCDF naming.                                       |
| Runtime | `engine/runtime/` | CUDA Graph warmup (`GraphPool`), cross-process `GpuGuard`, VRAM budget estimates.              |


A preset pairs a `ModelVariantSpec` (checkpoint, variable lists, grid shape $H \times W$, timestep $\Delta t$) with a `SourceProfile` (schema, latitude convention, cache layout). `DataDownloader.ensure()` fills the preset cache. `InitialConditionBuilder` reads cached files or adapter requests and attaches Hugging Face static fields. `BatchValidator` checks tensor shapes and variable names against the variant. `AuroraEngine.load()` resolves checkpoints, applies `inference_precision`, and optionally acquires a `GpuGuard` lease from estimated VRAM. `predict()` runs one forward step; `rollout_stream()` chains $K$ steps with model-internal history, advancing valid time by $\Delta t$ per step. `rollout_and_export()` writes CPU-side NetCDF under `export_dir`.

### Presets and data sources


| Preset              | Model                 | Grid $H \times W$   | Source                   | Download backend       |
| ------------------- | --------------------- | ------------------- | ------------------------ | ---------------------- |
| `era5_pretrained`   | AuroraPretrained      | $721 \times 1440$   | CDS ERA5                 | CDS                    |
| `hres_t0_finetuned` | Aurora (LoRA)         | $721 \times 1440$   | WeatherBench2 HRES       | WB2 + ERA5 static      |
| `small_pretrained`  | AuroraSmallPretrained | $400 \times 800$    | CDS ERA5                 | CDS                    |
| `hres_0.1`          | AuroraHighRes         | $1801 \times 3600$  | IFS GRIB analysis        | ECMWF Open Data / GRIB |
| `cams`              | AuroraAirPollution    | $451 \times 900$    | CAMS reanalysis          | ADS                    |
| `wave`              | AuroraWave            | $721 \times 1440$   | WB2 met + MARS wave GRIB | WB2 + MARS             |
| `tc_tracking`       | Aurora (LoRA)         | $721 \times 1440$   | WeatherBench2 HRES       | WB2 + ERA5 static      |


Personal ECMWF accounts typically lack MARS archive access. For `wave`, set direction of `{day}-wave.grib` under the cache manually or use an institutional MARS credential; see `docs/example_wave.ipynb`.

### Capabilities

- **Checkpoint and static assets.** Local `asset_root` with optional Hugging Face Hub download (`allow_hub_download`, mirror via `HF_MIRROR_ENDPOINT`).
- **Precision wiring.** `EngineConfig.inference_precision` selects the Triton fusion base and, when set, TF32/BF16 GEMM and CuTe DSL window attention (see above).
- **Automated ingress.** CDS (ERA5), ADS (CAMS), WeatherBench2 (HRES met), ECMWF Open Data (0.1-degree GRIB), and MARS (wave GRIB when permitted). Credentials merge from environment variables, `~/.cdsapirc`, `~/.ecmwfapirc`, and optional constructor kwargs.
- **Multi-step rollout.** `rollout_stream(batch, K)` and `run_from_netcdf(..., steps=K)`; optional `RolloutObserver` hooks per step.
- **NetCDF export.** `rollout_and_export()` writes forecast steps to `export_dir`. Optional async pipeline (Earth-2 style) overlaps GPU to CPU offload and disk writes with the next forward step.
- **Prepare overlap.** `prepare()` can build initial conditions on a CPU worker thread while the model loads on the main thread.
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

### Lifecycle optimizations

Cold-start time is usually dominated by **CPU ingress** (`build_ic`), **model init** (`build_model` / CuTe JIT), and **NetCDF export**—not GPU forward. Two optional optimizations address that; both are configured on `EngineConfig` / `from_preset()` and can be overridden per call.

| Field | Default | What it does |
|-------|---------|--------------|
| `overlap_ic_load` | `True` | `prepare()` / `prepare_from_netcdf()` runs IC build on a background thread while the model is built and checkpoint-loaded. |
| `async_export` | `False` | `rollout_and_export()` pipelines egress-stream GPU to CPU copy and background NetCDF writes (Earth-2 `AsyncZarrBackend` pattern, one file per step). |
| `export_pool_size` | `2` | Thread-pool size for async NetCDF writes. |
| `export_max_inflight` | `None` | Max queued writes before back-pressure (`pool_size - 1` when unset). |
| `export_use_egress_stream` | `True` | Use a dedicated CUDA stream for D2H during async export. |

**Per-call overrides:** `engine.prepare(request, overlap=False)` and `engine.rollout_and_export(batch, steps=K, async_export=True)` take precedence over the config for that invocation only.

**Which presets benefit**

Measured on RTX PRO 6000 (`bf16_mixed@fp32`, forward warmup 2). Forward/step is ~680 ms on 0.25° and 0.1° presets regardless of these flags.

| Preset | `overlap_ic_load` | `async_export` | Rationale |
|--------|-------------------|----------------|-----------|
| `hres_0.1` | **On** (default) | **On** for `K \gtrsim 4` | IC build (GRIB/regrid) dominates prepare (~2 min); each export step is ~5 s vs ~0.7 s forward. Largest win from both flags. |
| `era5_pretrained`, `hres_t0_finetuned`, `tc_tracking` | On (default) | Optional | Modest prepare overlap (~4–8 s saved). Export is smaller per step; async helps mainly on longer rollouts. |
| `cams` | On (default) | Optional for long rollouts | Medium grid; export cost grows with step count. |
| `wave` | On (default) | Optional if exporting many steps | Ingress can be heavy when GRIB cache is cold. |
| `small_pretrained` | Off | Off | Fast IC and tiny NetCDF; overhead of threading not worth it. |

**When to turn overlap off:** debugging ingress, profiling serial prepare stages, or when IC is already a pre-built NetCDF and `load()` alone is enough.

**When to turn async export off:** single-step smoke tests, very short rollouts (`K=1–2`), or debugging export correctness.

**Recommended service path (0.1° example)**

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
    download=False,
)

batch = engine.prepare(request, rollout_steps=4)
paths = list(engine.rollout_and_export(batch, steps=4))
engine.release_gpu()
```

**Serial fallback (debug / baseline)**

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

**Configuration surface.** Key fields on `EngineConfig`: `variant`, `source`, `asset_root`, `checkpoint_path`, `inference_precision`, `cuda_graph`, `device`, `export_dir`, `allow_hub_download`, `gpu_guard`, `gpu_rollout_steps`, `overlap_ic_load`, `async_export`, `export_pool_size`, `export_max_inflight`, `export_use_egress_stream`. Inspect registered names with `DEFAULT_PRESETS.names()`.

**Utilities.** `ecmwf_credential_status()` reports ECMWF API readiness before MARS requests; `normalize_user_path()` and `AssetStore` constrain file access to allowed roots under `asset_root`.

## Testing notes

`test_aurora_small` compares FP64 forward outputs to Microsoft Hugging Face reference pickles. On recent PyTorch builds (for example 2.12.x), a small drift on a few surface variables can appear even with the official `microsoft-aurora` wheel. The test passes and emits a `UserWarning` when drift exceeds upstream tolerances. Use the [vanilla microsoft-aurora library](https://microsoft.github.io/aurora) to verify on your stack.

## License

This repository is licensed under the [MIT License](LICENSE).

Third-party components bundled in the library:

- `flash_aurora.aurora` is derived from [Microsoft Aurora](https://github.com/microsoft/aurora) (MIT). See [flash_aurora/aurora/LICENSE.txt](flash_aurora/aurora/LICENSE.txt) and [flash_aurora/aurora/NOTICE.md](flash_aurora/aurora/NOTICE.md).
- Some source files include additional notices (for example NVIDIA BSD-3-Clause in `flash_aurora/aurora/ops/cute/_dense_gemm_sm120.py`). See per-file headers.

## Reference

**Aurora model.** Bodnar et al., *A Foundation Model for the Earth System*, Nature (2025). [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y). Upstream documentation: [microsoft.github.io/aurora](https://microsoft.github.io/aurora).

**CUTLASS / CuTe DSL.** CuTe DSL window-attention and dense GEMM kernels under `flash_aurora/aurora/ops/cute/` adapt layout, TMA, and GEMM patterns from [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) examples (BSD-3-Clause; see file headers such as `ops/cute/_dense_gemm_sm120.py`). Runtime package: `nvidia-cutlass-dsl`.

**Flash Attention.** FMHA mainloop, online softmax, and dispatch structure follow [flash-attn](https://github.com/Dao-AILab/flash-attention) (`flash_attn/cute/`; Tri Dao).