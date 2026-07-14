# Flash-Aurora benchmarks

This page collects the full latency, precision-drift, and window-attention microbenchmark tables for Flash-Aurora. Headline summary numbers and bar charts remain in the [project README](../README.md). Regenerate README figures with `uv run python benchmark/plot_readme_perf_charts.py`.

**Machine context.** Unless noted otherwise, end-to-end latency means a **single** `model.forward` (one rollout step / one lead), measured on NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch 2.12.1+cu130, CUDA 13.0, `CUTE_DSL_ARCH=sm_120a`, batch size 1, and cached ingress. Multi-step autoregressive rollouts are reported separately under [Distributed pipeline](#distributed-pipeline). Window-attention microbenchmarks also report RTX 4090 (`sm_89`). Distributed rollout tables cover 2x RTX 5090 and 2x RTX 4090.

**How to reproduce.** Commands live in [Reproducing the benchmarks](#reproducing-the-benchmarks) below and in scripts under [`../benchmark/`](../benchmark/). Install dependencies with `uv sync` from the repository root (see [Install](../README.md#install)).

## Window attention microbenchmarks

Measured with `../benchmark/bench_window_attn.py` (trimmed mean of 200 runs per shape). Kernel tensors use layout $(B, H, N, D_h)$: $B$ is the folded window batch ($B = B_{\mathrm{batch}} \cdot n_W$), $H$ is the head count, and $N$ is tokens per window ($N=144$ for window size $(2,6,12)$ on the default $0.25^{\circ}$ encoder). Tables below write $B$ as $B_{\mathrm{win}}$. SDPA baselines use PyTorch `scaled_dot_product_attention` with the same dtype as the CuTe path (BF16 for `BF16_MIXED`, FP32 for `TF32_ACC_FP32`).

### NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120a)

PyTorch **2.12.1**, `CUTE_DSL_ARCH=sm_120a`. Full report: `../benchmark/window_attn_latest.txt`.

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


| Mode          | Stage 1 ($B_{\mathrm{win}}=1800$, $H=8$) | Speedup vs SDPA |
| ------------- | ---------------------------------------- | --------------- |
| BF16 CuTe DSL | 0.829 ms vs 1.014 ms                     | 1.22x           |
| TF32 CuTe DSL | 1.906 ms vs 3.221 ms                     | 1.69x           |


### NVIDIA GeForce RTX 4090 (sm_89)

PyTorch **2.12.1**, `CUTE_DSL_ARCH=sm_89`.

**0.25-degree ERA5 encoder stages** (unmasked, $N=144$ tokens per window):


| Stage | $B_{\mathrm{win}}$ | $H$ | BF16 CuTe DSL (ms) | BF16 SDPA (ms) | Speedup |
| ----- | ------------------ | --- | ------------------ | -------------- | ------- |
| 1     | 1800               | 8   | 1.157              | 1.584          | 1.37x   |
| 2     | 450                | 16  | 0.589              | 0.804          | 1.36x   |
| 3     | 128                | 32  | 0.345              | 0.470          | 1.36x   |



| Stage | $B_{\mathrm{win}}$ | $H$ | TF32 CuTe DSL (ms) | FP32 SDPA (ms) | Speedup |
| ----- | ------------------ | --- | ------------------ | -------------- | ------- |
| 1     | 1800               | 8   | 2.443              | 5.491          | 2.25x   |
| 2     | 450                | 16  | 1.239              | 2.727          | 2.20x   |
| 3     | 128                | 32  | 0.713              | 1.598          | 2.24x   |


**Shifted-window mask** (Swin relative position bias $-100$, stage 1):


| Mode          | Stage 1 ($B_{\mathrm{win}}=1800$, $H=8$) | Speedup vs SDPA |
| ------------- | ---------------------------------------- | --------------- |
| BF16 CuTe DSL | 1.187 ms vs 1.401 ms                     | 1.18x           |
| TF32 CuTe DSL | 2.642 ms vs 5.997 ms                     | 2.27x           |


On RTX 4090, PyTorch SDPA autoselect is slower than the memory-efficient backend on these shapes. Forced `mem_eff` SDPA is within a few percent of CuTe BF16 (for example enc1: 1.157 ms CuTe vs 1.220 ms mem_eff). The larger speedups in the tables above are relative to default SDPA dispatch. CuTe absolute latency is higher on sm_89 than on sm_120 (enc1 BF16: 1.16 ms vs 0.73 ms) because tile sizes and memory bandwidth differ, but the kernel still wins on production $N{=}144$ shapes.

Production inference on the default $0.25^{\circ}$ grid uses $N=144$ windows per stage. BF16 CuTe DSL attention requires at least 32 tokens per window; on coarser downsampled stages with smaller $N$, use `tf32` or PyTorch SDPA.

## End to End Benchmarks

Benchmarks were run on NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch **2.12.1+cu130**, CUDA 13.0, `CUTE_DSL_ARCH=sm_120a`, batch size 1, and cached ingress. Custom tiers include Triton layout and AdaLN fusion. The PyTorch FP32 reference (`pytorch_backbone_fp32_encoder_decoder_fp32`) disables Triton and CuTe DSL. Finetuned presets report `lora_eager` and `lora_merged`; pretrained presets report forward latency.

The `wave` preset is omitted from benchmark tables. It requires MARS wave GRIB from the ECMWF archive; personal API accounts typically lack MARS access. See `example_wave.ipynb` for manual cache setup.

`bf16@`* is excluded from latency tables because it does not improve speed over `bf16_mixed@`* and has larger drift.

### Forward latency (all presets)

Two harness modes are reported.


| Mode           | Flag                        | Use                                                                                                                                                                                                                              |
| -------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fair speedup   | `--isolate-tiers` (default) | Each preset-by-tier pair in a fresh subprocess; use for vs-ref ratios and headline numbers.                                                                                                                                      |
| Single-process | `--no-isolate-tiers`        | All tiers in one process; illustrates how cuDNN autotune warms across tiers and can deflate the PyTorch FP32 reference when it is timed after custom kernels. Custom-tier absolute latency is stable; only vs ref is misleading. |


Both modes use warmup $2$ and repeat $5$. Speedup uses `lora_merged` on finetuned presets and forward latency on pretrained presets, each relative to `pytorch_backbone_fp32_encoder_decoder_fp32`. Machine-readable reports are `../benchmark/latency_all_isolated.md` and `../benchmark/latency_all_single_process.md`.

**Single-process reference deflation.** On `era5_pretrained`, the PyTorch FP32 reference is ${\sim}2128$ ms when isolated but ${\sim}1135$ ms when timed after Triton/CuTe DSL tiers in the same process. `bf16_mixed@fp32` remains ${\sim}676$ ms in both runs. The speedup ratio changes even though custom latency is unchanged.

**Finetuned models.** On finetuned models, encoder and decoder time plus backbone copy/cast overhead narrow the gap between custom tiers. LoRA eager adds a second low-rank GEMM; LoRA merge is independent of precision tier choice. For CAMS, `lora_merged` with `tf32@`* is the production latency path if strict `pm10` tolerance is required. Otherwise, `bf16_mixed@`* still keeps the balance of precision and speed.

#### Cold-start speedup (`--isolate-tiers`)

Generated 2026-06-23; full tables: `../benchmark/latency_all_isolated.md`.

##### `era5_pretrained` ($721 \times 1440$)


| Tier              | forward (ms) | vs PyTorch FP32 ref |
| ----------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 676.4        | 3.15x               |
| `bf16_mixed@tf32` | 676.8        | 3.14x               |
| `tf32@fp32`       | 1077.5       | 1.98x               |
| `tf32@tf32`       | 919.2        | 2.32x               |
| `fp32@fp32`       | 1945.0       | 1.09x               |
| PyTorch autocast  | 1004.4       | 2.12x               |
| PyTorch FP32 ref  | 2128.2       | base                |


##### `aurora_v1p5` ($721 \times 1440$)

Generated 2026-07-14 (`../benchmark/latency_aurora_v1p5_latest.md`); same machine and isolate-tiers harness as above. Uses the shared Swin Triton/CuTe + `inference_precision` path.


| Tier              | forward (ms) | vs PyTorch FP32 ref |
| ----------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 700.8        | 3.12x               |
| `bf16_mixed@tf32` | 702.4        | 3.11x               |
| `tf32@fp32`       | 1109.3       | 1.97x               |
| `tf32@tf32`       | 946.0        | 2.31x               |
| `fp32@fp32`       | 2005.6       | 1.09x               |
| PyTorch autocast  | 1034.5       | 2.11x               |
| PyTorch FP32 ref  | 2185.8       | base                |


##### `small_pretrained` ($400 \times 800$)


| Tier              | forward (ms) | vs PyTorch FP32 ref |
| ----------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 42.4         | 2.40x               |
| `bf16_mixed@tf32` | 42.4         | 2.40x               |
| `tf32@fp32`       | 64.1         | 1.59x               |
| `tf32@tf32`       | 57.3         | 1.78x               |
| `fp32@fp32`       | 94.7         | 1.08x               |
| PyTorch autocast  | 56.3         | 1.81x               |
| PyTorch FP32 ref  | 101.9        | base                |


##### `hres_t0_finetuned` ($721 \times 1440$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 881.7           | 638.7            | 1.38x        | 3.23x               |
| `bf16_mixed@tf32` | 881.6           | 638.4            | 1.38x        | 3.23x               |
| `tf32@fp32`       | 1249.6          | 1006.3           | 1.24x        | 2.05x               |
| `tf32@tf32`       | 1091.5          | 846.5            | 1.29x        | 2.44x               |
| `fp32@fp32`       | 2115.5          | 1890.4           | 1.12x        | 1.09x               |
| PyTorch autocast  | 1104.4          | 967.7            | 1.14x        | 2.13x               |
| PyTorch FP32 ref  | 2307.9          | 2061.9           | 1.12x        | base                |


##### `hres_0.1` ($1801 \times 3600$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 898.0           | 672.0            | 1.34x        | 2.97x               |
| `bf16_mixed@tf32` | 898.6           | 672.4            | 1.34x        | 2.97x               |
| `tf32@fp32`       | 1247.7          | 1019.9           | 1.22x        | 1.96x               |
| `tf32@tf32`       | 1091.1          | 861.3            | 1.27x        | 2.32x               |
| `fp32@fp32`       | 2051.0          | 1838.0           | 1.12x        | 1.09x               |
| PyTorch autocast  | 1112.1          | 986.2            | 1.13x        | 2.02x               |
| PyTorch FP32 ref  | 2227.5          | 1994.6           | 1.12x        | base                |


##### `cams` ($451 \times 900$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 747.3           | 571.0            | 1.31x        | 2.96x               |
| `bf16_mixed@tf32` | 747.5           | 571.9            | 1.31x        | 2.96x               |
| `tf32@fp32`       | 1096.1          | 916.5            | 1.20x        | 1.85x               |
| `tf32@tf32`       | 898.1           | 718.3            | 1.25x        | 2.35x               |
| `fp32@fp32`       | 1734.6          | 1562.3           | 1.11x        | 1.08x               |
| PyTorch autocast  | 985.7           | 888.6            | 1.11x        | 1.90x               |
| PyTorch FP32 ref  | 1874.5          | 1691.6           | 1.11x        | base                |


##### `tc_tracking` ($721 \times 1440$, LoRA)


| Tier              | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| ----------------- | --------------- | ---------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 881.7           | 638.5            | 1.38x        | 3.23x               |
| `bf16_mixed@tf32` | 881.7           | 638.3            | 1.38x        | 3.23x               |
| `tf32@fp32`       | 1249.6          | 1006.1           | 1.24x        | 2.05x               |
| `tf32@tf32`       | 1092.0          | 847.0            | 1.29x        | 2.43x               |
| `fp32@fp32`       | 2115.1          | 1890.9           | 1.12x        | 1.09x               |
| PyTorch autocast  | 1104.2          | 967.4            | 1.14x        | 2.13x               |
| PyTorch FP32 ref  | 2307.9          | 2059.9           | 1.12x        | base                |


#### Non-isolated benchmarking artifact (`--no-isolate-tiers`)

Custom tiers run first in one cold-start and the PyTorch FP32 reference is timed last. cuDNN state from earlier tiers is already warm, so vs-ref speedup is understated. Custom-tier absolute latency matches the isolated run.

##### `era5_pretrained` ($721 \times 1440$)


| Tier              | forward (ms) | vs PyTorch FP32 ref |
| ----------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 676.7        | 1.68x               |
| `bf16_mixed@tf32` | 677.1        | 1.68x               |
| `tf32@fp32`       | 920.7        | 1.23x               |
| `tf32@tf32`       | 921.3        | 1.23x               |
| `fp32@fp32`       | 944.9        | 1.20x               |
| PyTorch autocast  | 846.5        | 1.34x               |
| PyTorch FP32 ref  | 1135.5       | base                |


##### `small_pretrained` ($400 \times 800$)


| Tier              | forward (ms) | vs PyTorch FP32 ref |
| ----------------- | ------------ | ------------------- |
| `bf16_mixed@fp32` | 41.9         | 1.59x               |
| `bf16_mixed@tf32` | 41.8         | 1.59x               |
| `tf32@fp32`       | 57.7         | 1.15x               |
| `tf32@tf32`       | 57.1         | 1.17x               |
| `fp32@fp32`       | 59.6         | 1.12x               |
| PyTorch autocast  | 49.5         | 1.34x               |
| PyTorch FP32 ref  | 66.5         | base                |


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
| `bf16_mixed@fp32` | 747.4           | 571.4            | 1.31x        | 1.53x               |
| `bf16_mixed@tf32` | 747.6           | 571.2            | 1.31x        | 1.53x               |
| `tf32@fp32`       | 897.9           | 719.0            | 1.25x        | 1.22x               |
| `tf32@tf32`       | 898.0           | 719.8            | 1.25x        | 1.21x               |
| `fp32@fp32`       | 915.3           | 738.8            | 1.24x        | 1.18x               |
| PyTorch autocast  | 788.0           | 690.4            | 1.14x        | 1.27x               |
| PyTorch FP32 ref  | 1054.5          | 874.5            | 1.21x        | base                |


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


Recommended production tiers are `bf16_mixed@fp32` or `bf16_mixed@tf32` for weather presets with `lora_merged`. For CAMS, use `lora_merged` with `bf16_mixed@`* for speed, or `tf32@fp32` when strict `pm10` tolerance is required.

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

Measured with `../benchmark/bench_aurora_precision_all.py`, seed 42, baseline `pytorch_backbone_fp32_encoder_decoder_fp32`. Entries are $\bar{e}_v$; values above $\tau_v$ are **bold**.

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

#### `aurora_v1p5` ($721 \times 1440$, 31 vars)

Generated 2026-07-14 (`../benchmark/precision_aurora_v1p5_latest.md`). Table shows the nine core meteorology channels (same set as `era5_pretrained`); the remaining 22 extended surface fields also pass on every recommended tier (`bf16_mixed@*`, `tf32@*`, `fp32@fp32`).


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | `t`      | `u`      | `v`      | `q`      | `z`      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `bf16_mixed@fp32` | 2.46e-05 | 1.12e-03 | 1.45e-03 | 4.75e-06 | 1.23e-05 | 5.92e-04 | 1.13e-03 | 4.79e-04 | 5.38e-06 |
| `bf16_mixed@tf32` | 2.49e-05 | 1.19e-03 | 1.55e-03 | 4.83e-06 | 1.40e-05 | 7.18e-04 | 1.39e-03 | 5.23e-04 | 5.63e-06 |
| `tf32@fp32`       | 9.97e-06 | 4.93e-04 | 6.44e-04 | 2.05e-06 | 8.49e-06 | 4.32e-04 | 8.37e-04 | 2.67e-04 | 4.48e-06 |
| `tf32@tf32`       | 9.97e-06 | 4.93e-04 | 6.44e-04 | 2.05e-06 | 8.49e-06 | 4.32e-04 | 8.37e-04 | 2.67e-04 | 4.48e-06 |
| `fp32@fp32`       | 9.84e-06 | 4.73e-04 | 6.17e-04 | 1.97e-06 | 8.32e-06 | 4.22e-04 | 8.17e-04 | 2.60e-04 | 4.43e-06 |
| PyTorch autocast  | 3.38e-05 | 2.02e-03 | 2.66e-03 | 8.12e-06 | 2.19e-05 | 1.19e-03 | 2.30e-03 | 7.73e-04 | 8.62e-06 |


Recommended tiers pass **31/31** variables. `bf16@fp32` fails `mcc`, `hcc`, `scaled_tp_1h`, `scaled_sf_1h` (excluded from latency tables). PyTorch autocast fails `hcc`, `scaled_tp_1h`, `scaled_sf_1h`.

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


On CAMS, `bf16_mixed@`* exceeds $\tau_{\mathrm{pm10}} = 5\times10^{-3}$ by about 4% (meteorological channels remain within tolerance). `tf32@`* and `fp32@fp32` pass all 22 variables.

**Excluded tier `bf16@fp32` (not recommended):** on CAMS, $\bar{e}*{\mathrm{pm2p5}} = 5.32\times10^{-3}$ and $\bar{e}*{\mathrm{pm10}} = 6.35\times10^{-3}$; on `era5_pretrained`, $\bar{e}_{10u} = 1.53\times10^{-3}$ and $\bar{e}_v = 1.96\times10^{-3}$ (within $\tau$ but $2$--$3\times$ higher than `bf16_mixed@fp32`). Latency matches `bf16_mixed@`* on pretrained ERA5 (680.5 ms vs 681.9 ms) and offers no benefit.

### Reproducing the benchmarks

Commands below assume the repository root as the working directory. Script and report paths in prose use `../benchmark/` relative to this file; shell commands use `benchmark/` from the repo root. All commands assume PyTorch **2.12.1** from `uv.lock`, CUDA 13.0, and `CUTE_DSL_ARCH=sm_120a` on Blackwell.

**Prerequisites** (first run on a fresh machine):

```bash
export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora   # data disk; any absolute path is fine
export CDSAPI_KEY='<api_key>'  # Copernicus CDS, https://cds.climate.copernicus.eu/how-to-api
```

Set `CDSAPI_KEY` before downloading ERA5 ingress (presets `era5_pretrained`, `small_pretrained`, and ERA5 static for `hres_t0_finetuned`). Use the API key from the CDS API page (no legacy `UID:` prefix). This skips the interactive CDS prompt, which is unreliable in browser terminals (AutoDL, Cursor) where paste at a password prompt often fails. Equivalent: a `~/.cdsapirc` file with `url:` and `key:` lines.

Checkpoints and HF static pickles download on first run. When `huggingface.co` is unreachable, the engine uses `hf-mirror.com` automatically; `../benchmark/bench_aurora_pretrained.py` enables the mirror by default.

**End-to-end latency** (all presets except `wave`, every tier, `lora_eager` vs `lora_merged` where applicable):

Fair speedup (default):

```bash
export CDSAPI_KEY='<api_key>'
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_latency_all.py \
  --asset-root "$AURORA_ASSET_ROOT" --warmup 2 --repeat 5 \
  --isolate-tiers --report-out benchmark/latency_all_isolated.md
```

Single-process artifact (cuDNN cross-tier warmup demo; ref timed after custom tiers):

```bash
export CDSAPI_KEY='<api_key>'
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_latency_all.py \
  --asset-root "$AURORA_ASSET_ROOT" --warmup 2 --repeat 5 \
  --no-isolate-tiers --defer-ref \
  --report-out benchmark/latency_all_single_process.md
```

`--defer-ref` times the PyTorch FP32 reference after all custom tiers in the same process (reproduces the understated vs-ref column). Omit `--defer-ref` for a quicker single-process run with the reference timed first.

Finetuned-only shortcut (delegates to the same harness): `../benchmark/bench_aurora_finetuned_lora.py`.

**Window attention** (CuTe DSL vs PyTorch SDPA micro-benchmark):

```bash
CUTE_DSL_ARCH=sm_120a BENCH_MEASURED=200 uv run python benchmark/bench_window_attn.py
CUTE_DSL_ARCH=sm_89 BENCH_MEASURED=200 uv run python benchmark/bench_window_attn.py
```

On sm_89 the script stops at the optional N=576 streaming micro shape (TMA needs sm_90+). ERA5 and checkpoint shape tables complete normally.

**Precision drift** (seed 42, `lora_merged` on finetuned models):

```bash
export CDSAPI_KEY='<api_key>'
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_precision_all.py \
  --asset-root "$AURORA_ASSET_ROOT" --seed 42
```

Report: `../benchmark/precision_all_seed42.md`.

**Stage timing** (encoder / backbone / decoder breakdown; optional cast profiling):

```bash
export CDSAPI_KEY='<api_key>'
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_finetuned_stage_timing.py \
  --asset-root "$AURORA_ASSET_ROOT" --profile-casts
```

**ERA5 pretrained** (real CDS ingress + all precision tiers; subset of `bench_aurora_latency_all.py`):

```bash
export CDSAPI_KEY='<api_key>'
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_pretrained.py \
  --asset-root "$AURORA_ASSET_ROOT" --suite legacy --warmup 1 --repeat 3
```

Use `--skip-download` when checkpoint and `era5/` cache are already present. Use `--no-prompt` only if credentials are preconfigured and you want to fail fast instead of prompting.

### Distributed pipeline

Encoder, backbone, and spatial decoder can run on separate GPUs.

Single-process only (one Python interpreter, not `torchrun`). Pass a `DistributedConfig` with two or more CUDA devices. On 32 GiB cards, `era5_pretrained` does **not** fit a single GPU.

#### Pipeline placement (2x RTX 5090, `era5_pretrained`)

The planner in `plan.py` assigns stages to minimize peak VRAM per device. For the default 2-GPU layout:


| Device   | Stages                  | Role                                                                                                                                    |
| -------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `cuda:0` | Encoder + decoder west  | Runs the Perceiver encoder; holds a **replica** of the decoder for the western half of patch columns (`decoder_spatial_parallel=True`). |
| `cuda:1` | Backbone + decoder east | Runs the Swin backbone (about 63% of forward time) and the primary decoder for the eastern patch columns.                               |


Each autoregressive step runs encoder, then backbone, then decoder. Spatial decoder split does not change the math. Backbone tokens are split along longitude in patch space, each half is decoded independently, and surface and atmos fields are concatenated along width. On `era5_pretrained`, peak decoder VRAM drops from about 28 GiB on one card to about 14 GiB per card. On `hres_0.1` ($1801 \times 3600$) both cards use about 23-24 GiB peak in our 5090 runs. Numerical drift stays within bf16 noise.

With three or more GPUs, encoder, backbone, and decoder can each occupy a dedicated device without a spatial split.

#### Preset coverage

All Aurora presets share the same encoder / backbone / decoder layout. Distributed mode does not change checkpoint loading or the forward math. It only chooses device placement. The VRAM planner in `plan.py` picks a layout from `ModelVariantSpec` and `DistributedConfig`. Small models that fit one GPU can still use distributed mode with `force=True`, but the default path is single-GPU.

#### Multi-step rollout

`rollout_and_export()` calls `rollout_stream()` for the distributed forward path, then hands each prediction to the egress layer. With `async_export=True` (default), GPU-to-CPU offload and NetCDF writes run on a background thread while the next autoregressive step advances on the GPUs. On `era5_pretrained`, `cuda:0` is lightly loaded during backbone (about 8% of forward time vs about 63% on `cuda:1`), so export can overlap forward work without contending for the same SMs. On `hres_0.1`, the grid is $2.5\times$ wider and $2.5\times$ taller than ERA5. Each export step moves more data and both devices stay busier, so per-step latency is export-bound (about 3.6 s vs about 1.3 s on `era5_pretrained` in our 5090 runs).

#### 4-step rollout benchmark

`bf16_mixed@fp32`, warmup 1, repeat 3. Each mode runs in a fresh subprocess so JIT and cuDNN state do not carry between modes.

##### 2x NVIDIA GeForce RTX 5090 (32 GiB)

Host: **50 vCPU** Intel Xeon Platinum 8470Q, **180 GiB** system memory. 2x NVIDIA GeForce RTX 5090 (32 GiB). PyTorch **2.12.1**, `CUTE_DSL_ARCH=sm_120a`, `--max-vram-gib 32`.

**era5_pretrained** ($721 \times 1440$)


| mode   | total (ms) | per step (ms) | peak alloc (GiB)         |
| ------ | ---------- | ------------- | ------------------------ |
| `2gpu` | 5278       | 1319          | cuda:0=12.9, cuda:1=18.2 |


Utilization plot: [era5_pretrained](image/distributed_rollout_utilization_5090_era5_pretrained_2gpu.png).

**hres_0.1** ($1801 \times 3600$, AuroraHighRes LoRA merged)


| mode   | total (ms) | per step (ms) | peak alloc (GiB)         |
| ------ | ---------- | ------------- | ------------------------ |
| `2gpu` | 14502      | 3626          | cuda:0=23.6, cuda:1=22.9 |


Utilization plot: [hres_0.1](image/distributed_rollout_utilization_5090_hres_0.1_2gpu.png).

##### 2x NVIDIA GeForce RTX 4090 (24 GiB)

Host: **32 vCPU** AMD EPYC 9654 96-Core Processor, **120 GiB** system memory. PyTorch **2.12.1**, `CUTE_DSL_ARCH=sm_89`, `--max-vram-gib 24 --force`.

**era5_pretrained** ($721 \times 1440$)


| mode   | total (ms) | per step (ms) | peak alloc (GiB)         |
| ------ | ---------- | ------------- | ------------------------ |
| `2gpu` | 8790       | 2198          | cuda:0=12.6, cuda:1=17.8 |


Utilization plot: [era5_pretrained](image/distributed_rollout_utilization_4090_era5_pretrained_2gpu.png).

**hres_0.1** ($1801 \times 3600$, AuroraHighRes LoRA merged)


| mode   | total (ms) | per step (ms) | peak alloc (GiB)         |
| ------ | ---------- | ------------- | ------------------------ |
| `2gpu` | 13901      | 3475          | cuda:0=20.4, cuda:1=22.6 |


Utilization plot: [hres_0.1](image/distributed_rollout_utilization_4090_hres_0.1_2gpu.png).

```bash
export AURORA_ASSET_ROOT=/path/to/aurora
export AURORA_ROLLOUT_TMP=/path/on/data-disk/rollout_tmp   # keep NetCDF off system disk

# Single forward: stage timing + per-GPU memory profile
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_pipeline_profile.py \
  --preset era5_pretrained --asset-root "$AURORA_ASSET_ROOT" \
  --inference-precision bf16_mixed@fp32 --skip-download --force --warmup 1 --repeat 5

# Multi-step rollout + utilization figures (5090)
CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_distributed_rollout.py \
  --preset era5_pretrained hres_0.1 --inference-precision bf16_mixed@fp32 \
  --steps 4 --skip-download --force --warmup 1 --repeat 3 --no-prompt \
  --modes 2gpu \
  --plot-utilization docs/image/distributed_rollout_utilization_5090.png

# Same harness on 2x RTX 4090 (24 GiB); set max VRAM to match card size
CUTE_DSL_ARCH=sm_89 uv run python benchmark/bench_distributed_rollout.py \
  --preset era5_pretrained hres_0.1 --inference-precision bf16_mixed@fp32 \
  --steps 4 --skip-download --force --warmup 1 --repeat 3 --no-prompt \
  --max-vram-gib 24 --modes 2gpu \
  --plot-utilization docs/image/distributed_rollout_utilization_4090.png

# Programmatic use
python - <<'PY'
from pathlib import Path
from flash_aurora.engine.core.engine import AuroraEngine
from flash_aurora.engine.distributed import DistributedConfig

engine = AuroraEngine.from_preset(
    "era5_pretrained",
    asset_root=Path("/root/autodl-tmp/aurora"),
    inference_precision="bf16_mixed@fp32",
    distributed=DistributedConfig(
        devices=("cuda:0", "cuda:1"),
        max_vram_gib_per_device=32.0,
        force=True,
        decoder_spatial_parallel=True,
    ),
)
engine.load()
print(engine.distributed_status())
PY
```

Implementation lives under `flash_aurora/engine/distributed/` (see also [Distributed pipeline](../README.md#distributed-pipeline) in the README): `plan.py` (VRAM planner), `pipeline.py` and `rollout_pipeline.py` (pipeline forward and `distributed_rollout`), `decoder_spatial.py` (west/east split), `config.py`.

