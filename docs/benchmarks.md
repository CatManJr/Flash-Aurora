# Flash-Aurora benchmarks

This page collects the full latency, precision-drift, and window-attention microbenchmark tables for Flash-Aurora. Headline summary numbers and bar charts remain in the [project README](../README.md). Regenerate README figures with `uv run python benchmark/plot_readme_perf_charts.py`.

**Machine context.** Unless noted otherwise, end-to-end latency means a **single** `model.forward` (one rollout step / one lead), measured on NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch 2.12.1+cu130, CUDA 13.0, `CUTE_DSL_ARCH=sm_120a`, batch size 1, and cached ingress. Multi-step autoregressive rollouts are reported separately under [Distributed pipeline](#distributed-pipeline). Window-attention microbenchmarks also report RTX 4090 (`sm_89`). Distributed rollout tables cover 2x RTX 5090 and 2x RTX 4090.

**How to reproduce.** Commands live in [Reproducing the benchmarks](#reproducing-the-benchmarks) below and in scripts under [`../benchmark/`](../benchmark/). Install dependencies with `uv sync` from the repository root (see [Install](../README.md#install)).

## Window attention microbenchmarks

Measured with `../benchmark/bench_window_attn.py` (trimmed mean of 200 runs per shape). Kernel tensors use layout $(B, H, N, D_h)$: $B$ is the folded window batch ($B = B_{\mathrm{batch}} \cdot n_W$), $H$ is the head count, and $N$ is tokens per window ($N=144$ for window size $(2,6,12)$ on the default $0.25^{\circ}$ encoder). Tables below write $B$ as $B_{\mathrm{win}}$. SDPA baselines use PyTorch `scaled_dot_product_attention` with the same dtype as the CuTe path (BF16 for `BF16_MIXED`, FP32 for `TF32`).

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


The isolate-tiers tables were remeasured on 2026-09-29 with PyTorch **2.14.0+cu130**, warmup 5, and repeat 20. Speedup uses `lora_merged` on finetuned presets and forward latency on pretrained presets, each relative to `pytorch_backbone_fp32_encoder_decoder_fp32`. Source reports are the Group A `latency_<preset>.md` files. `small_pretrained` and the single-process tables were not remeasured.

**Single-process reference deflation.** The single-process tables later in this section are the 2026-06-23 PyTorch 2.12.1 run (warmup 2, repeat 5). On that harness, `era5_pretrained` FP32 was about $2128$ ms when isolated and about $1135$ ms in one process, while `bf16_mixed@fp32` stayed about $676$ ms. Do not compare those absolute times with the 2026-09-29 isolate tables.

**Finetuned models.** On finetuned models, encoder and decoder time plus backbone copy/cast overhead narrow the gap between custom tiers. LoRA eager adds a second low-rank GEMM; LoRA merge is independent of precision tier choice. For CAMS, `lora_merged` with `tf32@`* is the production latency path if strict `pm10` tolerance is required. Otherwise, `bf16_mixed@`* still keeps the balance of precision and speed.

#### Cold-start speedup (`--isolate-tiers`)

Generated 2026-09-29 on PyTorch 2.14.0+cu130 (warmup 5, repeat 20, one subprocess per tier). Tables now include `tf32x3@*` and `fp32@tf32`.

##### `era5_pretrained` ($721 \times 1440$)

| Tier | forward (ms) | vs PyTorch FP32 ref |
| --- | ---: | ---: |
| `bf16_mixed@fp32` | 827.8 | 2.61x |
| `bf16_mixed@tf32` | 677.6 | 3.19x |
| `tf32@fp32` | 1078.3 | 2.00x |
| `tf32@tf32` | 928.4 | 2.32x |
| `tf32x3@fp32` | 1095.1 | 1.97x |
| `tf32x3@tf32` | 944.5 | 2.29x |
| `fp32@fp32` | 1993.4 | 1.08x |
| `fp32@tf32` | 1841.1 | 1.17x |
| PyTorch autocast | 1000.7 | 2.16x |
| PyTorch FP32 ref | 2158.5 | base |

##### `aurora_v1p5` ($721 \times 1440$)

| Tier | forward (ms) | vs PyTorch FP32 ref |
| --- | ---: | ---: |
| `bf16_mixed@fp32` | 852.6 | 2.56x |
| `bf16_mixed@tf32` | 701.3 | 3.11x |
| `tf32@fp32` | 1102.9 | 1.98x |
| `tf32@tf32` | 951.8 | 2.29x |
| `tf32x3@fp32` | 1119.8 | 1.95x |
| `tf32x3@tf32` | 968.0 | 2.26x |
| `fp32@fp32` | 2032.0 | 1.07x |
| `fp32@tf32` | 1860.7 | 1.17x |
| PyTorch autocast | 1022.3 | 2.14x |
| PyTorch FP32 ref | 2183.0 | base |

##### `aurora_v1p5_ensemble` ($721 \times 1440$)

Generated 2026-09-29. Stochastic noise raises both mixed and FP32 latency. `bf16_mixed@fp32` is the slow end of the mixed tiers ($1148.6$ ms, $2.20\times$) because the ensemble MLP stays on TF32. A TF32 Perceiver (`bf16_mixed@tf32`) is $999.7$ ms ($2.53\times$).

| Tier | forward (ms) | vs PyTorch FP32 ref |
| --- | ---: | ---: |
| `bf16_mixed@fp32` | 1148.6 | 2.20x |
| `bf16_mixed@tf32` | 999.7 | 2.53x |
| `tf32@fp32` | 1263.1 | 2.00x |
| `tf32@tf32` | 1106.7 | 2.28x |
| `tf32x3@fp32` | 1275.2 | 1.98x |
| `tf32x3@tf32` | 1123.0 | 2.25x |
| `fp32@fp32` | 2326.9 | 1.09x |
| `fp32@tf32` | 2173.6 | 1.16x |
| PyTorch autocast | 1125.5 | 2.25x |
| PyTorch FP32 ref | 2527.1 | base |

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

Not remeasured on 2026-09-29. The numbers above are the 2026-06-23 PyTorch 2.12.1 run.


##### `hres_t0_finetuned` ($721 \times 1440$, LoRA)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| --- | ---: | ---: | ---: | ---: |
| `bf16_mixed@fp32` | 1035.0 | 796.0 | 1.30x | 2.68x |
| `bf16_mixed@tf32` | 883.4 | 640.4 | 1.38x | 3.33x |
| `tf32@fp32` | 1248.6 | 1017.5 | 1.23x | 2.09x |
| `tf32@tf32` | 1098.2 | 859.9 | 1.28x | 2.48x |
| `tf32x3@fp32` | 1264.3 | 1034.1 | 1.22x | 2.06x |
| `tf32x3@tf32` | 1113.3 | 875.2 | 1.27x | 2.44x |
| `fp32@fp32` | 2152.2 | 1957.4 | 1.10x | 1.09x |
| `fp32@tf32` | 2002.4 | 1796.0 | 1.11x | 1.19x |
| PyTorch autocast | 1099.6 | 966.0 | 1.14x | 2.21x |
| PyTorch FP32 ref | 2338.8 | 2131.4 | 1.10x | base |

##### `hres_0.1` ($1801 \times 3600$, LoRA)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| --- | ---: | ---: | ---: | ---: |
| `bf16_mixed@fp32` | 1036.5 | 819.4 | 1.27x | 2.50x |
| `bf16_mixed@tf32` | 897.4 | 673.6 | 1.33x | 3.04x |
| `tf32@fp32` | 1232.0 | 1019.2 | 1.21x | 2.01x |
| `tf32@tf32` | 1093.0 | 873.0 | 1.25x | 2.34x |
| `tf32x3@fp32` | 1246.8 | 1037.1 | 1.20x | 1.97x |
| `tf32x3@tf32` | 1107.6 | 889.8 | 1.24x | 2.30x |
| `fp32@fp32` | 2064.1 | 1888.4 | 1.09x | 1.08x |
| `fp32@tf32` | 1920.5 | 1734.1 | 1.11x | 1.18x |
| PyTorch autocast | 1096.0 | 976.8 | 1.12x | 2.09x |
| PyTorch FP32 ref | 2236.1 | 2044.5 | 1.09x | base |

##### `cams` ($451 \times 900$, LoRA)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| --- | ---: | ---: | ---: | ---: |
| `bf16_mixed@fp32` | 970.1 | 801.4 | 1.21x | 2.19x |
| `bf16_mixed@tf32` | 747.5 | 572.4 | 1.31x | 3.06x |
| `tf32@fp32` | 1123.1 | 958.2 | 1.17x | 1.83x |
| `tf32@tf32` | 900.3 | 726.6 | 1.24x | 2.41x |
| `tf32x3@fp32` | 1133.6 | 969.6 | 1.17x | 1.81x |
| `tf32x3@tf32` | 911.7 | 738.7 | 1.23x | 2.37x |
| `fp32@fp32` | 1773.6 | 1629.8 | 1.09x | 1.08x |
| `fp32@tf32` | 1551.0 | 1395.9 | 1.11x | 1.26x |
| PyTorch autocast | 1016.3 | 923.4 | 1.10x | 1.90x |
| PyTorch FP32 ref | 1909.9 | 1753.2 | 1.09x | base |

##### `tc_tracking` ($721 \times 1440$, LoRA)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
| --- | ---: | ---: | ---: | ---: |
| `bf16_mixed@fp32` | 1034.9 | 796.2 | 1.30x | 2.67x |
| `bf16_mixed@tf32` | 883.6 | 640.6 | 1.38x | 3.32x |
| `tf32@fp32` | 1249.1 | 1018.4 | 1.23x | 2.09x |
| `tf32@tf32` | 1098.6 | 860.4 | 1.28x | 2.47x |
| `tf32x3@fp32` | 1264.3 | 1034.7 | 1.22x | 2.05x |
| `tf32x3@tf32` | 1112.4 | 875.0 | 1.27x | 2.43x |
| `fp32@fp32` | 2152.8 | 1961.2 | 1.10x | 1.08x |
| `fp32@tf32` | 2007.2 | 1798.7 | 1.12x | 1.18x |
| PyTorch autocast | 1099.1 | 965.8 | 1.14x | 2.20x |
| PyTorch FP32 ref | 2335.0 | 2124.0 | 1.10x | base |

#### Non-isolated benchmarking artifact (`--no-isolate-tiers`)

These tables were not remeasured on 2026-09-29. They remain the 2026-06-23 PyTorch 2.12.1 run and are not comparable with the isolate-tiers times above.

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

### In-depth benchmarks

Isolate-tiers (warmup 2, repeat 5, one subprocess per tier) remains the headline **vs-FP32** and **vs-autocast** snapshot. A second harness, `../benchmark/bench_indepth_eval.py`, reports per-iteration CUDA-event mean$\pm$std ($n=12$), a CuTe-off row, CUDA-graph capture, encoder/backbone/decoder splits, one-forward VRAM, mean relative error on a second ERA5 initial condition (2026-07-01), and closed-loop drift via `../benchmark/bench_rollout_drift.py`.

Machine-readable reports: `../benchmark/indepth_eval_latest.md`, `../benchmark/indepth_eval_latest.json`, `../benchmark/rollout_drift_latest.md`.

**Do not mix harnesses.** Same-process autocast is $849.9\pm0.36$ ms versus isolate autocast $1004$ ms (cuDNN autotune leakage). Same-process unfused stage totals around $1137$ ms are likewise deflated; the isolated unfused split is encoder $90.3$ ms, backbone $1807.3$ ms ($83.5\%$), decoder $266.1$ ms. Custom-tier absolute latency is stable: `bf16_mixed@fp32` is $680.2\pm0.27$ ms (p95 $680.6$ ms) in the $n=12$ suite versus $676$ ms isolate.

**`fp32@fp32` is not CuTe.** `kernel_profile_for_backbone(FP32)` selects `fast_fp32` (Triton layout and AdaLN, `use_cute_window_attn=False`). Measured `fast_fp32` $2003.5\pm6.41$ ms versus `fp32@fp32` $2010.4\pm6.28$ ms. On the production mixed path, disabling CuTe (SDPA fallback) is $809.1\pm0.37$ ms versus $680.2\pm0.27$ ms ($1.19\times$). CUDA-graph backbone capture is $680.2\pm0.29$ ms at $26.9$ GiB: no latency win, $+0.6$ GiB. `compile_backbone=True` before `load_checkpoint` remaps keys to `backbone._orig_mod.*`. Compiling **after** load: unfused FP32 $995.7\pm0.47$ ms, autocast $768.1\pm0.25$ ms (Dynamo `recompile_limit` on `shift_size`), mixed $676.4\pm0.17$ ms (no win; custom kernels graph-break). Production mixed is $1.46\times$ versus compiled FP32 and $1.13\times$ versus this autocast+compile measurement; isolate $1.49\times$ versus **eager** autocast remains the comparison to PyTorch mixed precision without compile.

Mean relative error $\bar{e}_v=\mathrm{mean}(|y-\hat{y}|)/\mathrm{mean}(|\hat{y}|)$ versus the unfused FP32 twin (seed 42), not WeatherBench2:

| IC | tier | 2t | 10u | 10v | msl |
| --- | --- | ---: | ---: | ---: | ---: |
| 2023-01-01 | `bf16_mixed@fp32` | 3.85e-5 | 8.27e-4 | 1.01e-3 | 5.67e-6 |
| 2023-01-01 | `tf32@fp32` | 1.02e-5 | 3.62e-4 | 4.53e-4 | 1.84e-6 |
| 2023-01-01 | autocast | 4.36e-5 | 1.42e-3 | 1.80e-3 | 7.40e-6 |
| 2026-07-01 | `bf16_mixed@fp32` | 3.92e-5 | 8.25e-4 | 9.63e-4 | 5.53e-6 |
| 2026-07-01 | `tf32@fp32` | 3.05e-6 | 1.60e-4 | 2.01e-4 | 7.37e-7 |
| 2026-07-01 | autocast | 4.40e-5 | 1.39e-3 | 1.70e-3 | 7.41e-6 |

One-step weather channels pass the golden mean-rel test on both dates ($0/9$ fail). On `era5_pretrained`, `bf16_mixed@fp32` stays within those tolerances through step 12 ($72$ h) and first fails on winds at step 13. `tf32@fp32` first fails at step 17. Autocast first fails at step 8. On CAMS, `tf32@fp32` passes all 22 variables through $96$ h. Peak one-forward VRAM at `bf16_mixed@fp32` is $26.6$ GiB (`era5_pretrained`), $24.3$ (`aurora_v1p5`), $27.7$ (`hres_t0_finetuned`), $33.4$ (`hres_0.1`), $24.0$ (`cams`). A 40-step ERA5 rollout stays at $26.6$ GiB for FP32, mixed, TF32, and autocast.

```bash
export AURORA_ASSET_ROOT=/path/to/aurora
export CUTE_DSL_ARCH=sm_120a
uv run --python 3.12 python benchmark/bench_indepth_eval.py
uv run --python 3.12 python benchmark/bench_rollout_drift.py \
  --presets era5_pretrained cams --steps 40 --cams-steps 16
```

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

#### `aurora_v1p5_ensemble` ($721 \times 1440$, 31 vars)

Generated 2026-08-14 (`../benchmark/precision_aurora_v1p5_ensemble_latest.md`). Stochastic Ensemble. Core meteorology still passes on mixed; the two scaled precipitation/snow fields do not.


| Tier              | `2t`     | `10u`    | `10v`    | `msl`    | pass  |
| ----------------- | -------- | -------- | -------- | -------- | ----- |
| `bf16_mixed@fp32` | 2.89e-05 | 1.63e-03 | 2.12e-03 | 5.68e-06 | 29/31 |
| `tf32@fp32`       | 1.15e-05 | 6.85e-04 | 9.16e-04 | 2.35e-06 | 31/31 |
| `fp32@fp32`       | 1.12e-05 | 6.46e-04 | 8.69e-04 | 2.24e-06 | 31/31 |
| PyTorch autocast  | 4.02e-05 | 2.68e-03 | 3.50e-03 | 9.04e-06 | 26/31 |


`bf16_mixed@fp32` fails `scaled_tp_1h` ($7.39\times10^{-3}$) and `scaled_sf_1h` ($6.19\times10^{-3}$). Autocast additionally fails `lcc`, `mcc`, `hcc`. `tf32@fp32` is the numerically conservative path for Ensemble.

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

Single-process only (one Python interpreter, not `torchrun`). Pass a `DistributedConfig` with two or more CUDA devices. On 32 GiB cards, `era5_pretrained` does **not** fit a single GPU. Aurora 1.5 (`aurora_v1p5`) supports the same pipeline on **standard timestep** autoregressive steps (`lead_times` = model timestep hours). Fine-lead substeps inside `distributed_rollout` are not wired yet. CUDA graphs remain unsupported for Aurora 1.5 and stay forced off when distributed is enabled.

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

