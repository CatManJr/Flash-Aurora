# In-depth benchmarks

- Generated: 2026-08-14T11:33:16
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`, CUDA 13.0, `CUTE_DSL_ARCH=sm_120a`
- Seed: 42
- Scripts: `benchmark/bench_indepth_eval.py`, `benchmark/bench_indepth_fillin.py`, isolated VRAM/stages helper, `benchmark/bench_rollout_drift.py`

Headline **vs-FP32** and **vs-autocast** ratios remain the isolate-tiers snapshot in `docs/benchmarks.md` (warmup 2, repeat 5, one subprocess per tier). This report is the variance, kernel, compiler, stage, VRAM, and mean-rel suite. Same-process autocast (849.9 ms) is **deflated** relative to isolate autocast (1004 ms) and is not a vs-autocast baseline. Custom-tier absolute latency is stable (`bf16_mixed@fp32` 680.2 ms here vs 676 ms isolate).

`fp32@fp32` is **not** a CuTe path. `kernel_profile_for_backbone(FP32)` selects `fast_fp32`: Triton layout + AdaLN, `use_cute_window_attn=False`. Measured `fast_fp32` 2003.5 ms vs `fp32@fp32` 2010.4 ms (same within noise).

## One-step latency with variance (`era5_pretrained`, same process, n=12)

| config | mean±std (ms) | p95 | peak GiB |
| --- | --- | ---: | ---: |
| unfused FP32 | 2169.0 ± 2.54 | 2173.0 | 26.3 |
| `fast_fp32` (Triton + SDPA, no CuTe) | 2003.5 ± 6.41 | 2011.5 | 26.3 |
| `fp32@fp32` (Triton, no CuTe) | 2010.4 ± 6.28 | 2019.6 | 26.3 |
| `tf32@fp32` | 1093.4 ± 1.20 | 1094.7 | 26.3 |
| PyTorch BF16 autocast (**deflated**) | 849.9 ± 0.36 | 850.4 | 26.3 |
| `bf16_mixed@fp32` with SDPA (CuTe off) | 809.1 ± 0.37 | 809.6 | 26.3 |
| `bf16_mixed@fp32` | 680.2 ± 0.27 | 680.6 | 26.3 |
| CUDA graph, backbone scope | 680.2 ± 0.29 | 680.6 | 26.9 |
| `torch.compile` with `compile_backbone=True` before load | FAIL: keys become `backbone._orig_mod.*` | — | — |
| `torch.compile` after load, unfused FP32 | 995.7 ± 0.47 | 996.2 | 26.3 |
| `torch.compile` after load, autocast | 768.1 ± 0.25 | 768.3 | 26.3 |
| `torch.compile` after load, mixed | 676.4 ± 0.17 | 676.6 | 26.3 |

CuTe on the production mixed path is $809.1/680.2 = 1.19\times$. Isolated attention microbench on the same SKU is only $1.07\times$ because that microbench is not the fused QKV chain. Post-load compile of unfused FP32 is $2.18\times$ versus same-process eager FP32. Production mixed is $1.46\times$ versus compiled FP32 and $1.13\times$ versus autocast+compile (Dynamo hit `recompile_limit` on `shift_size`). Mixed+compile does not beat eager mixed.

## Isolate-tiers production snapshot (headline vs-ref)

From `benchmark/latency_all_isolated.md`, `era5_pretrained`:

| Path | Latency (ms) | vs unfused FP32 | vs autocast |
| --- | ---: | ---: | ---: |
| Unfused PyTorch FP32 | 2128 | $1.00\times$ | — |
| Fused FP32 (`fp32@fp32`, Triton, no CuTe) | 1945 | $1.09\times$ | — |
| Fused TF32 (`tf32@fp32`) | 1078 | $1.98\times$ | $0.93\times$ |
| PyTorch BF16 autocast | 1004 | $2.12\times$ | $1.00\times$ |
| `bf16_mixed@fp32` | 676 | $3.15\times$ | $1.49\times$ |

## Encoder / backbone / decoder

Isolated unfused FP32 (fresh process): encoder 90.3 ms, backbone 1807.3 ms (**83.5%**), decoder 266.1 ms, total 2164.6 ms.

`bf16_mixed@fp32` (matches the 680 ms e2e): encoder 65.0 ms, backbone 483.0 ms (**71.3%**), decoder 128.5 ms, total 677.3 ms.

A same-process unfused stage split that follows custom kernels is deflated (total 1136.9 ms) and must not be cited as the cold FP32 breakdown.

## One-step mean relative error vs unfused FP32 twin

All weather variables pass the golden test (`0/9` fail) on both ICs. $\bar{e}_v=\mathrm{mean}(|y-\hat{y}|)/\mathrm{mean}(|\hat{y}|)$.

| IC | tier | 2t | msl | 10u | 10v |
| --- | --- | ---: | ---: | ---: | ---: |
| 2023-01-01 | `bf16_mixed@fp32` | 3.85e-5 | 5.67e-6 | 8.27e-4 | 1.01e-3 |
| 2023-01-01 | `tf32@fp32` | 1.02e-5 | 1.84e-6 | 3.62e-4 | 4.53e-4 |
| 2023-01-01 | autocast | 4.36e-5 | 7.40e-6 | 1.42e-3 | 1.80e-3 |
| 2026-07-01 | `bf16_mixed@fp32` | 3.92e-5 | 5.53e-6 | 8.25e-4 | 9.63e-4 |
| 2026-07-01 | `tf32@fp32` | 3.05e-6 | 7.37e-7 | 1.60e-4 | 2.01e-4 |
| 2026-07-01 | autocast | 4.40e-5 | 7.41e-6 | 1.39e-3 | 1.70e-3 |

Closed-loop mean-rel at step 20 (120 h) on 2023-01-01, not teacher-forced (full series: `rollout_drift_latest.md`):

| tier | 2t | 10u | msl | worst |
| --- | ---: | ---: | ---: | --- |
| `bf16_mixed@fp32` | 1.03e-4 | 1.04e-2 | 4.02e-5 | 10v 1.37e-2 |
| `tf32@fp32` | 4.99e-5 | 5.56e-3 | 2.25e-5 | 10v 7.65e-3 |
| autocast | 1.57e-4 | 1.49e-2 | 5.62e-5 | 10v 2.06e-2 |

This is implementation fidelity against an internal FP32 twin, not WeatherBench2 RMSE against analysis.

## Peak allocated VRAM (`bf16_mixed@fp32`, one forward)

| preset | grid | peak GiB |
| --- | --- | ---: |
| `era5_pretrained` | 721x1440 | 26.6 |
| `aurora_v1p5` | 721x1440 | 24.3 |
| `hres_t0_finetuned` | 721x1440 | 27.7 |
| `hres_0.1` | 1801x3600 | 33.4 |
| `cams` | 451x900 | 24.0 |

40-step ERA5 AR peak is 26.6 GiB for FP32, mixed, TF32, and autocast (hybrid is a speed path, not a memory path). 16-step CAMS AR peak is 44.5 GiB (FP32/TF32) and 41.8 GiB (mixed/autocast).

## Closed-loop mean-rel drift

See `benchmark/rollout_drift_latest.md`. On `era5_pretrained`, `bf16_mixed@fp32` stays within one-step golden tolerances through step 12 (72 h); first fail is winds at step 13. `tf32@fp32` first fails at step 17. Autocast first fails at step 8. On CAMS, `tf32@fp32` passes all 22 variables through 96 h.

## Reproduce

```bash
export AURORA_ASSET_ROOT=/path/to/aurora
export CUTE_DSL_ARCH=sm_120a
uv run --python 3.12 python benchmark/bench_indepth_eval.py
uv run --python 3.12 python benchmark/bench_indepth_fillin.py
uv run --python 3.12 python benchmark/bench_rollout_drift.py --presets era5_pretrained cams --steps 40 --cams-steps 16
```
