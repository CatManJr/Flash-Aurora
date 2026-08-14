# Mechanism leave-one-out ablation

Paradigm: GFM inference keeps spatially structured tensors and high-intensity I/O.
This table attributes the fused forward path (layout, AdaLN, short-window attention, precision).
Each leave-one-out row turns off one mechanism in a fresh subprocess.
Scheduler, pipeline, and ROI egress are reported elsewhere. Do not mix with isolate-tiers ratios.

- Generated: 2026-08-14T16:48:12
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- Preset: `era5_pretrained`
- Seed: 42; warmup 3, per-iter CUDA events n=12
- Compile-after-load extra warmup: 8

| row | mechanism | mean±std (ms) | p50 | p99 | peak GiB | vs full | vs FP32 | 2t | 10u | n_fail |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | stacked production mixed fused path | 679.1 ± 0.29 | 679.1 | 679.6 | 26.3 | 1.00x (tie) | 3.20x | 3.841e-05 | 8.232e-04 | 0/9 |
| `no_layout` | fused window layout | 760.0 ± 0.30 | 760.0 | 760.4 | 26.3 | 1.12x | 2.86x | 3.841e-05 | 8.232e-04 | 0/9 |
| `no_adaln` | fused AdaLN and residual | 791.0 ± 0.28 | 791.0 | 791.5 | 26.3 | 1.16x | 2.74x | 3.849e-05 | 8.231e-04 | 0/9 |
| `no_cute` | short-window CuTe attention | 808.6 ± 0.40 | 808.7 | 809.2 | 26.3 | 1.19x | 2.68x | 3.760e-05 | 7.119e-04 | 0/9 |
| `no_bf16_routing` | BF16 mixed-precision routing | 1087.9 ± 1.11 | 1087.9 | 1089.4 | 26.3 | 1.60x | 1.99x | 4.863e-06 | 1.760e-04 | 0/9 |
| `unfused_fp32` | do-nothing PyTorch FP32 | 2169.7 ± 1.92 | 2169.3 | 2172.8 | 26.3 | 3.20x | 1.00x (tie) | 0.000e+00 | 0.000e+00 | 0/9 |
| `pytorch_autocast` | framework mixed precision | 1013.4 ± 0.63 | 1013.4 | 1014.4 | 26.3 | 1.49x | 2.14x | 4.303e-05 | 1.385e-03 | 0/9 |
| `compile_after_load` | torch.compile after checkpoint load | 676.4 ± 0.26 | 676.4 | 676.9 | 26.1 | 1.00x | 3.21x | 3.844e-05 | 8.290e-04 | 0/9 |

`vs full` is candidate mean / production mean (slowdown when a mechanism is removed).
`vs FP32` is unfused FP32 mean / candidate mean. Overlapping mean±std intervals are marked `(tie)`.

## Reproduce

```bash
export AURORA_ASSET_ROOT=/path/to/aurora
export CUTE_DSL_ARCH=sm_120a
uv run --python 3.12 python benchmark/bench_ablation_loo.py
```
