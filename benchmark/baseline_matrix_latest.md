# Group A PyTorch baseline matrix

- Generated: 2026-09-21T00:57:00
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- Preset: `era5_pretrained`
- Isolate-tiers, seed 42, warmup 2, n=5
- Compile extra warmup: 8
- Strongest tau_v-passing PyTorch baseline: `compile_autocast`

| row | mechanism | mean±std (ms) | p50 | p99 | peak GiB | vs mixed | vs strongest | n_fail |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mixed` | production mixed fused path | 678.7±0.31 | 678.7 | 679.1 | 26.3 | 1.00× | 1.23× | 0/9 |
| `eager_fp32` | eager PyTorch FP32, TF32 off, no Triton/CuTe | 2168.1±0.58 | 2167.9 | 2168.9 | 26.3 | 0.31× | 0.39× | 0/9 |
| `eager_tf32` | eager PyTorch TF32 on, no Triton/CuTe | 1302.0±0.42 | 1301.9 | 1302.6 | 26.3 | 0.52× | 0.64× | 0/9 |
| `compile_fp32` | torch.compile after load on eager FP32 | 2034.3±3.11 | 2033.7 | 2037.8 | 26.1 | 0.33× | 0.41× | 0/9 |
| `autocast` | framework autocast BF16, no Triton/CuTe | 1012.4±0.34 | 1012.3 | 1012.9 | 26.3 | 0.67× | 0.83× | 0/9 |
| `compile_autocast` | torch.compile after load on framework autocast | 838.1±0.36 | 838.2 | 838.5 | 26.1 | 0.81× | 1.00× | 0/9 |
| `sdpa_auto` | production mixed with CuTe off, default SDPA dispatch | 808.3±0.28 | 808.3 | 808.7 | 26.3 | 0.84× | 1.04× | 0/9 |
| `sdpa_flash` | production mixed with CuTe off, SDPA flash | RuntimeError: No available kernel. Abort | — | — | — | — | — | RuntimeError: No available kernel. Aborting exec |
| `sdpa_mem_eff` | production mixed with CuTe off, SDPA mem_eff | 807.7±0.39 | 807.8 | 808.2 | 26.3 | 0.84× | 1.04× | 0/9 |
| `sdpa_math` | production mixed with CuTe off, SDPA math | 1087.8±0.26 | 1087.9 | 1088.1 | 26.3 | 0.62× | 0.77× | 0/9 |
| `sdpa_cudnn` | production mixed with CuTe off, SDPA cudnn | RuntimeError: No available kernel. Abort | — | — | — | — | — | RuntimeError: No available kernel. Aborting exec |

`vs mixed` is mixed mean / row mean. `vs strongest` uses the fastest eager or compile row that passes every tau_v. SDPA rows turn CuTe off on the mixed path; they are attention backends, not the headline PyTorch floor.

## Reproduce

```bash
export AURORA_ASSET_ROOT=/path/to/aurora
export CUTE_DSL_ARCH=sm_120a
uv run python benchmark/bench_baseline_matrix.py
uv run python benchmark/bench_window_attn_sota.py
```
