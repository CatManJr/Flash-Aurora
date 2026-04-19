# Swin3D D2 + D3 compare

- Generated: 2026-03-27T02:09:57

- **D2:** `use_triton_layout` + `use_triton_adaln` (fused AdaLN + residual when eval FP32).
- **D3:** `InferenceWorkspacePool` on final decoder `cat` (same math, fewer alloc on that buffer).

## Forward latency

| Run | Timing |
| --- | --- |
| Baseline | GPU: 695.61 ms for 16 forwards -> 43.48 ms/forward |
| D2 (layout + AdaLN) | GPU: 888.44 ms for 16 forwards -> 55.53 ms/forward |
| D2 + D3 (pool) | GPU: 891.92 ms for 16 forwards -> 55.75 ms/forward |

- **D2 vs baseline:** 0.783x
- **D2+D3 vs baseline:** 0.780x
- **D2+D3 vs D2 ms/forward delta:** -0.2200 ms

## CUDA memory (timed forward loop)

Peak stats from `torch.cuda.reset_peak_memory_stats()` before the timed loop, then `max_memory_allocated` / `max_memory_reserved` after the loop (same window as `[timing]`).

| Run | peak allocated (MB) | peak reserved (MB) |
| --- | ---: | ---: |
| Baseline | 2668.5 | 2741.0 |
| D2 | 675.7 | 700.4 |
| D2+D3 | 692.5 | 734.0 |

## aten::addmm

| Run | calls | self (ms) |
| --- | ---: | ---: |
| Baseline | 1952 | 466.300 |
| D2 | 1952 | 521.455 |
| D2+D3 | 1952 | 493.205 |

