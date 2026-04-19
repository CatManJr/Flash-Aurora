# Swin3D D2 + D3 compare

- Generated: 2026-03-27T02:18:41

- **D2:** `use_triton_layout` + `use_triton_adaln` (fused AdaLN + residual when eval FP32).
- **D3:** `InferenceWorkspacePool` on final decoder `cat` (same math, fewer alloc on that buffer).

## Forward latency

| Run | Timing |
| --- | --- |
| Baseline | GPU: 2548.46 ms for 16 forwards -> 159.28 ms/forward |
| D2 (layout + AdaLN) | GPU: 2434.53 ms for 16 forwards -> 152.16 ms/forward |
| D2 + D3 (pool) | GPU: 2430.93 ms for 16 forwards -> 151.93 ms/forward |

- **D2 vs baseline:** 1.047x
- **D2+D3 vs baseline:** 1.048x
- **D2+D3 vs D2 ms/forward delta:** +0.2300 ms

## CUDA memory (timed forward loop)

Peak stats from `torch.cuda.reset_peak_memory_stats()` before the timed loop, then `max_memory_allocated` / `max_memory_reserved` after the loop (same window as `[timing]`).

| Run | peak allocated (MB) | peak reserved (MB) |
| --- | ---: | ---: |
| Baseline | 8417.2 | 8845.8 |
| D2 | 1346.7 | 1428.2 |
| D2+D3 | 1413.6 | 1495.3 |

## aten::addmm

| Run | calls | self (ms) |
| --- | ---: | ---: |
| Baseline | 1952 | 1497.792 |
| D2 | 1952 | 1500.383 |
| D2+D3 | 1952 | 1484.167 |

