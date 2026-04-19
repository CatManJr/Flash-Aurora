# Swin3D Stage-A compare (baseline vs Triton)

- Generated: 2026-03-26T22:58:30

## Forward latency

| Run | Timing |
| --- | --- |
| Baseline | GPU: 250.87 ms for 4 forwards -> 62.72 ms/forward |
| Triton (layout+AdaLN) | GPU: 232.65 ms for 4 forwards -> 58.16 ms/forward |
- **Speedup:** 1.078x
- **Delta:** 4.56 ms/forward

## Focus KPIs

| Bucket | Baseline (ms) | Triton (ms) | Delta (ms) | Delta % |
| --- | ---: | ---: | ---: | ---: |
| copy_layout | 11.022 | 1.024 | -9.998 | -90.7 |
| roll_pad_layout | 9.025 | 4.489 | -4.536 | -50.3 |
| layer_norm | 7.441 | 0.815 | -6.626 | -89.0 |

## Aggregate totals

| Run | Total aggregate self-time (ms) |
| --- | ---: |
| Baseline | 410.934 |
| Triton (layout+AdaLN) | 377.309 |
