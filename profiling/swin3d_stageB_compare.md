# Swin3D Stage-A compare (baseline vs Triton)

- Generated: 2026-03-26T23:02:06

## Forward latency

| Run | Timing |
| --- | --- |
| Baseline | GPU: 231.25 ms for 4 forwards -> 57.81 ms/forward |
| Triton (layout+AdaLN+MLP-GELU) | GPU: 214.10 ms for 4 forwards -> 53.53 ms/forward |
- **Speedup:** 1.080x
- **Delta:** 4.28 ms/forward

## Focus KPIs

| Bucket | Baseline (ms) | Triton (ms) | Delta (ms) | Delta % |
| --- | ---: | ---: | ---: | ---: |
| copy_layout | 11.789 | 1.049 | -10.739 | -91.1 |
| roll_pad_layout | 9.682 | 4.767 | -4.916 | -50.8 |
| layer_norm | 7.955 | 0.853 | -7.102 | -89.3 |

## Aggregate totals

| Run | Total aggregate self-time (ms) |
| --- | ---: |
| Baseline | 438.987 |
| Triton (layout+AdaLN+MLP-GELU) | 389.698 |
