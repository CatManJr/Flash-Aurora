# Swin3D Stage-C compare

- Generated: 2026-03-26T23:36:41
- **LoRA:** randomized A/B (seed=0) for non-zero ΔW.

## Forward latency

| Run | Timing |
| --- | --- |
| Baseline | GPU: 231.19 ms for 4 forwards -> 57.80 ms/forward |
| Stage-A/B (layout+AdaLN+MLP-GELU) | GPU: 216.22 ms for 4 forwards -> 54.05 ms/forward |
| Stage-C (layout+AdaLN+LoRA-merge) | GPU: 224.48 ms for 4 forwards -> 56.12 ms/forward |
- **Stage-A/B speedup vs baseline:** 1.069x
- **Stage-C speedup vs baseline:** 1.030x
- **Stage-C vs Stage-A/B delta:** -2.07 ms/forward (0.963x)

## Aggregate totals

| Run | Total aggregate self-time (ms) |
| --- | ---: |
| Baseline | 439.605 |
| Stage-A/B (layout+AdaLN+MLP-GELU) | 396.697 |
| Stage-C (layout+AdaLN+LoRA-merge) | 409.856 |

- **Stage-C vs Stage-A/B aggregate delta:** -13.159 ms

## addmm stats

| Run | `aten::addmm` calls | `aten::addmm` self-time (ms) |
| --- | ---: | ---: |
| Baseline | 488 | 143.104 |
| Stage-A/B (layout+AdaLN+MLP-GELU) | 488 | 140.037 |
| Stage-C (layout+AdaLN+LoRA-merge) | 488 | 144.590 |

- **Stage-C vs Stage-A/B addmm calls delta:** +0
- **Stage-C vs Stage-A/B addmm self-time delta:** -4.553 ms
