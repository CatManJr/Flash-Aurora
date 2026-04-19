# Fused MLP (Triton) ablation — sweep

- Generated: 2026-03-27T01:20:56
- PyTorch: 2.10.0+cu128
- Preset: `unit_mlp` → D=128, H=512
- Sweep M: [64, 256, 1024, 4096, 8192, 16384]
- Warmup: 15, repeat: 80

## Analysis

- - In this sweep, **PyTorch (cuBLAS) stayed faster** than the fused Triton kernel at every M tested.
- - **Rows/s** = `M / (t_ms/1000)` (one batch forward).
- - **GEMM GFLOP/s** uses `2·M·D·H + 2·M·H·D` FLOPs (two matmuls; GELU omitted).
- - Small `M`: two tiny cuBLAS calls often beat one custom fused kernel; cross-over is shape- and GPU-dependent.

## Throughput sweep

| M | torch ms | triton ms | torch/triton | torch rows/s | triton rows/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0464 | 0.4132 | 0.112x | 1,378,360 | 154,907 |
| 256 | 0.0682 | 0.4147 | 0.164x | 3,754,106 | 617,355 |
| 1024 | 0.0986 | 0.4106 | 0.240x | 10,381,184 | 2,494,057 |
| 4096 | 0.2450 | 0.6787 | 0.361x | 16,716,730 | 6,034,747 |
| 8192 | 0.3836 | 1.0156 | 0.378x | 21,355,579 | 8,066,295 |
| 16384 | 0.7145 | 1.9139 | 0.373x | 22,931,363 | 8,560,513 |

