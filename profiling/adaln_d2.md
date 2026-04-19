# AdaLN D2 profiling (Triton)

- Generated: 2026-03-27T01:37:27
- PyTorch: 2.10.0+cu128
- preset: `aurora_s0`, warmup: 15, repeat: 80

**Variants**

- **composed**: `residual + adaptive_layernorm_film_forward(x, ...)` (Triton AdaLN then PyTorch add).
- **fused**: `adaptive_layernorm_film_add_residual_forward(residual, x, ...)` (single kernel).

## Results

| L | composed ms | fused ms | composed/fused | max|err| | peak CUDA MB (c / f) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 0.0266 | 0.0181 | 1.467x | 0.000e+00 | 6.3 / 5.2 |
| 2048 | 0.0553 | 0.0459 | 1.205x | 0.000e+00 | 25.2 / 21.0 |
| 8192 | 0.2184 | 0.1705 | 1.281x | 0.000e+00 | 100.7 / 83.9 |
