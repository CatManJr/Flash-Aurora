# Aurora end-to-end latency (all presets except wave)

- Generated: 2026-06-23T12:48:00
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- CUDA: `13.0`
- `CUTE_DSL_ARCH`: `sm_120a`
- Asset root: `/root/autodl-tmp/aurora`
- Warmup: 2, repeat: 5
- Tier isolation: **subprocess per tier**
- Finetuned presets: `lora_eager` vs `lora_merged` (engine default)
- Pretrained presets: single forward in **merged** column (`eager` = —)
- Reference tier for speedup: `pytorch_backbone_fp32_encoder_decoder_fp32`
- Excluded: `wave` (MARS ingress)
- Tiers exclude `bf16@*` (see README)

## cams (451x900)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1874.5 | 1691.6 | 1.11x | base |
| bf16_mixed@fp32 | 747.3 | 571.0 | 1.31x | 2.96x |
| bf16_mixed@tf32 | 747.5 | 571.9 | 1.31x | 2.96x |
| tf32@fp32 | 1096.1 | 916.5 | 1.20x | 1.85x |
| tf32@tf32 | 898.1 | 718.3 | 1.25x | 2.35x |
| fp32@fp32 | 1734.6 | 1562.3 | 1.11x | 1.08x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 985.7 | 888.6 | 1.11x | 1.90x |

## era5_pretrained (721x1440)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | — | 2128.2 | — | base |
| bf16_mixed@fp32 | — | 676.4 | — | 3.15x |
| bf16_mixed@tf32 | — | 676.8 | — | 3.14x |
| tf32@fp32 | — | 1077.5 | — | 1.98x |
| tf32@tf32 | — | 919.2 | — | 2.32x |
| fp32@fp32 | — | 1945.0 | — | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | — | 1004.4 | — | 2.12x |

## hres_0.1 (1801x3600)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2227.5 | 1994.6 | 1.12x | base |
| bf16_mixed@fp32 | 898.0 | 672.0 | 1.34x | 2.97x |
| bf16_mixed@tf32 | 898.6 | 672.4 | 1.34x | 2.97x |
| tf32@fp32 | 1247.7 | 1019.9 | 1.22x | 1.96x |
| tf32@tf32 | 1091.1 | 861.3 | 1.27x | 2.32x |
| fp32@fp32 | 2051.0 | 1838.0 | 1.12x | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1112.1 | 986.2 | 1.13x | 2.02x |

## hres_t0_finetuned (721x1440)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2307.9 | 2061.9 | 1.12x | base |
| bf16_mixed@fp32 | 881.7 | 638.7 | 1.38x | 3.23x |
| bf16_mixed@tf32 | 881.6 | 638.4 | 1.38x | 3.23x |
| tf32@fp32 | 1249.6 | 1006.3 | 1.24x | 2.05x |
| tf32@tf32 | 1091.5 | 846.5 | 1.29x | 2.44x |
| fp32@fp32 | 2115.5 | 1890.4 | 1.12x | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1104.4 | 967.7 | 1.14x | 2.13x |

## small_pretrained (400x800)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | — | 101.9 | — | base |
| bf16_mixed@fp32 | — | 42.4 | — | 2.40x |
| bf16_mixed@tf32 | — | 42.4 | — | 2.40x |
| tf32@fp32 | — | 64.1 | — | 1.59x |
| tf32@tf32 | — | 57.3 | — | 1.78x |
| fp32@fp32 | — | 94.7 | — | 1.08x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | — | 56.3 | — | 1.81x |

## tc_tracking (721x1440)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2307.9 | 2059.9 | 1.12x | base |
| bf16_mixed@fp32 | 881.7 | 638.5 | 1.38x | 3.23x |
| bf16_mixed@tf32 | 881.7 | 638.3 | 1.38x | 3.23x |
| tf32@fp32 | 1249.6 | 1006.1 | 1.24x | 2.05x |
| tf32@tf32 | 1092.0 | 847.0 | 1.29x | 2.43x |
| fp32@fp32 | 2115.1 | 1890.9 | 1.12x | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1104.2 | 967.4 | 1.14x | 2.13x |

