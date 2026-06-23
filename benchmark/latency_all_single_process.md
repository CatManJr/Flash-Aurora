# Aurora end-to-end latency — single-process artifact (all presets except wave)

Snapshot from a single-process run with PyTorch FP32 ref timed **after** custom tiers (cuDNN cross-tier warmup). Regenerate with `--no-isolate-tiers --defer-ref`.

- Generated: 2026-06-23T11:15:21
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- CUDA: `13.0`
- `CUTE_DSL_ARCH`: `sm_120a`
- Asset root: `/root/autodl-tmp/aurora`
- Warmup: 2, repeat: 5
- Finetuned presets: `lora_eager` vs `lora_merged` (engine default)
- Pretrained presets: single forward in **merged** column (`eager` = —)
- Reference tier for speedup: `pytorch_backbone_fp32_encoder_decoder_fp32`
- Excluded: `wave` (MARS ingress)
- Tiers exclude `bf16@*` (see README)

## cams (451x900)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| bf16_mixed@fp32 | 1467.4 | 571.5 | 2.57x | 1.53x |
| bf16_mixed@tf32 | 746.7 | 571.1 | 1.31x | 1.53x |
| tf32@fp32 | 897.5 | 719.5 | 1.25x | 1.22x |
| tf32@tf32 | 897.8 | 720.3 | 1.25x | 1.22x |
| fp32@fp32 | 915.3 | 738.6 | 1.24x | 1.19x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1054.1 | 875.3 | 1.20x | base |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 788.2 | 691.6 | 1.14x | 1.27x |

## era5_pretrained (721x1440)

| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| bf16_mixed@fp32 | 676.7 | 1.68x |
| bf16_mixed@tf32 | 677.1 | 1.68x |
| tf32@fp32 | 920.7 | 1.23x |
| tf32@tf32 | 921.3 | 1.23x |
| fp32@fp32 | 944.9 | 1.20x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 846.5 | 1.34x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1135.5 | base |

## hres_0.1 (1801x3600)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| bf16_mixed@fp32 | 897.2 | 672.4 | 1.33x | 1.58x |
| bf16_mixed@tf32 | 896.8 | 671.8 | 1.33x | 1.58x |
| tf32@fp32 | 1089.6 | 862.5 | 1.26x | 1.23x |
| tf32@tf32 | 1089.8 | 863.6 | 1.26x | 1.23x |
| fp32@fp32 | 1111.9 | 889.4 | 1.25x | 1.19x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1289.8 | 1060.3 | 1.22x | base |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 955.5 | 829.4 | 1.15x | 1.28x |

## hres_t0_finetuned (721x1440)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| bf16_mixed@fp32 | 882.4 | 639.1 | 1.38x | 1.66x |
| bf16_mixed@tf32 | 882.2 | 639.1 | 1.38x | 1.66x |
| tf32@fp32 | 1091.7 | 847.6 | 1.29x | 1.25x |
| tf32@tf32 | 1092.5 | 848.2 | 1.29x | 1.25x |
| fp32@fp32 | 1115.6 | 874.2 | 1.28x | 1.21x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1308.5 | 1059.9 | 1.23x | base |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 946.0 | 808.8 | 1.17x | 1.31x |

## small_pretrained (400x800)

| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| bf16_mixed@fp32 | 41.9 | 1.59x |
| bf16_mixed@tf32 | 41.8 | 1.59x |
| tf32@fp32 | 57.7 | 1.15x |
| tf32@tf32 | 57.1 | 1.17x |
| fp32@fp32 | 59.6 | 1.12x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 49.5 | 1.34x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 66.5 | base |

## tc_tracking (721x1440)

| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |
|------|----------------:|-----------------:|-------------:|--------------------:|
| bf16_mixed@fp32 | 881.7 | 639.2 | 1.38x | 1.66x |
| bf16_mixed@tf32 | 882.1 | 638.6 | 1.38x | 1.66x |
| tf32@fp32 | 1091.8 | 847.8 | 1.29x | 1.25x |
| tf32@tf32 | 1093.2 | 848.8 | 1.29x | 1.25x |
| fp32@fp32 | 1115.6 | 874.5 | 1.28x | 1.21x |
| pytorch_backbone_fp32_encoder_decoder_fp32 | 1308.5 | 1059.3 | 1.24x | base |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 945.6 | 808.7 | 1.17x | 1.31x |

