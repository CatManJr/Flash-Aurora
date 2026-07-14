# Aurora end-to-end latency (all presets except wave)

- Generated: 2026-07-14T08:17:10
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- CUDA: `13.0`
- `CUTE_DSL_ARCH`: `sm_120a`
- Asset root: `/root/autodl-tmp/aurora`
- Warmup: 2, repeat: 5
- Tier isolation: **subprocess per tier**
- PyTorch FP32 ref: **fresh subprocess per tier**
- Finetuned presets: `lora_eager` vs `lora_merged` (engine default)
- Pretrained presets: single `forward` column (no LoRA)
- Reference tier for speedup: `pytorch_backbone_fp32_encoder_decoder_fp32`
- Excluded: `wave` (MARS ingress)
- Tiers exclude `bf16@*` (see README)

## aurora_v1p5 (721x1440)

| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2184.9 | base |
| bf16_mixed@fp32 | 701.4 | 3.12x |
| bf16_mixed@tf32 | 702.5 | 3.11x |
| tf32@fp32 | 1108.3 | 1.97x |
| tf32@tf32 | 947.4 | 2.31x |
| fp32@fp32 | 2002.7 | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1035.4 | 2.11x |

## era5_pretrained (721x1440)

| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2169.0 | base |
| bf16_mixed@fp32 | 679.7 | 3.19x |
| bf16_mixed@tf32 | 679.7 | 3.19x |
| tf32@fp32 | 1087.0 | 2.00x |
| tf32@tf32 | 923.7 | 2.35x |
| fp32@fp32 | 1982.7 | 1.09x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1013.5 | 2.14x |

