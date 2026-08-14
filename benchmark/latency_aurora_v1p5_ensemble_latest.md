# Aurora end-to-end latency (all presets except wave)

- Generated: 2026-08-14T12:01:47
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

## aurora_v1p5_ensemble (721x1440)

| Tier | forward (ms) | vs PyTorch FP32 ref |
|------|-------------:|--------------------:|
| pytorch_backbone_fp32_encoder_decoder_fp32 | 2535.1 | base |
| bf16_mixed@fp32 | 1004.9 | 2.52x |
| bf16_mixed@tf32 | 1006.6 | 2.52x |
| tf32@fp32 | 1411.9 | 1.80x |
| tf32@tf32 | 1248.7 | 2.03x |
| fp32@fp32 | 2450.8 | 1.03x |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 1137.1 | 2.23x |

