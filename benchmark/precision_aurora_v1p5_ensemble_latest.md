# Aurora precision suite (all presets except wave)

- Generated: 2026-08-14T12:04:10
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: `2.12.1+cu130`
- Asset root: `/root/autodl-tmp/aurora`
- Seed: **42** (torch / cuda / numpy / random)
- Baseline: `pytorch_backbone_fp32_encoder_decoder_fp32` (PyTorch backbone FP32, E/D FP32, no Triton/CuTe)
- Finetuned presets: `lora_merged`
- Pollution vars (CAMS): heuristic tol `5e-3` (same as wind/q)

## Failure summary

- **aurora_v1p5_ensemble** / `bf16_mixed@fp32` / `scaled_tp_1h`: mean_rel=7.3852e-03 (tol=5e-03, **1.5x**)
- **aurora_v1p5_ensemble** / `bf16_mixed@fp32` / `scaled_sf_1h`: mean_rel=6.1901e-03 (tol=5e-03, **1.2x**)
- **aurora_v1p5_ensemble** / `bf16_mixed@tf32` / `hcc`: mean_rel=5.3957e-03 (tol=5e-03, **1.1x**)
- **aurora_v1p5_ensemble** / `bf16_mixed@tf32` / `scaled_tp_1h`: mean_rel=8.1831e-03 (tol=5e-03, **1.6x**)
- **aurora_v1p5_ensemble** / `bf16_mixed@tf32` / `scaled_sf_1h`: mean_rel=6.5951e-03 (tol=5e-03, **1.3x**)
- **aurora_v1p5_ensemble** / `bf16@fp32` / `lcc`: mean_rel=6.0493e-03 (tol=5e-03, **1.2x**)
- **aurora_v1p5_ensemble** / `bf16@fp32` / `mcc`: mean_rel=7.6588e-03 (tol=5e-03, **1.5x**)
- **aurora_v1p5_ensemble** / `bf16@fp32` / `hcc`: mean_rel=8.2426e-03 (tol=5e-03, **1.6x**)
- **aurora_v1p5_ensemble** / `bf16@fp32` / `scaled_tp_1h`: mean_rel=1.2487e-02 (tol=5e-03, **2.5x**)
- **aurora_v1p5_ensemble** / `bf16@fp32` / `scaled_sf_1h`: mean_rel=1.0575e-02 (tol=5e-03, **2.1x**)
- **aurora_v1p5_ensemble** / `pytorch_backbone_autocast_bf16_encoder_decoder_fp32` / `lcc`: mean_rel=5.4709e-03 (tol=5e-03, **1.1x**)
- **aurora_v1p5_ensemble** / `pytorch_backbone_autocast_bf16_encoder_decoder_fp32` / `mcc`: mean_rel=7.0957e-03 (tol=5e-03, **1.4x**)
- **aurora_v1p5_ensemble** / `pytorch_backbone_autocast_bf16_encoder_decoder_fp32` / `hcc`: mean_rel=7.5404e-03 (tol=5e-03, **1.5x**)
- **aurora_v1p5_ensemble** / `pytorch_backbone_autocast_bf16_encoder_decoder_fp32` / `scaled_tp_1h`: mean_rel=1.1629e-02 (tol=5e-03, **2.3x**)
- **aurora_v1p5_ensemble** / `pytorch_backbone_autocast_bf16_encoder_decoder_fp32` / `scaled_sf_1h`: mean_rel=9.6814e-03 (tol=5e-03, **1.9x**)

## aurora_v1p5_ensemble

| tier | pass | 2t | 10u | 10v | msl | 2d | tcwv | tcc | 100u | 100v | sp | lcc | mcc | hcc | skt | stl1 | swvl1 | ci | scaled_sd | insolation | i10fg | blh | uvb_1h | ssrd_1h | ttr_1h | scaled_tp_1h | scaled_sf_1h | z | u | v | t | q |
|------|-----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16_mixed@fp32 | 29/31 | 2.89e-05 | 1.63e-03 | 2.12e-03 | 5.68e-06 | 3.71e-05 | 6.92e-04 | 2.29e-03 | 1.62e-03 | 2.04e-03 | 2.73e-05 | 3.44e-03 | 4.63e-03 | 4.76e-03 | 3.32e-05 | 2.90e-05 | 6.34e-04 | 1.23e-03 | 1.28e-03 | 0.00e+00 | 1.14e-03 | 1.65e-03 | 1.04e-03 | 1.10e-03 | 4.15e-04 | **7.39e-03** | **6.19e-03** | 6.80e-06 | 1.21e-03 | 2.31e-03 | 2.38e-05 | 1.18e-03 |
| bf16_mixed@tf32 | 28/31 | 2.94e-05 | 1.75e-03 | 2.26e-03 | 5.79e-06 | 3.78e-05 | 7.16e-04 | 2.56e-03 | 1.73e-03 | 2.18e-03 | 2.73e-05 | 3.70e-03 | 4.99e-03 | **5.40e-03** | 3.36e-05 | 2.92e-05 | 6.35e-04 | 1.23e-03 | 1.29e-03 | 0.00e+00 | 1.20e-03 | 1.75e-03 | 1.09e-03 | 1.18e-03 | 4.70e-04 | **8.18e-03** | **6.60e-03** | 7.16e-06 | 1.38e-03 | 2.63e-03 | 2.61e-05 | 1.25e-03 |
| tf32@fp32 | 31/31 | 1.15e-05 | 6.85e-04 | 9.16e-04 | 2.35e-06 | 1.32e-05 | 2.71e-04 | 1.16e-03 | 6.77e-04 | 8.90e-04 | 1.09e-05 | 1.51e-03 | 2.04e-03 | 2.58e-03 | 1.16e-05 | 1.03e-05 | 2.09e-04 | 3.55e-04 | 7.76e-04 | 0.00e+00 | 4.20e-04 | 6.34e-04 | 4.90e-04 | 5.57e-04 | 2.25e-04 | 3.59e-03 | 2.62e-03 | 5.23e-06 | 6.53e-04 | 1.27e-03 | 1.19e-05 | 4.41e-04 |
| tf32@tf32 | 31/31 | 1.15e-05 | 6.85e-04 | 9.16e-04 | 2.35e-06 | 1.32e-05 | 2.71e-04 | 1.16e-03 | 6.77e-04 | 8.90e-04 | 1.09e-05 | 1.51e-03 | 2.04e-03 | 2.58e-03 | 1.16e-05 | 1.03e-05 | 2.09e-04 | 3.55e-04 | 7.76e-04 | 0.00e+00 | 4.20e-04 | 6.34e-04 | 4.90e-04 | 5.57e-04 | 2.25e-04 | 3.59e-03 | 2.62e-03 | 5.23e-06 | 6.53e-04 | 1.27e-03 | 1.19e-05 | 4.41e-04 |
| fp32@fp32 | 31/31 | 1.12e-05 | 6.46e-04 | 8.69e-04 | 2.24e-06 | 1.28e-05 | 2.55e-04 | 1.11e-03 | 6.36e-04 | 8.39e-04 | 1.07e-05 | 1.43e-03 | 1.92e-03 | 2.48e-03 | 1.14e-05 | 1.02e-05 | 2.09e-04 | 3.53e-04 | 7.78e-04 | 0.00e+00 | 3.91e-04 | 5.89e-04 | 4.68e-04 | 5.33e-04 | 2.13e-04 | 3.37e-03 | 2.46e-03 | 5.18e-06 | 6.30e-04 | 1.23e-03 | 1.16e-05 | 4.21e-04 |
| bf16@fp32 | 26/31 | 5.62e-05 | 3.06e-03 | 3.93e-03 | 1.31e-05 | 6.43e-05 | 1.25e-03 | 4.01e-03 | 3.02e-03 | 3.89e-03 | 5.82e-05 | **6.05e-03** | **7.66e-03** | **8.24e-03** | 6.41e-05 | 6.57e-05 | 1.11e-03 | 2.13e-03 | 3.67e-03 | 0.00e+00 | 2.02e-03 | 2.76e-03 | 2.04e-03 | 2.17e-03 | 7.29e-04 | **1.25e-02** | **1.06e-02** | 1.46e-05 | 2.23e-03 | 4.19e-03 | 4.22e-05 | 1.80e-03 |
| pytorch_backbone_autocast_bf16_encoder_decoder_fp32 | 26/31 | 4.02e-05 | 2.68e-03 | 3.50e-03 | 9.04e-06 | 5.04e-05 | 9.73e-04 | 3.62e-03 | 2.67e-03 | 3.49e-03 | 3.24e-05 | **5.47e-03** | **7.10e-03** | **7.54e-03** | 4.19e-05 | 3.44e-05 | 7.05e-04 | 1.34e-03 | 1.43e-03 | 0.00e+00 | 1.85e-03 | 2.54e-03 | 1.92e-03 | 2.04e-03 | 6.65e-04 | **1.16e-02** | **9.68e-03** | 1.04e-05 | 1.91e-03 | 3.64e-03 | 3.50e-05 | 1.55e-03 |
