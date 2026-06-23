# Engine cycle profile (no download)

- Generated: 2026-06-23T13:27:13
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 4
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue

## Summary

| preset | total (s) | bottleneck | build_ic | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|---------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 31.68 | build_model | 7.04 | 10.27 | 3.25 | 6.39 | 3.65 |
| hres_t0_finetuned | 20.90 | build_model | 1.59 | 10.67 | 3.23 | 2.57 | 1.99 |
| hres_0.1 | 110.28 | build_ic | 74.37 | 9.49 | 3.36 | 2.72 | 18.89 |
| cams | 21.98 | build_model | 2.61 | 10.78 | 3.46 | 2.32 | 2.11 |
| small_pretrained | 1.81 | build_model | 0.13 | 0.85 | 0.19 | 0.18 | 0.40 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 10268.0 | 32.4 |
| build_ic | 7038.5 | 22.2 |
| rollout_forward | 6384.5 | 20.2 |
| export | 3653.6 | 11.5 |
| load_ckpt | 3250.5 | 10.3 |
| model_h2d | 872.0 | 2.8 |
| batch_prep_h2d | 212.8 | 0.7 |
| rollout_overhead | 0.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 1596.1 ms (rollout overhead: 0.9 ms total)
- batch H2D prep: 212.8 ms; model H2D: 872.0 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 481.4 ms, decoder 127.4 ms

### hres_t0_finetuned

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 10667.6 | 51.1 |
| load_ckpt | 3230.3 | 15.5 |
| rollout_forward | 2570.7 | 12.3 |
| export | 1990.0 | 9.5 |
| build_ic | 1591.7 | 7.6 |
| model_h2d | 711.2 | 3.4 |
| batch_prep_h2d | 131.0 | 0.6 |
| rollout_overhead | 0.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 642.7 ms (rollout overhead: 0.9 ms total)
- batch H2D prep: 131.0 ms; model H2D: 711.2 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 442.8 ms, decoder 127.6 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| build_ic | 74372.6 | 67.4 |
| export | 18888.2 | 17.1 |
| build_model | 9490.1 | 8.6 |
| load_ckpt | 3360.1 | 3.0 |
| rollout_forward | 2706.1 | 2.5 |
| batch_prep_h2d | 788.5 | 0.7 |
| model_h2d | 659.2 | 0.6 |
| rollout_overhead | 1.1 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 676.5 ms (rollout overhead: 1.1 ms total)
- batch H2D prep: 788.5 ms; model H2D: 659.2 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 410.3 ms, decoder 146.5 ms

### cams

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 10782.9 | 49.1 |
| load_ckpt | 3460.0 | 15.7 |
| build_ic | 2605.7 | 11.9 |
| rollout_forward | 2320.4 | 10.6 |
| export | 2111.8 | 9.6 |
| model_h2d | 594.6 | 2.7 |
| batch_prep_h2d | 102.7 | 0.5 |
| rollout_overhead | 1.6 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 580.1 ms (rollout overhead: 1.6 ms total)
- batch H2D prep: 102.7 ms; model H2D: 594.6 ms
- forward stages (CUDA avg): encoder 51.8 ms, backbone 316.5 ms, decoder 194.3 ms

### small_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 849.5 | 46.8 |
| export | 398.1 | 21.9 |
| load_ckpt | 185.7 | 10.2 |
| rollout_forward | 176.7 | 9.7 |
| build_ic | 130.9 | 7.2 |
| model_h2d | 61.1 | 3.4 |
| batch_prep_h2d | 12.0 | 0.7 |
| rollout_overhead | 0.5 | 0.0 |

- forward/step: 44.2 ms (rollout overhead: 0.5 ms total)
- batch H2D prep: 12.0 ms; model H2D: 61.1 ms
- forward stages (CUDA avg): encoder 7.8 ms, backbone 22.6 ms, decoder 10.4 ms
