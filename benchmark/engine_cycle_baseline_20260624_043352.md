# Engine cycle profile (no download)

- Generated: 2026-06-24T04:37:56
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 2
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `False`
- Async export: `False`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 36.36 | build_model | 32.21 | 19.15 | 5.21 | 1.36 | 2.29 |
| hres_0.1 | 159.97 | build_ic | 146.50 | 19.38 | 4.81 | 1.36 | 9.93 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 19152.3 | 52.7 |
| build_ic | 6914.9 | 19.0 |
| load_ckpt | 5205.8 | 14.3 |
| export | 2289.8 | 6.3 |
| rollout_forward | 1359.6 | 3.7 |
| model_h2d | 935.2 | 2.6 |
| batch_prep_h2d | 505.5 | 1.4 |
| rollout_overhead | 0.4 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 679.8 ms (rollout overhead: 0.4 ms total)
- batch H2D prep: 505.5 ms; model H2D: 935.2 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.5 ms, decoder 128.2 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| build_ic | 121582.6 | 76.0 |
| build_model | 19382.9 | 12.1 |
| export | 9933.8 | 6.2 |
| load_ckpt | 4805.7 | 3.0 |
| batch_prep_h2d | 2183.1 | 1.4 |
| rollout_forward | 1355.2 | 0.8 |
| model_h2d | 725.5 | 0.5 |
| rollout_overhead | 0.4 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 677.6 ms (rollout overhead: 0.4 ms total)
- batch H2D prep: 2183.1 ms; model H2D: 725.5 ms
- forward stages (CUDA avg): encoder 100.1 ms, backbone 412.5 ms, decoder 147.3 ms
