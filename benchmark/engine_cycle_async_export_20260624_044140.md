# Engine cycle profile (no download)

- Generated: 2026-06-24T04:45:35
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 2
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `False`
- Async export: `True`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 38.68 | build_model | 33.84 | 19.72 | 5.64 | 1.93 | 2.38 |
| hres_0.1 | 157.82 | build_ic | 143.85 | 18.71 | 4.42 | 3.35 | 8.60 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 19723.1 | 51.0 |
| build_ic | 7002.6 | 18.1 |
| load_ckpt | 5640.2 | 14.6 |
| export | 2377.0 | 6.1 |
| model_h2d | 1474.1 | 3.8 |
| rollout_forward | 1379.2 | 3.6 |
| batch_prep_h2d | 527.2 | 1.4 |
| rollout_overhead | 27.6 | 0.1 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 689.6 ms (rollout overhead: 27.6 ms total)
- batch H2D prep: 527.2 ms; model H2D: 1474.1 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.6 ms, decoder 128.2 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| build_ic | 119861.0 | 76.0 |
| build_model | 18705.3 | 11.9 |
| export | 8598.3 | 5.4 |
| load_ckpt | 4418.6 | 2.8 |
| batch_prep_h2d | 2017.5 | 1.3 |
| rollout_forward | 1361.5 | 0.9 |
| model_h2d | 861.5 | 0.5 |
| rollout_overhead | 8.3 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 680.8 ms (rollout overhead: 8.3 ms total)
- batch H2D prep: 2017.5 ms; model H2D: 861.5 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 412.3 ms, decoder 147.1 ms
