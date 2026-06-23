# Engine cycle profile (no download)

- Generated: 2026-06-24T04:10:30
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 8
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `False`
- Async export: `False`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 49.16 | build_model | 34.46 | 20.75 | 5.26 | 5.45 | 8.79 |
| hres_0.1 | 203.56 | build_ic | 155.96 | 26.56 | 4.72 | 5.46 | 40.53 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 20752.8 | 42.2 |
| export | 8794.0 | 17.9 |
| build_ic | 7208.2 | 14.7 |
| rollout_forward | 5443.2 | 11.1 |
| load_ckpt | 5263.7 | 10.7 |
| model_h2d | 1233.9 | 2.5 |
| batch_prep_h2d | 452.4 | 0.9 |
| rollout_overhead | 1.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 680.4 ms (rollout overhead: 1.9 ms total)
- batch H2D prep: 452.4 ms; model H2D: 1233.9 ms
- forward stages (CUDA avg): encoder 65.2 ms, backbone 483.8 ms, decoder 128.6 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| build_ic | 123902.6 | 60.9 |
| export | 40530.7 | 19.9 |
| build_model | 26557.3 | 13.0 |
| rollout_forward | 5428.4 | 2.7 |
| load_ckpt | 4717.3 | 2.3 |
| batch_prep_h2d | 1603.7 | 0.8 |
| model_h2d | 786.4 | 0.4 |
| rollout_overhead | 1.7 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 678.5 ms (rollout overhead: 1.7 ms total)
- batch H2D prep: 1603.7 ms; model H2D: 786.4 ms
- forward stages (CUDA avg): encoder 100.5 ms, backbone 412.4 ms, decoder 147.5 ms
