# Engine cycle profile (no download)

- Generated: 2026-06-24T04:23:54
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
| era5_pretrained | 45.95 | build_model | 32.46 | 19.36 | 5.10 | 5.44 | 7.62 |
| hres_0.1 | 199.61 | build_ic | 147.06 | 19.49 | 5.14 | 5.46 | 45.09 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| build_model | 19364.3 | 42.1 |
| export | 7619.2 | 16.6 |
| build_ic | 6904.0 | 15.0 |
| rollout_forward | 5437.4 | 11.8 |
| load_ckpt | 5096.0 | 11.1 |
| model_h2d | 1096.6 | 2.4 |
| batch_prep_h2d | 423.0 | 0.9 |
| rollout_overhead | 1.5 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 679.7 ms (rollout overhead: 1.5 ms total)
- batch H2D prep: 423.0 ms; model H2D: 1096.6 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.7 ms, decoder 128.3 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| build_ic | 121233.2 | 60.7 |
| export | 45085.6 | 22.6 |
| build_model | 19492.2 | 9.8 |
| rollout_forward | 5428.5 | 2.7 |
| load_ckpt | 5139.4 | 2.6 |
| batch_prep_h2d | 1999.0 | 1.0 |
| model_h2d | 1192.6 | 0.6 |
| rollout_overhead | 1.7 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 678.6 ms (rollout overhead: 1.7 ms total)
- batch H2D prep: 1999.0 ms; model H2D: 1192.6 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 412.4 ms, decoder 147.4 ms
