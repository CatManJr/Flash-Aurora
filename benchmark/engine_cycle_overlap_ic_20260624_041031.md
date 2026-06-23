# Engine cycle profile (no download)

- Generated: 2026-06-24T04:14:33
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 8
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `True`
- Async export: `False`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 47.33 | prepare_overlap | 32.54 | 25.87 | 5.62 | 5.45 | 8.97 |
| hres_0.1 | 179.65 | prepare_overlap | 126.71 | 20.40 | 5.90 | 5.49 | 45.37 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 32538.0 | 68.7 |
| export | 8969.4 | 18.9 |
| rollout_forward | 5444.9 | 11.5 |
| batch_prep_h2d | 373.6 | 0.8 |
| rollout_overhead | 1.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 680.6 ms (rollout overhead: 1.9 ms total)
- batch H2D prep: 373.6 ms; model H2D: 1048.8 ms
- forward stages (CUDA avg): encoder 65.0 ms, backbone 483.8 ms, decoder 128.8 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 126705.4 | 70.5 |
| export | 45368.7 | 25.3 |
| rollout_forward | 5454.4 | 3.0 |
| batch_prep_h2d | 2082.4 | 1.2 |
| rollout_overhead | 1.6 | 0.0 |
| ingest_request | 0.1 | 0.0 |

- forward/step: 681.8 ms (rollout overhead: 1.6 ms total)
- batch H2D prep: 2082.4 ms; model H2D: 1288.1 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 412.4 ms, decoder 147.3 ms
