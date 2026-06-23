# Engine cycle profile (no download)

- Generated: 2026-06-24T04:27:47
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
| era5_pretrained | 39.82 | prepare_overlap | 26.43 | 19.97 | 5.31 | 5.45 | 7.47 |
| hres_0.1 | 184.52 | prepare_overlap | 132.36 | 21.32 | 5.25 | 5.46 | 44.11 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 26429.8 | 66.4 |
| export | 7468.7 | 18.8 |
| rollout_forward | 5443.1 | 13.7 |
| batch_prep_h2d | 474.6 | 1.2 |
| rollout_overhead | 1.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 680.4 ms (rollout overhead: 1.9 ms total)
- batch H2D prep: 474.6 ms; model H2D: 1156.2 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.8 ms, decoder 128.5 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 132355.6 | 71.7 |
| export | 44107.7 | 23.9 |
| rollout_forward | 5425.2 | 2.9 |
| batch_prep_h2d | 2595.3 | 1.4 |
| rollout_overhead | 1.9 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 678.2 ms (rollout overhead: 1.9 ms total)
- batch H2D prep: 2595.3 ms; model H2D: 1914.4 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 412.4 ms, decoder 147.2 ms
