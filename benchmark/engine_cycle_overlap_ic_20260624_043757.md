# Engine cycle profile (no download)

- Generated: 2026-06-24T04:41:39
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 2
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `True`
- Async export: `False`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 32.44 | prepare_overlap | 28.71 | 20.61 | 6.74 | 1.36 | 1.82 |
| hres_0.1 | 144.48 | prepare_overlap | 129.48 | 20.76 | 6.52 | 1.36 | 11.26 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 28713.0 | 88.5 |
| export | 1815.7 | 5.6 |
| rollout_forward | 1361.2 | 4.2 |
| batch_prep_h2d | 543.7 | 1.7 |
| rollout_overhead | 0.5 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 680.6 ms (rollout overhead: 0.5 ms total)
- batch H2D prep: 543.7 ms; model H2D: 1361.9 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.6 ms, decoder 128.2 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 129482.4 | 89.6 |
| export | 11256.1 | 7.8 |
| batch_prep_h2d | 2377.7 | 1.6 |
| rollout_forward | 1355.9 | 0.9 |
| rollout_overhead | 0.5 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 678.0 ms (rollout overhead: 0.5 ms total)
- batch H2D prep: 2377.7 ms; model H2D: 1062.3 ms
- forward stages (CUDA avg): encoder 99.9 ms, backbone 412.5 ms, decoder 147.2 ms
