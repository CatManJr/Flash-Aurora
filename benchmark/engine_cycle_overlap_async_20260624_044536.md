# Engine cycle profile (no download)

- Generated: 2026-06-24T04:49:13
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Asset root: `/root/autodl-tmp/aurora`
- Rollout steps: 2
- Inference precision: `bf16_mixed@fp32` (unless preset default)
- Excludes: CDS/ADS/MARS download and API queue
- Overlap IC+load: `True`
- Async export: `True`
- Forward warmup: `2` (excluded from rollout timing)

## Summary

| preset | total (s) | bottleneck | prepare (s) | build_model | load_ckpt | rollout | export |
|--------|----------:|------------|------------:|------------:|----------:|--------:|-------:|
| era5_pretrained | 30.73 | prepare_overlap | 26.29 | 20.58 | 4.90 | 1.73 | 2.18 |
| hres_0.1 | 149.53 | prepare_overlap | 134.20 | 21.31 | 7.22 | 3.64 | 8.92 |

## Per-stage breakdown

### era5_pretrained

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 26285.9 | 85.5 |
| export | 2180.3 | 7.1 |
| rollout_forward | 1372.6 | 4.5 |
| batch_prep_h2d | 529.8 | 1.7 |
| rollout_overhead | 9.5 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 686.3 ms (rollout overhead: 9.5 ms total)
- batch H2D prep: 529.8 ms; model H2D: 808.5 ms
- forward stages (CUDA avg): encoder 64.8 ms, backbone 483.5 ms, decoder 128.3 ms

### hres_0.1

| stage | ms | % of total |
|-------|---:|-----------:|
| prepare_overlap | 134198.2 | 89.7 |
| export | 8917.7 | 6.0 |
| batch_prep_h2d | 2769.2 | 1.9 |
| rollout_forward | 1364.2 | 0.9 |
| rollout_overhead | 11.4 | 0.0 |
| ingest_request | 0.0 | 0.0 |

- forward/step: 682.1 ms (rollout overhead: 11.4 ms total)
- batch H2D prep: 2769.2 ms; model H2D: 2631.6 ms
- forward stages (CUDA avg): encoder 100.0 ms, backbone 412.4 ms, decoder 147.3 ms
