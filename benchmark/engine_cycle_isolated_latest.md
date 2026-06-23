# Engine cycle isolated comparison

- Generated: 2026-06-24T04:49:15
- Isolation: **one subprocess per configuration**
- Each run: `--model-warmup` + `--forward-warmup 2`
- Includes NetCDF export
- Export parent: `/tmp/flash-aurora-engine-cycle` (cleaned per config)

## Summary (total wall time, seconds)

| preset | baseline | overlap_ic | async_export | overlap_async | best | vs baseline |
|--------|----------:|----------:|----------:|----------:|------|------------:|
| era5_pretrained | 36.4 | 32.4 | 38.7 | 30.7 | overlap_async | -15.5% |
| hres_0.1 | 160.0 | 144.5 | 157.8 | 149.5 | overlap_ic | -9.7% |

## Rollout forward/step (ms, warmed)

| preset | baseline | overlap_ic | async_export | overlap_async |
|--------|-------:|-------:|-------:|-------:|
| era5_pretrained | 679.8 | 680.6 | 689.6 | 686.3 |
| hres_0.1 | 677.6 | 678.0 | 680.8 | 682.1 |

## Export total (seconds, timed steps)

| preset | baseline | overlap_ic | async_export | overlap_async |
|--------|-------:|-------:|-------:|-------:|
| era5_pretrained | 2.3 | 1.8 | 2.4 | 2.2 |
| hres_0.1 | 9.9 | 11.3 | 8.6 | 8.9 |
