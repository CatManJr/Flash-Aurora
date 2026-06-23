# hres_0.1 IC cache benchmark

- Generated: 2026-06-24 (local)
- GPU host: RTX PRO 6000 96GB environment
- Asset root: `/root/autodl-tmp/aurora`
- Valid time: `2022-05-11 06:00`
- Inputs: cached NetCDF (`2022-05-11-*.nc`)
- `ic_cache=True`, `allow_hub_download=False`

## Results

| run | wall time | notes |
|-----|----------:|-------|
| cold (1st, writes cache) | **111.1 s** | regrid + static + SHA256 fingerprint + write 3.41 GiB `.pt` |
| warm (avg of 3) | **7.9 s** | fingerprint + `torch.load` |
| warm (best) | **7.8 s** | |
| baseline `ic_cache=False` | **96.3 s** | regrid + static only (parallel vars) |

- **Speedup warm vs cold:** 14.0×
- **Speedup warm vs no-cache:** 12.3×
- **Bitwise equal** cold vs warm: yes (`torch.equal` on all fields)

## Warm-path breakdown

| stage | time |
|-------|-----:|
| SHA256 fingerprint (3 NetCDF + static pickle) | 6.0 s |
| `torch.load` cached batch (3.41 GiB) | 4.0 s |

Cache artifacts: `hres_0.1/.ic-cache/2022-05-11-hres_0.1-ic.pt` + `.meta.json`

## Notes

- First run is slower than `ic_cache=False` because of full-file hashing and 3.4 GiB cache write.
- Repeated prepares on the same day/inputs drop **~96 s → ~8 s** for IC build.
- Further wins: lighter cache validation (mtime/size), compressed or memmap store, or pre-regridded smaller on-disk layout.
