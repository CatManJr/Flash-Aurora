# Aurora precision matrix benchmark

- device: NVIDIA RTX PRO 6000 Blackwell Server Edition
- model: full (Aurora 0.25° pretrained (embed=512, 48 Swin blocks))
- checkpoint: /root/autodl-tmp/aurora/aurora-0.25-pretrained.ckpt
- batch_size: 1
- vram_fraction: 0.9
- preset: production
- grid: 721x1440
- lora_merged: False
- cuda_graph_flag: False
- warmup/repeat: 5/30

| tier | precision | ms/forward | forwards/s | peak alloc MB | peak reserved MB | speedup vs baseline | cuda graph | max abs diff | mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fp32 | fp32 | 1967.272 | 0.51 | 24795.0 | 29274.1 | 1.00x | no |  |  |
| pytorch_autocast | pytorch_autocast | 852.998 | 1.17 | 23234.1 | 29280.4 | 2.31x | no | 4.550438e+03 | 7.498215e+01 |
| fast_fp32 | fast_fp32 | 1809.245 | 0.55 | 24795.0 | 29274.1 | 1.09x | no | 4.695922e+03 | 7.517817e+01 |
| tf32_1x | tf32_1x | 1798.157 | 0.56 | 24793.8 | 29278.3 | 1.09x | no | 4.778734e+03 | 7.677281e+01 |
| bf16_mixed | bf16_mixed | 1815.871 | 0.55 | 24795.0 | 29274.1 | 1.08x | no | 4.581031e+03 | 7.540578e+01 |
| full_bf16 | full_bf16 | 1819.579 | 0.55 | 24795.8 | 29869.7 | 1.08x | no | 5.322531e+03 | 9.963027e+01 |
