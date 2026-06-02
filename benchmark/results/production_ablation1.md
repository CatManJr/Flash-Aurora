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

| tier | precision | ms/forward | forwards/s | peak alloc MB | peak reserved MB | speedup vs baseline | cuda graph | max abs diff | mean abs diff | cosine sim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| fp32 | fp32 | 1956.935 | 0.51 | 24795.0 | 29274.1 | 1.00x | no |  |  |  |
| pytorch_autocast | pytorch_autocast | 851.658 | 1.17 | 23234.1 | 29280.4 | 2.30x | no | 4.393273e+03 | 7.631197e+01 | 0.897309 |
| fast_fp32 | fast_fp32 | 1903.981 | 0.53 | 24265.3 | 29274.1 | 1.03x | no | 4.673547e+03 | 7.619654e+01 | 0.899660 |
| tf32_1x | tf32_1x | 1024.473 | 0.98 | 24264.0 | 29278.3 | 1.91x | no | 4.755984e+03 | 7.629995e+01 | 0.898411 |
| bf16_mixed | bf16_mixed | 685.796 | 1.46 | 23334.8 | 29806.8 | 2.85x | no | 4.951328e+03 | 7.972248e+01 | 0.897819 |
