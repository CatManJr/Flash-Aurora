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
| fp32 | fp32 | 1957.449 | 0.51 | 24795.0 | 29274.1 | 1.00x | no |  |  |  |
| pytorch_autocast | pytorch_autocast | 852.019 | 1.17 | 23234.1 | 29280.4 | 2.30x | no | 4.742406e+03 | 7.551718e+01 | 0.899801 |
| fast_fp32 | fast_fp32 | 1810.509 | 0.55 | 24795.0 | 29274.1 | 1.08x | no | 4.441422e+03 | 7.472166e+01 | 0.896748 |
| tf32_1x | tf32_1x | 917.285 | 1.09 | 24793.8 | 29278.3 | 2.13x | no | 4.486906e+03 | 7.575533e+01 | 0.898403 |
| bf16_mixed | bf16_mixed | 564.729 | 1.77 | 23334.8 | 29806.8 | 3.47x | no | 4.570469e+03 | 8.024253e+01 | 0.897597 |
