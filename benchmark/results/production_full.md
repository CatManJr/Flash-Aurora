# Aurora precision matrix benchmark

- device: NVIDIA RTX PRO 6000 Blackwell Server Edition
- model: full (Aurora 0.25° pretrained (embed=512, 48 Swin blocks))
- checkpoint: /root/autodl-tmp/aurora/aurora-0.25-pretrained.ckpt
- batch_size: 4
- vram_fraction: 0.9
- preset: production
- grid: 721x1440
- lora_merged: False
- cuda_graph_flag: False
- warmup/repeat: 5/30

| tier | precision | ms/forward | forwards/s | peak alloc MB | peak reserved MB | speedup vs baseline | cuda graph | max abs diff | mean abs diff | cosine sim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| fp32 | fp32 | 8035.020 | 0.12 | 53138.5 | 72488.1 | 1.00x | no |  |  |  |
| pytorch_autocast | pytorch_autocast | 3407.026 | 0.29 | 50081.5 | 76739.0 | 2.36x | no | 5.269297e+03 | 7.561849e+01 | 0.897469 |
| fast_fp32 | fast_fp32 | 7338.103 | 0.14 | 53135.7 | 72492.3 | 1.09x | no | 6.913453e+03 | 1.597610e+02 | 0.859796 |
| tf32_1x | tf32_1x | 3668.473 | 0.27 | 53136.4 | 72488.1 | 2.19x | no | 6.974172e+03 | 1.588190e+02 | 0.860603 |
| bf16_mixed | bf16_mixed | 2264.081 | 0.44 | 50178.5 | 76739.0 | 3.55x | no | 6.717141e+03 | 1.606212e+02 | 0.859896 |
