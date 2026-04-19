# Swin3DTransformerBackbone profiling

- Generated: 2026-03-26T22:10:22
- Torch: 2.10.0+cu128
- Config: batch=1, patch_res=(4, 32, 64), L=8192, autocast_backbone=False

## Timer

GPU: 209.48 ms for 4 forwards → 52.37 ms/forward

## Top operators

| Rank | Operator | Self (ms) |
| ---: | --- | ---: |
| 1 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 147.644 |
| 2 | aten::addmm | 85.604 |
| 3 | aten::mm | 81.272 |
| 4 | ## Call CompiledFxGraph fksrlen2kiyuz3g55ml6zwzeozd66kmdxi7cgapms5qwjkaxxn5v ## | 59.669 |
| 5 | ## Call CompiledFxGraph fvsfuvsqjofguwd2qvcubl4m5cguz74cn3bcofziibz5reaiem2i ## | 58.985 |
| 6 | ## Call CompiledFxGraph fehqb4bsecbk722ikbea7hma7dryziedwooq74xutcafzgyae6n3 ## | 25.808 |
| 7 | aten::_efficient_attention_forward | 25.294 |
| 8 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 25.294 |
| 9 | ## Call CompiledFxGraph fehmgmzcpq7xys6xenmezslwzt4yggbo3ioxrabknxgm6nocycfs ## | 24.304 |
| 10 | ## Call CompiledFxGraph fn6rrbxpwbab7w7jy7xw65kpolwa4s2ryuafyc5bmyldu2jypawg ## | 24.183 |
| 11 | ## Call CompiledFxGraph fqeexyu42322pot36jjxpzn4jg7hbdkm7m44r25exw7la2koijej ## | 22.775 |
| 12 | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) | 8.270 |
| 13 | triton_poi_fused__unsafe_view_add_addmm_mul_view_1 | 6.100 |
| 14 | triton_poi_fused__unsafe_view_add_addmm_mul_view_1 | 6.100 |
| 15 | void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Params) | 5.039 |
| 16 | triton_poi_fused_gelu_view_5 | 3.727 |
| 17 | triton_poi_fused_gelu_view_5 | 3.727 |
| 18 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params) | 3.184 |
| 19 | triton_per_fused_add_addmm_mul_native_layer_norm_split_unsqueeze_view_6 | 1.239 |
| 20 | triton_per_fused_add_addmm_mul_native_layer_norm_split_unsqueeze_view_6 | 1.239 |
| 21 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params) | 1.072 |
| 22 | triton_red_fused__unsafe_view_add_addmm_clone_mul_native_layer_norm_native_layer_norm_backward_permute_roll_slice_split_unsqueeze_view_21 | 0.788 |
| 23 | triton_red_fused__unsafe_view_add_addmm_clone_mul_native_layer_norm_native_layer_norm_backward_permute_roll_slice_split_unsqueeze_view_21 | 0.788 |
| 24 | void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float) | 0.728 |
| 25 | void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1>(cutlass_80_simt_sgemm_32x128_8x5_tn_align1::Params) | 0.649 |

## Full profiler table

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us     147.644ms        70.97%     147.644ms     380.525us           0 B           0 B           0 B           0 B           388  
                                            aten::addmm         0.77%       1.699ms         1.24%       2.741ms      16.318us      85.604ms        41.15%      85.604ms     509.549us           0 B           0 B       8.00 KB       8.00 KB           168  
                                               aten::mm         1.61%       3.546ms         3.19%       7.028ms      10.585us      81.272ms        39.07%      81.272ms     122.397us           0 B           0 B           0 B           0 B           664  
## Call CompiledFxGraph fksrlen2kiyuz3g55ml6zwzeozd6...         0.00%       0.000us         0.00%       0.000us       0.000us      59.669ms        28.68%      59.669ms      14.917ms           0 B           0 B           0 B           0 B             4  
## Call CompiledFxGraph fvsfuvsqjofguwd2qvcubl4m5cgu...         0.00%       0.000us         0.00%       0.000us       0.000us      58.985ms        28.35%      58.985ms      14.746ms           0 B           0 B           0 B           0 B             4  
## Call CompiledFxGraph fehqb4bsecbk722ikbea7hma7dry...         0.00%       0.000us         0.00%       0.000us       0.000us      25.808ms        12.41%      25.808ms       6.452ms           0 B           0 B           0 B           0 B             4  
                     aten::_efficient_attention_forward         0.24%     538.643us         0.94%       2.066ms      25.825us      25.294ms        12.16%      25.294ms     316.181us       1.25 KB          64 B     491.78 MB           0 B            80  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us      25.294ms        12.16%      25.294ms     316.181us           0 B           0 B           0 B           0 B            80  
## Call CompiledFxGraph fehmgmzcpq7xys6xenmezslwzt4y...         0.00%       0.000us         0.00%       0.000us       0.000us      24.304ms        11.68%      24.304ms       6.076ms           0 B           0 B           0 B           0 B             4  
## Call CompiledFxGraph fn6rrbxpwbab7w7jy7xw65kpolwa...         0.00%       0.000us         0.00%       0.000us       0.000us      24.183ms        11.62%      24.183ms       6.046ms           0 B           0 B           0 B           0 B             4  
## Call CompiledFxGraph fqeexyu42322pot36jjxpzn4jg7h...         0.00%       0.000us         0.00%       0.000us       0.000us      22.775ms        10.95%      22.775ms       5.694ms           0 B           0 B           0 B           0 B             4  
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       8.270ms         3.98%       8.270ms     516.865us           0 B           0 B           0 B           0 B            16  
     triton_poi_fused__unsafe_view_add_addmm_mul_view_1         0.20%     436.918us         0.40%     888.037us      11.100us       6.100ms         2.93%       6.100ms      76.253us           0 B           0 B           0 B           0 B            80  
     triton_poi_fused__unsafe_view_add_addmm_mul_view_1         0.00%       0.000us         0.00%       0.000us       0.000us       6.100ms         2.93%       6.100ms      76.253us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x...         0.00%       0.000us         0.00%       0.000us       0.000us       5.039ms         2.42%       5.039ms      39.371us           0 B           0 B           0 B           0 B           128  
                           triton_poi_fused_gelu_view_5         0.13%     286.903us         0.31%     677.759us       8.472us       3.727ms         1.79%       3.727ms      46.593us           0 B           0 B           0 B           0 B            80  
                           triton_poi_fused_gelu_view_5         0.00%       0.000us         0.00%       0.000us       0.000us       3.727ms         1.79%       3.727ms      46.593us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       3.184ms         1.53%       3.184ms      33.166us           0 B           0 B           0 B           0 B            96  
triton_per_fused_add_addmm_mul_native_layer_norm_spl...         0.22%     479.578us         0.39%     860.716us      10.759us       1.239ms         0.60%       1.239ms      15.484us           0 B           0 B           0 B           0 B            80  
triton_per_fused_add_addmm_mul_native_layer_norm_spl...         0.00%       0.000us         0.00%       0.000us       0.000us       1.239ms         0.60%       1.239ms      15.484us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.072ms         0.52%       1.072ms     267.877us           0 B           0 B           0 B           0 B             4  
triton_red_fused__unsafe_view_add_addmm_clone_mul_na...         0.12%     254.489us         0.21%     457.622us      14.301us     788.077us         0.38%     788.077us      24.627us           0 B           0 B           0 B           0 B            32  
triton_red_fused__unsafe_view_add_addmm_clone_mul_na...         0.00%       0.000us         0.00%       0.000us       0.000us     788.077us         0.38%     788.077us      24.627us           0 B           0 B           0 B           0 B            32  
void gemv2T_kernel_val<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us     727.943us         0.35%     727.943us       5.687us           0 B           0 B           0 B           0 B           128  
void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8...         0.00%       0.000us         0.00%       0.000us       0.000us     648.782us         0.31%     648.782us      20.274us           0 B           0 B           0 B           0 B            32  
triton_poi_fused_clone_constant_pad_nd_permute_view_...         0.12%     268.143us         0.32%     711.999us      17.800us     585.769us         0.28%     585.769us      14.644us           0 B           0 B           0 B           0 B            40  
triton_poi_fused_clone_constant_pad_nd_permute_view_...         0.00%       0.000us         0.00%       0.000us       0.000us     585.769us         0.28%     585.769us      14.644us           0 B           0 B           0 B           0 B            40  
triton_poi_fused_clone_constant_pad_nd_permute_roll_...         0.08%     174.911us         0.19%     426.700us      10.667us     500.519us         0.24%     500.519us      12.513us           0 B           0 B           0 B           0 B            40  
triton_poi_fused_clone_constant_pad_nd_permute_roll_...         0.00%       0.000us         0.00%       0.000us       0.000us     500.519us         0.24%     500.519us      12.513us           0 B           0 B           0 B           0 B            40  
                                              aten::cat         0.04%      86.526us         0.06%     135.829us      33.957us     448.008us         0.22%     448.008us     112.002us           0 B           0 B      64.00 MB      64.00 MB             4  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 220.554ms
Self CUDA time total: 208.035ms
```

