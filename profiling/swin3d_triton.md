# Swin3DTransformerBackbone profiling

- Generated: 2026-03-26T22:27:09
- Torch: 2.10.0+cu128
- Config: preset=baseline, batch=1, patch_res=(4, 32, 64), L=8192, autocast_backbone=False, use_triton_layout=True, use_triton_adaln=True

## Timer

GPU: 231.71 ms for 4 forwards → 57.93 ms/forward

## Top operators

| Rank | Operator | Self (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 120.968 |
| 2 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 118.147 |
| 3 | aten::_efficient_attention_forward | 20.674 |
| 4 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 20.674 |
| 5 | aten::mm | 12.758 |
| 6 | aten::add | 8.732 |
| 7 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 8.732 |
| 8 | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) | 6.434 |
| 9 | void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Params) | 4.077 |
| 10 | aten::mul | 3.378 |
| 11 | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 3.367 |
| 12 | aten::gelu | 2.845 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 2.845 |
| 14 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params) | 2.634 |
| 15 | _roll_pad_partition_kernel | 2.535 |
| 16 | _adaln_film_kernel | 1.637 |
| 17 | _crop_roll_unmerge_kernel | 1.605 |
| 18 | cuLaunchKernelEx | 1.181 |
| 19 | aten::copy_ | 0.961 |
| 20 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params) | 0.840 |
| 21 | Memcpy DtoD (Device -> Device) | 0.696 |
| 22 | void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float) | 0.612 |
| 23 | void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1>(cutlass_80_simt_sgemm_32x128_8x5_tn_align1::Params) | 0.540 |
| 24 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 0.469 |
| 25 | aten::native_layer_norm | 0.371 |

## Aggregate KPIs (Self CUDA, all profiler rows)

| Bucket | Self (ms) | % of total |
| --- | ---: | ---: |
| gemm | 267.201 | 76.5 |
| attention | 41.347 | 11.8 |
| elementwise | 30.588 | 8.8 |
| roll_pad_layout | 4.140 | 1.2 |
| other | 3.238 | 0.9 |
| copy_layout | 1.444 | 0.4 |
| memcpy | 0.744 | 0.2 |
| layer_norm | 0.743 | 0.2 |

- **Total** self cuda: 349.45 ms


## Full profiler table

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         4.28%       6.815ms         7.30%      11.617ms      23.805us     120.968ms        68.34%     120.968ms     247.886us           0 B           0 B       3.61 GB       3.61 GB           488  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us     118.147ms        66.74%     118.147ms     304.503us           0 B           0 B           0 B           0 B           388  
                     aten::_efficient_attention_forward         0.37%     590.772us         1.56%       2.480ms      31.006us      20.674ms        11.68%      20.674ms     258.421us       1.25 KB          80 B     490.77 MB           0 B            80  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us      20.674ms        11.68%      20.674ms     258.421us           0 B           0 B           0 B           0 B            80  
                                               aten::mm         2.39%       3.799ms         4.02%       6.397ms      18.596us      12.758ms         7.21%      12.758ms      37.086us           0 B           0 B       2.00 GB       2.00 GB           344  
                                              aten::add         1.38%       2.192ms         2.32%       3.697ms      10.158us       8.732ms         4.93%       8.732ms      23.989us       1.62 KB       1.62 KB       2.58 GB       2.58 GB           364  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.732ms         4.93%       8.732ms      26.951us           0 B           0 B           0 B           0 B           324  
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       6.434ms         3.63%       6.434ms     402.141us           0 B           0 B           0 B           0 B            16  
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x...         0.00%       0.000us         0.00%       0.000us       0.000us       4.077ms         2.30%       4.077ms      31.849us           0 B           0 B           0 B           0 B           128  
                                              aten::mul         0.93%       1.487ms         1.52%       2.422ms      14.079us       3.378ms         1.91%       3.378ms      19.642us           0 B           0 B       1.86 GB       1.86 GB           172  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.367ms         1.90%       3.367ms      20.532us           0 B           0 B           0 B           0 B           164  
                                             aten::gelu         0.34%     544.430us         0.59%     933.220us      11.665us       2.845ms         1.61%       2.845ms      35.566us           0 B           0 B       1.38 GB       1.38 GB            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.845ms         1.61%       2.845ms      35.566us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       2.634ms         1.49%       2.634ms      27.435us           0 B           0 B           0 B           0 B            96  
                             _roll_pad_partition_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.535ms         1.43%       2.535ms      31.692us           0 B           0 B           0 B           0 B            80  
                                     _adaln_film_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       1.637ms         0.92%       1.637ms      10.231us           0 B           0 B           0 B           0 B           160  
                              _crop_roll_unmerge_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       1.605ms         0.91%       1.605ms      20.056us           0 B           0 B           0 B           0 B            80  
                                       cuLaunchKernelEx         1.37%       2.183ms         1.37%       2.183ms       6.820us       1.181ms         0.67%       1.181ms       3.691us           0 B           0 B           0 B           0 B           320  
                                            aten::copy_         0.21%     329.187us         0.76%       1.210ms      15.125us     960.883us         0.54%     960.883us      12.011us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_...         0.00%       0.000us         0.00%       0.000us       0.000us     839.572us         0.47%     839.572us     209.893us           0 B           0 B           0 B           0 B             4  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     695.501us         0.39%     695.501us       3.105us           0 B           0 B           0 B           0 B           224  
void gemv2T_kernel_val<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us     611.511us         0.35%     611.511us       4.777us           0 B           0 B           0 B           0 B           128  
void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8...         0.00%       0.000us         0.00%       0.000us       0.000us     539.959us         0.31%     539.959us      16.874us           0 B           0 B           0 B           0 B            32  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     468.542us         0.26%     468.542us      29.284us           0 B           0 B           0 B           0 B            16  
                                aten::native_layer_norm         0.13%     212.252us         0.26%     410.069us      25.629us     371.422us         0.21%     371.422us      23.214us           0 B           0 B     104.89 MB           0 B            16  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     371.422us         0.21%     371.422us      23.214us           0 B           0 B           0 B           0 B            16  
                                             aten::silu         0.68%       1.076ms         1.25%       1.987ms      12.118us     289.526us         0.16%     289.526us       1.765us           0 B           0 B     164.00 KB     164.00 KB           164  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     289.526us         0.16%     289.526us       1.765us           0 B           0 B           0 B           0 B           164  
                                              aten::cat         0.23%     361.907us         0.35%     554.745us      11.557us     152.702us         0.09%     152.702us       3.181us       3.12 KB       3.12 KB      64.01 MB      64.01 MB            48  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us     147.582us         0.08%     147.582us      36.895us           0 B           0 B           0 B           0 B             4  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 159.193ms
Self CUDA time total: 177.020ms
```

