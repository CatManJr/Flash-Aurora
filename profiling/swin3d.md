# Swin3DTransformerBackbone profiling

- Generated: 2026-03-26T21:57:21
- Torch: 2.10.0+cu128
- Config: batch=1, patch_res=(4, 32, 64), L=8192, autocast_backbone=False

## Timer

GPU: 230.54 ms for 4 forwards → 57.63 ms/forward

## Top operators

| Rank | Operator | Self (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 143.139 |
| 2 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 139.866 |
| 3 | aten::_efficient_attention_forward | 24.304 |
| 4 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 24.304 |
| 5 | aten::mm | 15.329 |
| 6 | aten::add | 11.909 |
| 7 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 9.807 |
| 8 | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) | 7.840 |
| 9 | aten::copy_ | 6.147 |
| 10 | aten::mul | 5.691 |
| 11 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 5.606 |
| 12 | aten::roll | 4.848 |
| 13 | void at::native::roll_cuda_kernel<float>(float const*, float*, long, long, long, long, long, long) | 4.848 |
| 14 | void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Params) | 4.699 |
| 15 | aten::native_layer_norm | 3.976 |
| 16 | void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*) | 3.976 |
| 17 | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 3.902 |
| 18 | aten::gelu | 3.233 |
| 19 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 3.233 |
| 20 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params) | 3.127 |
| 21 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1.849 |
| 22 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 1.776 |
| 23 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params) | 1.053 |
| 24 | Memcpy DtoD (Device -> Device) | 0.792 |
| 25 | void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float) | 0.720 |

## Full profiler table

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         3.40%       7.141ms         5.88%      12.343ms      25.293us     143.139ms        65.10%     143.139ms     293.317us           0 B           0 B       3.62 GB       3.62 GB           488  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us     139.866ms        63.61%     139.866ms     360.479us           0 B           0 B           0 B           0 B           388  
                     aten::_efficient_attention_forward         0.27%     563.434us         1.14%       2.385ms      29.816us      24.304ms        11.05%      24.304ms     303.801us       1.25 KB          88 B     492.28 MB           0 B            80  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us      24.304ms        11.05%      24.304ms     303.801us           0 B           0 B           0 B           0 B            80  
                                               aten::mm         1.70%       3.564ms         2.84%       5.959ms      17.322us      15.329ms         6.97%      15.329ms      44.562us           0 B           0 B       2.03 GB       2.03 GB           344  
                                              aten::add         1.55%       3.250ms         2.86%       6.009ms       8.784us      11.909ms         5.42%      11.909ms      17.411us       1.61 KB       1.61 KB       3.32 GB       3.32 GB           684  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       9.807ms         4.46%       9.807ms      30.268us           0 B           0 B           0 B           0 B           324  
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       7.840ms         3.57%       7.840ms     490.027us           0 B           0 B           0 B           0 B            16  
                                            aten::copy_         0.78%       1.629ms         2.21%       4.648ms      11.621us       6.147ms         2.80%       6.147ms      15.367us           0 B           0 B           0 B           0 B           400  
                                              aten::mul         1.09%       2.284ms         1.81%       3.801ms      11.450us       5.691ms         2.59%       5.691ms      17.143us           0 B           0 B       2.59 GB       2.59 GB           332  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       5.606ms         2.55%       5.606ms      16.683us           0 B           0 B           0 B           0 B           336  
                                             aten::roll         0.45%     940.658us         3.34%       7.005ms      17.513us       4.848ms         2.20%      13.846ms      34.615us           0 B           0 B       1.76 GB    -909.50 MB           400  
void at::native::roll_cuda_kernel<float>(float const...         0.00%       0.000us         0.00%       0.000us       0.000us       4.848ms         2.20%       4.848ms      20.198us           0 B           0 B           0 B           0 B           240  
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x...         0.00%       0.000us         0.00%       0.000us       0.000us       4.699ms         2.14%       4.699ms      36.709us           0 B           0 B           0 B           0 B           128  
                                aten::native_layer_norm         0.72%       1.516ms         1.77%       3.718ms      21.125us       3.976ms         1.81%       3.976ms      22.592us           0 B           0 B     826.02 MB           0 B           176  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us       3.976ms         1.81%       3.976ms      22.592us           0 B           0 B           0 B           0 B           176  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.902ms         1.77%       3.902ms      23.794us           0 B           0 B           0 B           0 B           164  
                                             aten::gelu         0.23%     488.882us         0.41%     864.190us      10.802us       3.233ms         1.47%       3.233ms      40.417us           0 B           0 B       1.38 GB       1.38 GB            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.233ms         1.47%       3.233ms      40.417us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       3.127ms         1.42%       3.127ms      32.571us           0 B           0 B           0 B           0 B            96  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.849ms         0.84%       1.849ms      11.555us           0 B           0 B           0 B           0 B           160  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.776ms         0.81%       1.776ms      11.103us           0 B           0 B           0 B           0 B           160  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.053ms         0.48%       1.053ms     263.310us           0 B           0 B           0 B           0 B             4  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     791.707us         0.36%     791.707us       3.534us           0 B           0 B           0 B           0 B           224  
void gemv2T_kernel_val<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us     720.315us         0.33%     720.315us       5.627us           0 B           0 B           0 B           0 B           128  
                                            aten::fill_         0.14%     288.994us         1.52%       3.183ms      37.889us     644.893us         0.29%     646.556us       7.697us           0 B           0 B           0 B           0 B            84  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     644.893us         0.29%     644.893us       7.677us           0 B           0 B           0 B           0 B            84  
void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8...         0.00%       0.000us         0.00%       0.000us       0.000us     624.441us         0.28%     624.441us      19.514us           0 B           0 B           0 B           0 B            32  
                                             aten::silu         0.55%       1.148ms         0.96%       2.013ms      12.273us     350.656us         0.16%     350.656us       2.138us           0 B           0 B     164.00 KB     164.00 KB           164  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     350.656us         0.16%     350.656us       2.138us           0 B           0 B           0 B           0 B           164  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 209.853ms
Self CUDA time total: 219.864ms
```

