# Swin3DTransformerBlock profiling

- Generated: 2026-03-26T23:48:43
- Torch: 2.10.0+cu128
- Config: preset=small, dim=256, heads=4, patch_res=(4, 32, 64), window=(2, 6, 12), shift=(0, 0, 0), autocast=False, compile=False, Triton layout/AdaLN/MLP=False/False/False

## Timer

GPU: 22.87 ms for 8 iters -> 2.859 ms/iter

## Bottleneck buckets (block-local)

| Bucket | Self (ms) | % |
| --- | ---: | ---: |
| GEMM (Linear / matmul) | 13.795 | 61.1 |
| attention (SDPA / FMHA) | 3.877 | 17.2 |
| elementwise (other) | 2.017 | 8.9 |
| copy / scatter | 0.954 | 4.2 |
| GELU | 0.948 | 4.2 |
| LayerNorm | 0.825 | 3.7 |
| other | 0.106 | 0.5 |
| SiLU (AdaLN modulation MLP) | 0.035 | 0.2 |
| memcpy | 0.026 | 0.1 |

## Focus

- ATen `scaled_dot_product*` / `_efficient_attention_forward`: calls≈24, self-time≈3.877 ms
- `aten::addmm`: calls=48, self-time≈13.795 ms

## Top operators

| Rank | Operator | Self (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 13.795 |
| 2 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 13.706 |
| 3 | aten::_efficient_attention_forward | 3.877 |
| 4 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 3.877 |
| 5 | aten::add | 1.718 |
| 6 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul>) | 1.054 |
| 7 | aten::copy_ | 0.954 |
| 8 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 0.954 |
| 9 | aten::gelu | 0.948 |
| 10 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 0.948 |
| 11 | aten::native_layer_norm | 0.825 |
| 12 | void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*) | 0.825 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 0.347 |
| 14 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 0.317 |
| 15 | aten::mul | 0.299 |
| 16 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 0.299 |
| 17 | aten::fill_ | 0.106 |
| 18 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 0.106 |
| 19 | std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, true, false, 8, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>) | 0.063 |
| 20 | aten::silu | 0.035 |
| 21 | void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 0.035 |
| 22 | Memcpy DtoD (Device -> Device) | 0.026 |
| 23 | Activity Buffer Request | 0.015 |

## Full profiler table

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         3.58%     860.558us         6.70%       1.610ms      33.549us      13.795ms        61.15%      13.795ms     287.388us           0 B           0 B     644.03 MB     644.03 MB            48  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us      13.706ms        60.76%      13.706ms     428.301us           0 B           0 B           0 B           0 B            32  
                     aten::_efficient_attention_forward         0.42%     101.252us         3.43%     824.267us     103.033us       3.877ms        17.19%       3.877ms     484.572us         128 B           0 B      81.00 MB           0 B             8  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us       3.877ms        17.19%       3.877ms     484.572us           0 B           0 B           0 B           0 B             8  
                                              aten::add         2.10%     503.903us         3.53%     848.308us      13.255us       1.718ms         7.62%       1.718ms      26.851us           0 B           0 B     580.02 MB     580.02 MB            64  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.054ms         4.67%       1.054ms      32.953us           0 B           0 B           0 B           0 B            32  
                                            aten::copy_         0.66%     159.728us         1.92%     460.406us      14.388us     953.630us         4.23%     953.630us      29.801us           0 B           0 B           0 B           0 B            32  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     953.630us         4.23%     953.630us      29.801us           0 B           0 B           0 B           0 B            32  
                                             aten::gelu         0.16%      39.624us         0.33%      80.022us      10.003us     948.422us         4.20%     948.422us     118.553us           0 B           0 B     256.00 MB     256.00 MB             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     948.422us         4.20%     948.422us     118.553us           0 B           0 B           0 B           0 B             8  
                                aten::native_layer_norm         0.48%     114.803us         1.40%     337.179us      21.074us     825.461us         3.66%     825.461us      51.591us           0 B           0 B     129.00 MB           0 B            16  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us     825.461us         3.66%     825.461us      51.591us           0 B           0 B           0 B           0 B            16  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     347.188us         1.54%     347.188us      21.699us           0 B           0 B           0 B           0 B            16  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     316.757us         1.40%     316.757us      19.797us           0 B           0 B           0 B           0 B            16  
                                              aten::mul         0.28%      67.907us         0.57%     137.270us       8.579us     298.907us         1.33%     298.907us      18.682us           0 B           0 B     128.00 MB     128.00 MB            16  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     298.907us         1.33%     298.907us      18.682us           0 B           0 B           0 B           0 B            16  
                                            aten::fill_         0.27%      65.628us        10.59%       2.546ms     318.309us     105.777us         0.47%     120.335us      15.042us           0 B           0 B           0 B           0 B             8  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     105.777us         0.47%     105.777us      13.222us           0 B           0 B           0 B           0 B             8  
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      62.776us         0.28%      62.776us       3.924us           0 B           0 B           0 B           0 B            16  
                                             aten::silu         0.31%      74.955us         0.85%     204.629us      12.789us      35.450us         0.16%      35.450us       2.216us           0 B           0 B      16.00 KB      16.00 KB            16  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      35.450us         0.16%      35.450us       2.216us           0 B           0 B           0 B           0 B            16  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      26.240us         0.12%      26.240us       1.640us           0 B           0 B           0 B           0 B            16  
                                Activity Buffer Request         9.80%       2.355ms         9.80%       2.355ms       2.355ms      14.558us         0.06%      14.558us      14.558us           0 B           0 B           0 B           0 B             1  
                                             aten::view         0.42%     101.957us         0.42%     101.957us       0.671us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B           152  
                                              aten::pad         0.46%     111.120us        13.20%       3.174ms     396.799us       0.000us         0.00%     346.480us      43.310us           0 B           0 B      81.00 MB           0 B             8  
                                  aten::constant_pad_nd         0.74%     178.897us        12.74%       3.063ms     382.909us       0.000us         0.00%     346.480us      43.310us           0 B           0 B      81.00 MB           0 B             8  
                                            aten::empty         1.34%     323.083us         1.34%     323.083us       2.885us       0.000us         0.00%       0.000us       0.000us         128 B         128 B     517.00 MB     517.00 MB           112  
                                       cudaLaunchKernel         6.58%       1.583ms         6.58%       1.583ms       7.329us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B           216  
                                           aten::narrow         0.58%     139.281us         1.07%     257.891us       4.030us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            64  
                                            aten::slice         0.46%     110.318us         0.61%     147.127us       1.839us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            80  
                                       aten::as_strided         0.87%     209.370us         0.87%     209.370us       0.844us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B           248  
                                          aten::permute         0.21%      50.644us         0.56%     133.853us       4.183us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            32  
                                          aten::reshape         0.33%      78.925us         2.19%     525.769us      10.954us       0.000us         0.00%     727.485us      15.156us           0 B           0 B     226.00 MB           0 B            48  
                                            aten::clone         0.21%      51.671us         1.73%     415.142us      17.298us       0.000us         0.00%     727.485us      30.312us           0 B           0 B     226.00 MB           0 B            24  
                                       aten::empty_like         0.10%      23.445us         0.39%      93.446us       3.894us       0.000us         0.00%       0.000us       0.000us           0 B           0 B     226.00 MB           0 B            24  
                                     aten::_unsafe_view         0.06%      15.420us         0.06%      15.420us       0.642us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            24  
                                           aten::linear         0.51%     122.942us         7.99%       1.921ms      40.020us       0.000us         0.00%      13.795ms     287.388us           0 B           0 B     644.03 MB           0 B            48  
                                                aten::t         0.24%      57.720us         0.68%     162.810us       3.392us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            48  
                                        aten::transpose         0.48%     116.506us         0.77%     184.804us       2.310us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            80  
                                               [memory]         0.00%       0.000us         0.00%       0.000us       0.000us       0.000us         0.00%       0.000us       0.000us         -16 B         -16 B      -2.07 GB      -2.07 GB           279  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 24.041ms
Self CUDA time total: 22.557ms
```

