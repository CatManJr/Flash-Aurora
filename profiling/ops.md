# Aurora end-to-end profiling report

- Generated: 2026-03-26T21:41:01
- PyTorch: 2.10.0+cu128
- CUDA (PyTorch): 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU

## Run configuration

```text
batch_size=1
synthetic=False
repeat=1
rollout_steps=2
forward_only=False
autocast_backbone=True
device=cuda
```

## GPU / CPU timer

GPU timer: 701.96 ms / 2 forwards = 350.98 ms/forward

Profiler window: `1× run_once` ≈ **2** model forwards.

- CUDA allocated after warmup: **533.1 MB**

- Peak CUDA memory (this process): **2655.6 MB**

## Top operators (Self time)

| Rank | Operator | Self time (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 239.069 |
| 2 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8::Params) | 140.586 |
| 3 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 134.435 |
| 4 | aten::_efficient_attention_forward | 99.963 |
| 5 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 85.283 |
| 6 | aten::mm | 76.840 |
| 7 | aten::copy_ | 72.538 |
| 8 | aten::add | 68.620 |
| 9 | Command Buffer Full | 46.179 |
| 10 | aten::mul | 27.719 |
| 11 | aten::roll | 25.862 |
| 12 | aten::native_layer_norm | 24.670 |
| 13 | void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*) | 24.670 |
| 14 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) | 22.921 |
| 15 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 22.432 |
| 16 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 21.571 |
| 17 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params) | 20.140 |
| 18 | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 19.894 |
| 19 | aten::gelu | 19.399 |
| 20 | fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 14.681 |
| 21 | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 14.489 |
| 22 | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 14.451 |
| 23 | aten::_flash_attention_forward | 13.945 |
| 24 | void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::bfloat16_t> >, false, false, false, false, false, true, false, false>(pytorch_flash::Flash_fwd_params) | 13.945 |
| 25 | void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8::Params) | 13.831 |

## PyTorch profiler table (full text)

Same columns as the terminal table (may be wide).

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         0.44%       3.099ms         1.67%      11.868ms      42.084us     239.069ms        34.62%     249.336ms     884.168us           0 B           0 B      10.41 GB      10.41 GB           282  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us     140.586ms        20.36%     140.586ms     976.294us           0 B           0 B           0 B           0 B           144  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us     134.435ms        19.47%     134.435ms       3.538ms           0 B           0 B           0 B           0 B            38  
                     aten::_efficient_attention_forward         0.03%     188.067us         0.45%       3.185ms     132.702us      99.963ms        14.48%     102.324ms       4.264ms         384 B          40 B       1.13 GB           0 B            24  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us      85.283ms        12.35%      85.283ms      21.321ms           0 B           0 B           0 B           0 B             4  
                                               aten::mm         0.20%       1.391ms         0.51%       3.651ms      19.631us      76.840ms        11.13%      79.228ms     425.957us           0 B           0 B       6.62 GB       6.62 GB           186  
                                            aten::copy_         1.85%      13.167ms        55.44%     394.769ms     321.998us      72.538ms        10.50%      79.609ms      64.934us           0 B           0 B           0 B           0 B          1226  
                                              aten::add         0.19%       1.349ms         1.58%      11.218ms      28.617us      68.620ms         9.94%      78.221ms     199.544us       1.02 KB       1.02 KB      13.05 GB      13.05 GB           392  
                                    Command Buffer Full         7.30%      51.975ms         7.30%      51.975ms     288.750us      46.179ms         6.69%      46.179ms     256.552us           0 B           0 B           0 B           0 B           180  
                                              aten::mul         0.13%     930.505us         0.74%       5.302ms      23.054us      27.719ms         4.01%      32.431ms     141.003us          -8 B          -8 B       7.37 GB       7.37 GB           230  
                                             aten::roll         0.05%     375.849us         1.17%       8.361ms      41.803us      25.862ms         3.75%      84.541ms     422.706us           0 B           0 B       6.38 GB      -2.79 GB           200  
                                aten::native_layer_norm         0.07%     511.363us         0.66%       4.678ms      47.739us      24.670ms         3.57%      28.754ms     293.406us           0 B           0 B       5.18 GB           0 B            98  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      24.670ms         3.57%      24.670ms     251.737us           0 B           0 B           0 B           0 B            98  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      22.921ms         3.32%      22.921ms     286.518us           0 B           0 B           0 B           0 B            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      22.432ms         3.25%      22.432ms     273.562us           0 B           0 B           0 B           0 B            82  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      21.571ms         3.12%      21.571ms     138.276us           0 B           0 B           0 B           0 B           156  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      20.140ms         2.92%      20.140ms     774.616us           0 B           0 B           0 B           0 B            26  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      19.894ms         2.88%      19.894ms      24.805us           0 B           0 B           0 B           0 B           802  
                                             aten::gelu         0.03%     199.199us         0.08%     586.825us      12.757us      19.399ms         2.81%      19.989ms     434.540us           0 B           0 B       5.04 GB       5.04 GB            46  
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemE...         0.00%       0.000us         0.00%       0.000us       0.000us      14.681ms         2.13%      14.681ms     734.036us           0 B           0 B           0 B           0 B            20  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      14.489ms         2.10%      14.489ms     181.107us           0 B           0 B           0 B           0 B            80  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      14.451ms         2.09%      14.451ms     180.632us           0 B           0 B           0 B           0 B            80  
                         aten::_flash_attention_forward         0.02%     148.242us         0.06%     460.149us      23.007us      13.945ms         2.02%      13.945ms     697.260us           0 B           0 B     508.80 MB           0 B            20  
void pytorch_flash::flash_fwd_kernel<Flash_fwd_kerne...         0.00%       0.000us         0.00%       0.000us       0.000us      13.945ms         2.02%      13.945ms     697.260us           0 B           0 B           0 B           0 B            20  
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...         0.00%       0.000us         0.00%       0.000us       0.000us      13.831ms         2.00%      13.831ms     108.052us           0 B           0 B           0 B           0 B           128  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 712.098ms
Self CUDA time total: 690.531ms
```

## Notes

Profiler Self CPU/CUDA totals sum over **all** recorded ops in this window (1× run_once ≈ 2 forwards). Divide by that forward count for a rough per-forward share, or use `--repeat 1`.

## Artifacts

- Plot: `/home/catmanjr/projects/triton_dev/profiling/ops.png`

