# Aurora end-to-end profiling report

- Generated: 2026-03-26T21:38:34
- PyTorch: 2.10.0+cu128
- CUDA (PyTorch): 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU

## Run configuration

```text
batch_size=4 (--stress default)
synthetic=False
repeat=1
rollout_steps=2
forward_only=False
autocast_backbone=True
device=cuda
```

## GPU / CPU timer

GPU timer: 5725.12 ms / 2 forwards = 2862.56 ms/forward

Profiler window: `1× run_once` ≈ **2** model forwards.

- CUDA allocated after warmup: **533.1 MB**

- Peak CUDA memory (this process): **9462.2 MB**

## Top operators (Self time)

| Rank | Operator | Self time (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 3907.220 |
| 2 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 3607.089 |
| 3 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8::Params) | 411.653 |
| 4 | aten::_efficient_attention_forward | 303.570 |
| 5 | aten::copy_ | 276.157 |
| 6 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 259.653 |
| 7 | aten::add | 252.791 |
| 8 | aten::mm | 242.074 |
| 9 | aten::gelu | 193.759 |
| 10 | Command Buffer Full | 159.624 |
| 11 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 144.878 |
| 12 | aten::mul | 108.544 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 90.104 |
| 14 | aten::roll | 87.224 |
| 15 | aten::native_layer_norm | 85.308 |
| 16 | void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*) | 85.308 |
| 17 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) | 84.636 |
| 18 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 82.496 |
| 19 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8::Params) | 70.230 |
| 20 | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 67.157 |
| 21 | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 55.072 |
| 22 | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 52.271 |
| 23 | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 52.239 |
| 24 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) | 48.881 |
| 25 | void at::native::roll_cuda_kernel<float>(float const*, float*, long, long, long, long, long, long) | 45.827 |

## PyTorch profiler table (full text)

Same columns as the terminal table (may be wide).

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         0.07%       3.815ms         0.45%      25.332ms      92.452us        3.907s        70.28%        3.943s      14.391ms           0 B           0 B      41.38 GB      41.38 GB           274  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us        3.607s        64.88%        3.607s      94.923ms           0 B           0 B           0 B           0 B            38  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us     411.653ms         7.40%     411.653ms       2.820ms           0 B           0 B           0 B           0 B           146  
                     aten::_efficient_attention_forward         0.00%     255.700us         0.08%       4.695ms     167.669us     303.570ms         5.46%     311.141ms      11.112ms         448 B          64 B       4.52 GB           0 B            28  
                                            aten::copy_         0.71%      39.581ms        23.41%        1.312s       1.062ms     276.157ms         4.97%     303.980ms     245.939us           0 B           0 B           0 B           0 B          1236  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us     259.653ms         4.67%     259.653ms      32.457ms           0 B           0 B           0 B           0 B             8  
                                              aten::add         0.03%       1.618ms         0.61%      34.164ms      87.153us     252.791ms         4.55%     280.392ms     715.286us       1.05 KB       1.05 KB      52.17 GB      52.17 GB           392  
                                               aten::mm         0.04%       2.107ms         0.40%      22.467ms     115.811us     242.074ms         4.35%     254.637ms       1.313ms           0 B           0 B      26.50 GB      26.50 GB           194  
                                             aten::gelu         0.00%     269.655us         0.06%       3.123ms      67.891us     193.759ms         3.49%     200.347ms       4.355ms           0 B           0 B      20.15 GB      20.15 GB            46  
                                    Command Buffer Full         4.08%     228.794ms         4.08%     228.794ms       1.144ms     159.624ms         2.87%     159.624ms     798.119us           0 B           0 B           0 B           0 B           200  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     144.878ms         2.61%     144.878ms      18.110ms           0 B           0 B           0 B           0 B             8  
                                              aten::mul         0.02%       1.191ms         0.28%      15.843ms      68.881us     108.544ms         1.95%     120.564ms     524.191us         -16 B         -16 B      29.21 GB      29.21 GB           230  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      90.104ms         1.62%      90.104ms     446.061us           0 B           0 B           0 B           0 B           202  
                                             aten::roll         0.01%     458.989us         0.60%      33.497ms     167.483us      87.224ms         1.57%     269.052ms       1.345ms           0 B           0 B      25.18 GB     -11.75 GB           200  
                                aten::native_layer_norm         0.01%     628.343us         0.39%      22.022ms     224.714us      85.308ms         1.53%      98.941ms       1.010ms           0 B           0 B      20.71 GB           0 B            98  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      85.308ms         1.53%      85.308ms     870.489us           0 B           0 B           0 B           0 B            98  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      84.636ms         1.52%      84.636ms       1.058ms           0 B           0 B           0 B           0 B            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      82.496ms         1.48%      82.496ms       1.006ms           0 B           0 B           0 B           0 B            82  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      70.230ms         1.26%      70.230ms     949.060us           0 B           0 B           0 B           0 B            74  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      67.157ms         1.21%      67.157ms      83.737us           0 B           0 B           0 B           0 B           802  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      55.072ms         0.99%      55.072ms     688.400us           0 B           0 B           0 B           0 B            80  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      52.271ms         0.94%      52.271ms     653.384us           0 B           0 B           0 B           0 B            80  
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      52.239ms         0.94%      52.239ms     652.991us           0 B           0 B           0 B           0 B            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      48.881ms         0.88%      48.881ms       1.222ms           0 B           0 B           0 B           0 B            40  
void at::native::roll_cuda_kernel<float>(float const...         0.00%       0.000us         0.00%       0.000us       0.000us      45.827ms         0.82%      45.827ms     763.783us           0 B           0 B           0 B           0 B            60  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 5.607s
Self CUDA time total: 5.559s
```

## Notes

Profiler Self CPU/CUDA totals sum over **all** recorded ops in this window (1× run_once ≈ 2 forwards). Divide by that forward count for a rough per-forward share, or use `--repeat 1`.

## Artifacts

- Plot: `/home/catmanjr/projects/triton_dev/profiling/ops.png`

