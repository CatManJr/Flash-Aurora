# Swin3DTransformerBackbone profiling

- Generated: 2026-03-26T22:27:00
- Torch: 2.10.0+cu128
- Config: preset=baseline, batch=1, patch_res=(4, 32, 64), L=8192, autocast_backbone=False, use_triton_layout=False, use_triton_adaln=False

## Timer

GPU: 253.50 ms for 4 forwards → 63.37 ms/forward

## Top operators

| Rank | Operator | Self (ms) |
| ---: | --- | ---: |
| 1 | aten::addmm | 94.873 |
| 2 | void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) | 92.416 |
| 3 | aten::_efficient_attention_forward | 15.771 |
| 4 | fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) | 15.771 |
| 5 | aten::add | 10.377 |
| 6 | aten::mm | 9.550 |
| 7 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 9.091 |
| 8 | void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) | 5.195 |
| 9 | aten::copy_ | 4.190 |
| 10 | aten::mul | 4.175 |
| 11 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 3.760 |
| 12 | void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 3.050 |
| 13 | aten::roll | 3.041 |
| 14 | void at::native::roll_cuda_kernel<float>(float const*, float*, long, long, long, long, long, long) | 3.041 |
| 15 | aten::gelu | 2.964 |
| 16 | void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 2.964 |
| 17 | void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Params) | 2.926 |
| 18 | aten::native_layer_norm | 2.450 |
| 19 | void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float const*, float*, float*, float*) | 2.450 |
| 20 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutlass_80_simt_sgemm_128x64_8x5_tn_align1::Params) | 1.998 |
| 21 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1.127 |
| 22 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 1.116 |
| 23 | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params) | 0.630 |
| 24 | Memcpy DtoD (Device -> Device) | 0.597 |
| 25 | void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float) | 0.508 |

## Aggregate KPIs (Self CUDA, all profiler rows)

| Bucket | Self (ms) | % of total |
| --- | ---: | ---: |
| gemm | 208.617 | 70.3 |
| elementwise | 35.994 | 12.1 |
| attention | 31.542 | 10.6 |
| copy_layout | 7.962 | 2.7 |
| roll_pad_layout | 6.082 | 2.1 |
| layer_norm | 4.901 | 1.7 |
| other | 0.795 | 0.3 |
| memcpy | 0.662 | 0.2 |

- **Total** self cuda: 296.55 ms

## Stage-A execution (baseline vs Triton)

Run a single command to execute Phase-A acceptance (same shape/config, baseline then Triton):

```bash
uv run python aurora/profiling_swin3d.py \
  --preset baseline \
  --repeat 4 \
  --compare-triton \
  --compare-report-out profiling/swin3d_stageA_compare.md
```

Acceptance focus:

- Primary KPI: `ms/forward` speedup from the compare summary.
- Bucket deltas (all profiler rows): `copy_layout`, `roll_pad_layout`, `layer_norm`.
- Keep SDPA path unchanged (`scaled_dot_product_attention` still handled by PyTorch backends).

## Stage-B quick probe (MLP boundary)

To evaluate `fc1 -> GELU -> fc2` on inference `dropout=0`, enable MLP Triton GELU on top of
Stage-A compare:

```bash
uv run python aurora/profiling_swin3d.py \
  --preset baseline \
  --repeat 4 \
  --compare-triton \
  --use-triton-mlp \
  --compare-report-out profiling/swin3d_stageB_compare.md
```

Interpretation:

- This keeps GEMM on cuBLAS (`Linear`/`addmm` unchanged).
- It only swaps GELU to Triton in CUDA float32 eval path.
- Compare Stage-A report vs Stage-B report to see whether MLP boundary still has measurable gap.

## Stage-C full coverage probe (LoRA merge)

To evaluate full-scenario LoRA merged inference (`single/from_second/all` rollout modes), run:

```bash
uv run python aurora/profiling_swin3d.py \
  --preset baseline \
  --repeat 4 \
  --compare-stagec \
  --use-triton-mlp \
  --compare-report-out profiling/swin3d_stageC_compare.md
```

Without a finetuned checkpoint, LoRA defaults to `lora_B=0` (no ΔW). For a non-trivial LoRA delta stress test, add:

```bash
  --randomize-lora --lora-random-seed 0
```

Interpretation:

- Run 1: baseline (no Triton, no merge).
- Run 2: Stage-A/B (`layout+AdaLN`, optional `MLP-GELU`).
- Run 3: Stage-C = Stage-A/B + `use_lora_merged_inference`.
- Look at both `ms/forward` and `Stage-C vs Stage-A/B aggregate delta`.
- Report includes `aten::addmm` call counts for merge verification.


## Full profiler table

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         4.99%       7.059ms         8.60%      12.178ms      24.954us      94.873ms        63.98%      94.873ms     194.411us           0 B           0 B       3.62 GB       3.62 GB           488  
void magma_sgemmEx_kernel<float, float, float, true,...         0.00%       0.000us         0.00%       0.000us       0.000us      92.416ms        62.33%      92.416ms     238.186us           0 B           0 B           0 B           0 B           388  
                     aten::_efficient_attention_forward         0.44%     616.468us         2.18%       3.081ms      38.516us      15.771ms        10.64%      15.771ms     197.138us       1.25 KB          56 B     502.53 MB           0 B            80  
fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us      15.771ms        10.64%      15.771ms     197.138us           0 B           0 B           0 B           0 B            80  
                                              aten::add         2.55%       3.609ms         4.65%       6.582ms       9.622us      10.377ms         7.00%      10.377ms      15.171us       1.62 KB       1.62 KB       3.34 GB       3.34 GB           684  
                                               aten::mm         2.67%       3.784ms         4.43%       6.267ms      18.218us       9.550ms         6.44%       9.550ms      27.762us           0 B           0 B       2.04 GB       2.04 GB           344  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       9.091ms         6.13%       9.091ms      28.059us           0 B           0 B           0 B           0 B           324  
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...         0.00%       0.000us         0.00%       0.000us       0.000us       5.195ms         3.50%       5.195ms     324.708us           0 B           0 B           0 B           0 B            16  
                                            aten::copy_         1.15%       1.633ms         3.29%       4.665ms      11.661us       4.190ms         2.83%       4.190ms      10.475us           0 B           0 B           0 B           0 B           400  
                                              aten::mul         1.76%       2.488ms         2.94%       4.170ms      12.559us       4.175ms         2.82%       4.175ms      12.576us           0 B           0 B       2.60 GB       2.60 GB           332  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       3.760ms         2.54%       3.760ms      11.190us           0 B           0 B           0 B           0 B           336  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.050ms         2.06%       3.050ms      18.598us           0 B           0 B           0 B           0 B           164  
                                             aten::roll         0.72%       1.018ms         5.08%       7.192ms      17.981us       3.041ms         2.05%       8.673ms      21.682us           0 B           0 B       1.78 GB    -907.41 MB           400  
void at::native::roll_cuda_kernel<float>(float const...         0.00%       0.000us         0.00%       0.000us       0.000us       3.041ms         2.05%       3.041ms      12.671us           0 B           0 B           0 B           0 B           240  
                                             aten::gelu         0.37%     522.303us         0.65%     924.518us      11.556us       2.964ms         2.00%       2.964ms      37.044us           0 B           0 B       1.38 GB       1.38 GB            80  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.964ms         2.00%       2.964ms      37.044us           0 B           0 B           0 B           0 B            80  
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x...         0.00%       0.000us         0.00%       0.000us       0.000us       2.926ms         1.97%       2.926ms      22.861us           0 B           0 B           0 B           0 B           128  
                                aten::native_layer_norm         1.31%       1.850ms         3.07%       4.353ms      24.732us       2.450ms         1.65%       2.450ms      13.923us           0 B           0 B     813.27 MB           0 B           176  
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us       2.450ms         1.65%       2.450ms      13.923us           0 B           0 B           0 B           0 B           176  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8...         0.00%       0.000us         0.00%       0.000us       0.000us       1.998ms         1.35%       1.998ms      20.814us           0 B           0 B           0 B           0 B            96  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.127ms         0.76%       1.127ms       7.042us           0 B           0 B           0 B           0 B           160  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.116ms         0.75%       1.116ms       6.976us           0 B           0 B           0 B           0 B           160  
void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_...         0.00%       0.000us         0.00%       0.000us       0.000us     629.770us         0.42%     629.770us     157.442us           0 B           0 B           0 B           0 B             4  
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     596.645us         0.40%     596.645us       2.664us           0 B           0 B           0 B           0 B           224  
void gemv2T_kernel_val<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us     508.331us         0.34%     508.331us       3.971us           0 B           0 B           0 B           0 B           128  
                                            aten::fill_         0.21%     298.006us         2.28%       3.231ms      38.462us     436.359us         0.29%     437.959us       5.214us           0 B           0 B           0 B           0 B            84  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     436.359us         0.29%     436.359us       5.195us           0 B           0 B           0 B           0 B            84  
void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8...         0.00%       0.000us         0.00%       0.000us       0.000us     367.693us         0.25%     367.693us      11.490us           0 B           0 B           0 B           0 B            32  
                                             aten::silu         0.71%       1.011ms         1.36%       1.931ms      11.773us     225.732us         0.15%     225.732us       1.376us           0 B           0 B     164.00 KB     164.00 KB           164  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     225.732us         0.15%     225.732us       1.376us           0 B           0 B           0 B           0 B           164  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 141.598ms
Self CUDA time total: 148.276ms
```

