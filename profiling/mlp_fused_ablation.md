# Fused MLP (Triton) ablation

- Generated: 2026-03-27T00:59:47
- PyTorch: 2.10.0+cu128
- Preset: `unit_mlp` → D=128, H=512, M=64
- Warmup: 10, repeat: 50

## Numerical check

- max |torch − triton| = `2.980232e-07`

## Timing (median of CUDA event intervals, ms/iter)

| Variant | ms/iter | peak CUDA MB | vs torch |
| --- | ---: | ---: | ---: |
| torch_fc1_gelu_fc2 | 0.0624 | 10.5 |  |
| triton_mlp_fused_ieee | 0.1595 | 10.2 | 0.391x |

### torch_fc1_gelu_fc2

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         9.15%     357.609us        74.69%       2.920ms     291.973us     117.985us        93.91%     127.908us      12.791us            10  
void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x...         0.00%       0.000us         0.00%       0.000us       0.000us      95.611us        76.10%      95.611us       9.561us            10  
void cublasLt::splitKreduce_kernel<32, 16, int, floa...         0.00%       0.000us         0.00%       0.000us       0.000us      22.374us        17.81%      22.374us       2.237us            10  
                                Activity Buffer Request        59.91%       2.342ms        59.91%       2.342ms       2.342ms       9.923us         7.90%       9.923us       9.923us             1  
                                             aten::gelu         3.89%     152.000us         6.42%     250.847us      50.169us       7.650us         6.09%       7.650us       1.530us             5  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       7.650us         6.09%       7.650us       1.530us             5  
                                           aten::linear        15.64%     611.435us        93.41%       3.651ms     365.142us       0.000us         0.00%     127.908us      12.791us            10  
                                                aten::t         1.55%      60.584us         3.08%     120.257us      12.026us       0.000us         0.00%       0.000us       0.000us            10  
                                        aten::transpose         0.98%      38.324us         1.53%      59.673us       5.967us       0.000us         0.00%       0.000us       0.000us            10  
                                       aten::as_strided         0.55%      21.349us         0.55%      21.349us       2.135us       0.000us         0.00%       0.000us       0.000us            10  
                                 cudaDeviceGetAttribute         0.15%       5.711us         0.15%       5.711us       0.571us       0.000us         0.00%       0.000us       0.000us            10  
                                         cuLaunchKernel         3.54%     138.484us         3.54%     138.484us      13.848us       0.000us         0.00%       0.000us       0.000us            10  
                                       cudaLaunchKernel         4.47%     174.721us         4.47%     174.721us      11.648us       0.000us         0.00%       0.000us       0.000us            15  
                                  cudaDeviceSynchronize         0.17%       6.723us         0.17%       6.723us       3.361us       0.000us         0.00%       0.000us       0.000us             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.909ms
Self CUDA time total: 125.635us

```

### triton_mlp_fused

```text
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
      _fused_mlp_row_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     552.876us       100.00%     552.876us     110.575us             5  
              aten::reshape         0.60%      18.074us         1.65%      49.665us       4.967us       0.000us         0.00%       0.000us       0.000us            10  
                 aten::view         1.05%      31.591us         1.05%      31.591us       3.159us       0.000us         0.00%       0.000us       0.000us            10  
                aten::empty         1.43%      42.902us         1.43%      42.902us       8.580us       0.000us         0.00%       0.000us       0.000us             5  
    Activity Buffer Request        79.33%       2.382ms        79.33%       2.382ms       2.382ms       0.000us         0.00%       0.000us       0.000us             1  
           cuLaunchKernelEx         2.93%      87.958us         2.93%      87.958us      17.592us       0.000us         0.00%       0.000us       0.000us             5  
      cudaDeviceSynchronize        14.66%     440.146us        14.66%     440.146us     220.073us       0.000us         0.00%       0.000us       0.000us             2  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.003ms
Self CUDA time total: 552.876us

```

