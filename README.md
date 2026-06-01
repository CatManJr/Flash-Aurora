# Design and Implement High-Performance Computing Framework for GFMS: MS Aurora as an example

## Install CuTe DSL & Flash Attention

```bash
git clone ...
source .venv/bin/activate
chmod +x cutlass/python/CuTeDSL/setup.sh   # 仅当没有执行权限时
# For CUDA Toolkit 12.9:
./cutlass/python/CuTeDSL/setup.sh --cu12
# For CUDA Toolkit 13.1:
./cutlass/python/CuTeDSL/setup.sh --cu13

cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```

Set `CUTE_DSL_ARCH=sm_120a` when compiling on Blackwell (SM120). Enable the CuTe path with `AURORA_CUTE_WINDOW_ATTN=1`.

## Notes (this repo)

The [`aurora/`](aurora/) tree is the [Microsoft Aurora](https://github.com/microsoft/aurora) model code with local changes for inference-oriented performance experiments.

- **Triton ops** (CUDA float32 inference paths) under [`aurora/aurora/ops/`](aurora/aurora/ops/): e.g. window **layout** only in [`triton_swin3d_layout.py`](aurora/aurora/ops/triton_swin3d_layout.py) (roll/pad/partition + inverse), **AdaLN** in `triton_adaln.py`, **GELU** in `triton_gelu.py` (not a full Swin block in one file).
- **D2 (fused AdaLN + residual)**: used from `AdaptiveLayerNorm` and `Swin3DTransformerBlock` when enabled; see tests in [`aurora/tests/test_triton_swin3d.py`](aurora/tests/test_triton_swin3d.py).
- **D3 (inference workspace pool)**: [`InferenceWorkspacePool`](aurora/aurora/model/workspace_pool.py) reuses a scratch buffer for the backbone’s final decoder `concat` (`torch.cat(..., out=buf)`), optional on [`Swin3DTransformerBackbone`](aurora/aurora/model/swin3d.py) / [`Aurora`](aurora/aurora/model/aurora.py). Tests: [`aurora/tests/test_inference_workspace_pool.py`](aurora/tests/test_inference_workspace_pool.py).
- **CuTe window attention** (SM120): [`aurora/aurora/ops/cute/`](aurora/aurora/ops/cute/) — custom CuTeDSL kernels for Swin3D window attention, wired through [`window_attn_fwd.py`](aurora/aurora/ops/cute/window_attn_fwd.py). Tests: [`aurora/tests/test_cute_window_attn.py`](aurora/tests/test_cute_window_attn.py). Benchmarks: [`benchmark/bench_window_attn.py`](benchmark/bench_window_attn.py), [`benchmark/bench_swin_block.py`](benchmark/bench_swin_block.py).
- **Profiling write-ups** (local): [`profiling/`](profiling/). Backbone **D2 vs D2+D3** (from repo root, CUDA): default `--compare-d2d3` is **light** (batch=1, repeat=4). **`--preset stress`** uses `L=2048`, batch=4 (8192 tokens/step, `warmup`/`repeat` raised). **`--preset stress-heavy`** is batch=8, `L=8192` (very large VRAM). Example:  
  `uv run python aurora/profiling_swin3d.py --compare-d2d3 --preset stress --compare-report-out profiling/swin3d_d2d3_stress.md`

## CuTe Window Attention — current state

We've been replacing Aurora's per-window SDPA with hand-written CuTeDSL kernels on Blackwell (SM120). The work lives entirely in our `ops/cute` layer — upstream Aurora torch code is untouched.

Two precision modes:

| Mode | I/O | Matmul | Kernel file |
|------|-----|--------|-------------|
| `BF16_MIXED` | BF16 | FP32 accum (`mma.sync.m16n8k16.bf16.bf16.f32`) | `_kernel_bf16.py` (v1), `_kernel_bf16_v2.py` (v2) |
| `TF32_ACC_FP32` | FP32 Q/K, BF16 V in smem | TF32 (`mma.sync.m16n8k8.tf32.tf32.f32`) | `_kernel_fp32_v2.py` |

Dispatch in `window_attn_fwd.py` picks the kernel variant by KV-pass count. Production shapes are all single-pass (`N=144`, `tile_n ≥ N`), so BF16 goes through the simpler 128-thread v1 (`cp.async`). Multi-pass cases (long sequences, streaming) use the 160-thread v2 with a dedicated DMA warp and `PipelineTmaAsync` for K/V prefetch overlap.

For TF32 single-pass we recently fused the host-side `v.to(bfloat16)` into the kernel: V stays FP32 in global memory and converts to BF16 on the gmem→register→smem load. That removed ~27% overhead on large `Bwin` and is the main reason TF32 now pulls ahead of FP32 SDPA by a wide margin.

Other small fixes along the way: `torch.zeros_like` → `torch.empty_like` for output allocation (avoided a memset that was masking kernel gains), and adaptive v1/v2 routing instead of a global env flag.

Correctness: 56/56 tests in `test_cute_window_attn.py` pass. Full Swin3D block integration (`bench_swin_block.py`) shows bitwise match against the respective baselines (BF16 OPT vs BF16 PyTorch, TF32 OPT vs FP32 strict SDPA).

### Benchmarks — attention kernel only

Hardware: **NVIDIA RTX PRO 6000 Blackwell Server Edition** (SM120).  
Run: `uv run python benchmark/bench_window_attn.py` (1000 samples, trimmed mean).

**Realistic Aurora shapes** (all single-pass, `N=144`):

| Shape | Mode | CuTe | SDPA | Speedup |
|-------|------|------|------|---------|
| Stage1 enc, Bwin=1800, H=8 | BF16 | 0.729 ms | 0.783 ms | **1.07×** |
| Stage2 enc, Bwin=450, H=16 | BF16 | 0.374 ms | 0.402 ms | **1.07×** |
| Stage3 enc, Bwin=128, H=32 | BF16 | 0.222 ms | 0.242 ms | **1.09×** |
| Stage1 enc, Bwin=1800, H=8 | TF32 | 1.642 ms | 2.583 ms | **1.57×** |
| Stage2 enc, Bwin=450, H=16 | TF32 | 0.838 ms | 1.310 ms | **1.56×** |
| Stage3 enc, Bwin=128, H=32 | TF32 | 0.490 ms | 0.762 ms | **1.56×** |

Micro-shapes (`N=144`, `H=8/16/32`) sit around 1.08–1.09× for BF16 and ~1.50× for TF32. The one outlier is `N=576` streaming (8 KV passes, `tile_n=80`) at ~0.86× — SDPA's fused kernel wins on long multi-pass sequences, and Aurora doesn't hit that shape in practice.

### Benchmarks — full Swin3D block

Run: `uv run python benchmark/bench_swin_block.py` (B=1, warmup=20, measured=100).

TF32 OPT = CuTe TF32 attention + Triton projections (`allow_tf32=True`). Compared against PyTorch FP32 with `allow_tf32=True` on linear layers (same projection speed; attention kernel differs):

| Shape | FP32 strict (µs) | FP32-TF32* (µs) | TF32 OPT (µs) | vs TF32* | vs strict |
|-------|------------------|-----------------|---------------|----------|-----------|
| Stage1 W | 42585 | 26285 | 23270 | 1.13× | 1.83× |
| Stage2 W | 33917 | 16808 | 15327 | 1.10× | 2.21× |
| Stage3 W | 31729 | 13380 | 12409 | 1.08× | 2.56× |
| Stage1 SW | 47633 | 31281 | 24272 | 1.29× | 1.96× |
| Stage2 SW | 36168 | 19240 | 15883 | 1.21× | 2.28× |
| Stage3 SW | 33050 | 14613 | 12744 | 1.15× | 2.59× |

BF16 OPT vs BF16 PyTorch baseline: 1.04–1.26× depending on stage/shift. Stage3 regular window is roughly parity (~0.99×) — launch overhead dominates when `Bwin` is small.

Peak extra memory drops roughly in half on TF32 OPT (e.g. Stage1: 5064 → 2531 MB) because V no longer needs a full FP32 copy on the host.

Note: `allow_tf32` speeds up Triton/cuBLAS linear projections but has no effect on PyTorch's FP32 SDPA backend (mem-efficient/math ignores the flag). The TF32 OPT speedup over FP32-TF32* comes almost entirely from our attention kernel.

### Known limitation: small windows (`N < 32`)

Production Aurora inference uses **`N = 144`** tokens per window (`window_size = 2×6×12` on the default patch grid). All encoder/decoder stages in our benchmarks and tests use that size or larger; BF16/TF32 CuTe paths are validated there.

**Downsampled stages can shrink the window** (e.g. patch merge → `N = 16`). We do **not** support BF16 CuTe attention for `N < 32` and have no plan to fix it: those shapes are out of scope for production inference in this repo.

| Observation | Detail |
|-------------|--------|
| Symptom | `bf16_mixed` can crash with `cudaErrorIllegalAddress` the first time CuTe BF16 runs on a small window (often misreported at the next CUDA sync, e.g. Triton AdaLN). |
| Repro | Isolated `window_attn_fwd_cute` / qkv-packed BF16 at `N = 16`; TF32 CuTe at the same shape is fine. |
| Root cause | BF16 **v1** single-pass prefetches **V** with **K** via **cp.async** before QK. For short `N`, the 128-thread MMA tile covers more rows/cols than `seqlen`; unpredicated V/K gmem copies can **read past valid memory**. TF32 v2 avoids this by loading V **after** QK+softmax with a sync, predicated path. |
| Why not fixed | A dedicated short-window kernel was explored; padding to 32×32 tiles caused correctness issues on some `N`. User decision: **ignore** — these windows will not appear in target deployments. |
| Workaround | Use `TF32_ACC_FP32`, PyTorch SDPA, or `AURORA_CUTE_WINDOW_ATTN=0` if you must run architectures with merged windows `N < 32`. |

If you add new patch resolutions or window layouts, confirm **`N ≥ 32`** (ideally `N = 144`) on every stage that enables `use_cute_window_attn` + `cute_window_attn_dtype=bfloat16`.

### What's next

Attention alone is a small slice of block time (QKV + MLP dominate). BF16 still has headroom — Stage3 parity and shifted-window gap are the obvious places to look. Multi-pass streaming (`N=576`) is a lower priority since Aurora's encoder/decoder stages all use `N=144`.

## Earlier benchmark notes (D2 / D3, different hardware)

These numbers are from a **GeForce RTX 5070 Ti Laptop GPU** and predate the CuTe attention work.

Profile snapshot (`uv run python aurora/profiling_swin3d.py --compare-d2d3 --compare-report-out profiling/swin3d_d2d3.md --preset stress`, with `batch=4`, `patch_res=(4,32,64)`, `L=8192`):

| Config | ms/forward | vs baseline | Peak alloc/reserved |
|--------|------------|-------------|---------------------|
| baseline | 159.28 | — | 8417 / 8846 MB |
| D2 | 152.16 | 1.047× | 1347 / 1428 MB |
| D2+D3 | 151.93 | 1.048× | 1414 / 1495 MB |

D2 provides the main gain (speed + large memory drop). D3 (`InferenceWorkspacePool`) is near-neutral for end-to-end latency here and mainly helps allocation stability.
