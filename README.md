# Design and Implement High-Performance Computing Framework for GFMS: MS Aurora as an example

## Install CuTe DSL & Flash Attention
git clone
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


## Notes (this repo)

The [`aurora/`](aurora/) tree is the [Microsoft Aurora](https://github.com/microsoft/aurora) model code with local changes for inference-oriented performance experiments.

- **Triton ops** (CUDA float32 inference paths) under [`aurora/aurora/ops/`](aurora/aurora/ops/): e.g. window **layout** only in [`triton_swin3d_layout.py`](aurora/aurora/ops/triton_swin3d_layout.py) (roll/pad/partition + inverse), **AdaLN** in `triton_adaln.py`, **GELU** in `triton_gelu.py` (not a full Swin block in one file).
- **D2 (fused AdaLN + residual)**: used from `AdaptiveLayerNorm` and `Swin3DTransformerBlock` when enabled; see tests in [`aurora/tests/test_triton_swin3d.py`](aurora/tests/test_triton_swin3d.py).
- **D3 (inference workspace pool)**: [`InferenceWorkspacePool`](aurora/aurora/model/workspace_pool.py) reuses a scratch buffer for the backbone’s final decoder `concat` (`torch.cat(..., out=buf)`), optional on [`Swin3DTransformerBackbone`](aurora/aurora/model/swin3d.py) / [`Aurora`](aurora/aurora/model/aurora.py). Tests: [`aurora/tests/test_inference_workspace_pool.py`](aurora/tests/test_inference_workspace_pool.py).
- **Profiling write-ups** (local): [`profiling/`](profiling/). Backbone **D2 vs D2+D3** (from repo root, CUDA): default `--compare-d2d3` is **light** (batch=1, repeat=4). **`--preset stress`** uses `L=2048`, batch=4 (8192 tokens/step, `warmup`/`repeat` raised). **`--preset stress-heavy`** is batch=8, `L=8192` (very large VRAM). Example:  
  `uv run python aurora/profiling_swin3d.py --compare-d2d3 --preset stress --compare-report-out profiling/swin3d_d2d3_stress.md`

## Current Benchmark Notes

- **Hardware**: NVIDIA GeForce **RTX 5070 Ti Laptop GPU** (mobile).
- **Profile snapshot** (`uv run python aurora/profiling_swin3d.py --compare-d2d3 --compare-report-out profiling/swin3d_d2d3.md --preset stress`, with `batch=4`, `patch_res=(4,32,64)`, `L=8192`):
  - baseline: `159.28 ms/forward`, peak CUDA allocated/reserved `8417.2 / 8845.8 MB`
  - D2: `152.16 ms/forward` (**1.047x** vs baseline), peak `1346.7 / 1428.2 MB`
  - D2+D3: `151.93 ms/forward` (**1.048x** vs baseline), peak `1413.6 / 1495.3 MB`
- **Interpretation**: D2 provides the main gain here (speed + large memory drop). D3 (`InferenceWorkspacePool`) is near-neutral for end-to-end latency in this profile and is mainly useful for allocation stability.