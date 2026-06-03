"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Perceiver / Swin MLP FFN — **planned** CuTeDSL or cuDNN Graph paths (not wired).

Two targets only (see ``benchmark/bench_perceiver_gemm_layout.py`` microbench):

1. **fc2 fast path** — ``F.linear`` GEMM ``M×1024 @ 1024×512`` (bottleneck at M≈140k).
   Weight TN prepack, optional decoder-only BF16 TC, or CuTe dense GEMM matching cuBLAS.

2. **fc1 + GELU + fc2 fused** — single graph; keep hidden in registers/SMEM between epilogues.
   Exact ``GELU(erf)`` epilogue on fc1; no standalone Triton MLP kernels.

References:

- CUTLASS ``cutlass/examples/python/CuTeDSL/`` (dense GEMM, Blackwell GeForce TMA).
- flash-attn ``flash_attn/ops/fused_dense.py``, ``flash_attn/cute/flash_fwd.py``.
- NVIDIA cuDNN Python frontend — fused GEMM + activation graphs, e.g.
  `gemm_swiglu <https://github.com/NVIDIA/cudnn-frontend/tree/develop/python/cudnn/gemm_swiglu>`__
  and sibling ``gemm_*`` ops under
  `python/cudnn <https://github.com/NVIDIA/cudnn-frontend/tree/develop/python/cudnn>`__.
"""

from __future__ import annotations

MLP_FFN_CUTE_AVAILABLE = False
