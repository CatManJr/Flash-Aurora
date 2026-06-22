"""Nsight profile for CuTe BF16 window attention (v1 single-pass path).

Run from repo root:
    uv run python benchmark/profile_nsight_window_attn.py

Outputs under profiling/nsight/ (nsys .nsys-rep + summary; ncu .ncu-rep if run).
"""
from __future__ import annotations

import os
import subprocess
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

from flash_aurora.aurora.ops.cute.window_attn_fwd import WinAttnPrecision, window_attn_fwd_cute

NSYS = "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
NCU = "/usr/local/cuda/bin/ncu"

OUT_DIR = os.path.join(os.path.dirname(_BENCH_DIR), "profiling", "nsight")
os.makedirs(OUT_DIR, exist_ok=True)

# Realistic Stage1 + small shape for single-kernel deep dive
PROFILES = [
    ("stage1", 1800, 8, 144, 64),
    ("micro", 16, 8, 144, 64),
]


def _make_tensors(Bwin: int, H: int, N: int, Dh: int):
    q = torch.randn(Bwin, H, N, Dh, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _run_kernel(tag: str, Bwin: int, H: int, N: int, Dh: int, iters: int) -> None:
    q, k, v = _make_tensors(Bwin, H, N, Dh)
    # Compile + warmup
    for _ in range(5):
        window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"[{tag}] done  Bwin={Bwin} H={H} N={N} iters={iters}", flush=True)


def main() -> None:
    tag, Bwin, H, N, Dh = PROFILES[0]
    if len(sys.argv) >= 6:
        tag = sys.argv[1]
        Bwin, H, N, Dh = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    elif len(sys.argv) >= 2:
        for p in PROFILES:
            if p[0] == sys.argv[1]:
                tag, Bwin, H, N, Dh = p
                break

    iters = int(os.environ.get("PROFILE_ITERS", "20"))
    _run_kernel(tag, Bwin, H, N, Dh, iters)


if __name__ == "__main__":
    main()
