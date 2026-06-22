"""Per-component time breakdown of an optimized BF16 Swin3D block.

Answers: within a full Swin3D block, what fraction is window attention (and thus
how much can the attention kernel's heterogeneous DMA-warp design possibly move
the block-level needle)?

Times each sub-op on the real intermediate tensors at Aurora ERA5 stage dims.

Run:
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_block_breakdown.py
"""
from __future__ import annotations

import math
import os
import statistics
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.model.swin3d import Swin3DTransformerBlock
from aurora.model.workspace_pool import InferenceWorkspacePool

WARMUP = 20
MEASURED = 200
WINDOW_SIZE = (2, 6, 12)

# (C, H, W, D, num_heads, label)  - Aurora ERA5 encoder stages
SHAPES = [
    (4, 180, 360,  512,  8, "Stage1 D=512  H=8"),
    (4,  90, 180, 1024, 16, "Stage2 D=1024 H=16"),
    (4,  45,  90, 2048, 32, "Stage3 D=2048 H=32"),
]


def bench(fn) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(MEASURED):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    trim = max(1, int(len(ts) * 0.05))
    return statistics.mean(ts[trim:len(ts) - trim])


def run_stage(C, H, W, D, nh, label, shift):
    res = (C, H, W)
    L = C * H * W
    B = 1
    blk = Swin3DTransformerBlock(
        dim=D, num_heads=nh, time_dim=D,
        window_size=WINDOW_SIZE,
        shift_size=(1, 3, 6) if shift else (0, 0, 0),
        mlp_ratio=4.0,
        use_triton_layout=True, use_triton_adaln=True,
        use_triton_mlp=True, use_cute_window_attn=True,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    pool = InferenceWorkspacePool()
    blk._layout_pool = pool

    torch.manual_seed(0)
    x = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)

    timings = {}
    with torch.no_grad():
        # Full block
        timings["FULL block"] = bench(lambda: blk(x, c, res, rollout_step=0))

        # --- isolated components on real shapes ---
        # qkv projection
        attn = blk.attn
        x_norm = x  # shape ok for Linear timing
        timings["qkv proj"] = bench(lambda: attn.qkv(x_norm))

        # attention core (window_attn_fwd_cute) at (Bwin,H,N,Dh)
        from aurora.ops.cute import WinAttnPrecision, window_attn_fwd_cute
        Bwin = L // (WINDOW_SIZE[0] * WINDOW_SIZE[1] * WINDOW_SIZE[2])
        N = WINDOW_SIZE[0] * WINDOW_SIZE[1] * WINDOW_SIZE[2]
        Dh = D // nh
        q = torch.randn(Bwin, nh, N, Dh, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(Bwin, nh, N, Dh, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(Bwin, nh, N, Dh, device="cuda", dtype=torch.bfloat16) * 0.1
        timings["  attention core"] = bench(
            lambda: window_attn_fwd_cute(q, k, v, precision=WinAttnPrecision.BF16_MIXED)
        )

        # output projection
        x_attn = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
        timings["out proj"] = bench(lambda: attn.proj(x_attn))

        # MLP
        timings["MLP"] = bench(lambda: blk.mlp(x))

        # adaLN + residual (x2 in block)
        timings["adaLN+res (1 of 2)"] = bench(
            lambda: blk.norm1.forward_add_residual(x, x_attn, c)
        )

        # layout: roll/pad/partition + crop/unmerge (x1 each)
        from aurora.ops.triton_swin3d_layout import (
            roll_pad_partition_windows_triton,
            crop_roll_unmerge_windows_triton,
        )
        from aurora.model.swin3d import maybe_adjust_windows
        ws, ss = maybe_adjust_windows(WINDOW_SIZE, (1, 3, 6) if shift else (0, 0, 0), res)
        x_5d = x.view(B, C, H, W, D)
        xw = roll_pad_partition_windows_triton(x_5d, res, WINDOW_SIZE,
                                               (1, 3, 6) if shift else (0, 0, 0), pool=pool)
        timings["layout partition"] = bench(
            lambda: roll_pad_partition_windows_triton(
                x_5d, res, WINDOW_SIZE, (1, 3, 6) if shift else (0, 0, 0), pool=pool)
        )
        timings["layout unmerge"] = bench(
            lambda: crop_roll_unmerge_windows_triton(
                xw, res, WINDOW_SIZE, (1, 3, 6) if shift else (0, 0, 0), pool=pool)
        )

    full = timings["FULL block"]
    print(f"\n{'='*70}\n{label}   ({'SW' if shift else 'W'})   L={L}  Bwin={Bwin}\n{'='*70}")
    print(f"{'component':<24}{'ms':>10}{'% of full':>12}")
    print("-" * 46)
    for name, t in timings.items():
        pct = 100.0 * t / full
        print(f"{name:<24}{t:>10.4f}{pct:>11.1f}%")
    print("-" * 46)
    print("note: components measured in isolation; sum ≠ full due to launch gaps,")
    print("      but ratios show where block time concentrates.")
    return timings


def main():
    for shape in SHAPES:
        try:
            run_stage(*shape, shift=False)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[skip {shape[-1]}: OOM]")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
