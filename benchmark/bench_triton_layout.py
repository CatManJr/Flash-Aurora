"""Benchmark: Triton fused layout ops vs PyTorch ops for Swin3D window partition.

Measures the wall time of:
  roll + pad + window_partition  (forward)
  window_reverse + unpad + unroll  (inverse)

for the realistic Aurora ERA5 inference shapes, in both float32 and bfloat16.

Run:
    uv run python benchmark/bench_triton_layout.py
"""

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.ops.triton_swin3d_layout import (
    crop_roll_unmerge_windows_triton,
    roll_pad_partition_windows_triton,
)
from aurora.model.swin3d import crop_3d, pad_3d, window_partition_3d, window_reverse_3d
from aurora.model.util import maybe_adjust_windows

WARMUP = 20
MEASURED = 500

# ERA5 default: patch_res=(4,180,360), window_size=(2,6,12)
# PatchMerging halves H,W each stage (C stays 4):
#   Stage1: (4,180,360)  Stage2: (4,90,180)  Stage3: (4,45,90)
SHAPES = [
    # (B, C, H, W, D, window_size, shift_size, label)
    (1, 4, 180, 360, 512,  (2, 6, 12), (0, 0, 0), "Stage1 W  Bwin=1800"),
    (1, 4, 180, 360, 512,  (2, 6, 12), (1, 3, 6), "Stage1 SW Bwin=1800"),
    (1, 4,  90, 180, 1024, (2, 6, 12), (0, 0, 0), "Stage2 W  Bwin=450"),
    (1, 4,  90, 180, 1024, (2, 6, 12), (1, 3, 6), "Stage2 SW Bwin=450"),
    (1, 4,  45,  90, 2048, (2, 6, 12), (0, 0, 0), "Stage3 W  Bwin=128"),
    (1, 4,  45,  90, 2048, (2, 6, 12), (1, 3, 6), "Stage3 SW Bwin=128"),
]


def bench(fn, warmup=WARMUP, measured=MEASURED):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(measured):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    trim = max(1, int(len(times) * 0.05))
    core = times[trim:-trim]
    return statistics.mean(core)  # milliseconds


def pt_partition(x, res, ws, ss):
    """PyTorch: roll + pad + partition → (nW*B, N, D)."""
    if not all(s == 0 for s in ss):
        x = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
    C, H, W = res
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    x = pad_3d(x, pad_size)
    x = window_partition_3d(x, ws)
    return x.view(-1, ws[0] * ws[1] * ws[2], x.shape[-1])


def pt_reverse(windows, res, ws, ss):
    """PyTorch: unpartition + crop (symmetric) + unroll → (B, C, H, W, D)."""
    C, H, W = res
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    pad_C = C + pad_size[0]
    pad_H = H + pad_size[1]
    pad_W = W + pad_size[2]
    D = windows.shape[-1]
    attn = windows.view(-1, ws[0], ws[1], ws[2], D)
    x = window_reverse_3d(attn, ws, pad_C, pad_H, pad_W)
    x = crop_3d(x, pad_size)  # symmetric crop matching pad_3d
    if not all(s == 0 for s in ss):
        x = torch.roll(x, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
    return x


def check_accuracy(x, res, window_size, shift_size):
    """Verify Triton BF16 matches PyTorch BF16 for partition and reverse."""
    ws, ss = maybe_adjust_windows(window_size, shift_size, res)
    with torch.no_grad():
        ref = pt_partition(x, res, ws, ss)
        tri = roll_pad_partition_windows_triton(x, res, window_size, shift_size)
    ok_fwd = torch.allclose(ref, tri)

    with torch.no_grad():
        ref_rev = pt_reverse(ref, res, ws, ss)
        tri_rev = crop_roll_unmerge_windows_triton(tri, res, window_size, shift_size)
    ok_rev = torch.allclose(ref_rev, tri_rev)
    return ok_fwd, ok_rev


def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU : {props.name}  (SM{props.major}{props.minor})")
    print()

    # -----------------------------------------------------------------------
    # Accuracy check (both dtypes)
    # -----------------------------------------------------------------------
    all_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        print("=" * 60)
        print(f"Accuracy: Triton {dtype} vs PyTorch {dtype}")
        print("=" * 60)
        for B, C, H, W, D, window_size, shift_size, label in SHAPES:
            x = torch.randn(B, C, H, W, D, dtype=dtype, device="cuda")
            ok_fwd, ok_rev = check_accuracy(x, (C, H, W), window_size, shift_size)
            status = "PASS" if (ok_fwd and ok_rev) else "FAIL"
            print(f"  [{status}] {label}  fwd={ok_fwd} rev={ok_rev}")
            all_ok = all_ok and ok_fwd and ok_rev
        print()
    print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()

    # -----------------------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------------------
    col_l = 24
    hdr = (
        f"{'Shape':<{col_l}}"
        f"{'dtype':>8}"
        f"{'PT fwd':>10}"
        f"{'Tri fwd':>10}"
        f"{'spd':>6}"
        f"{'PT rev':>10}"
        f"{'Tri rev':>10}"
        f"{'spd':>6}"
        f"{'PT tot':>10}"
        f"{'Tri tot':>10}"
        f"{'spd':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for dtype in (torch.float32, torch.bfloat16):
        for B, C, H, W, D, window_size, shift_size, label in SHAPES:
            res = (C, H, W)
            ws, ss = maybe_adjust_windows(window_size, shift_size, res)
            x = torch.randn(B, C, H, W, D, dtype=dtype, device="cuda")
            with torch.no_grad():
                x_win = roll_pad_partition_windows_triton(x, res, window_size, shift_size)

            t_pt_f  = bench(lambda: pt_partition(x, res, ws, ss))
            t_tri_f = bench(lambda: roll_pad_partition_windows_triton(x, res, window_size, shift_size))
            t_pt_r  = bench(lambda: pt_reverse(x_win, res, ws, ss))
            t_tri_r = bench(lambda: crop_roll_unmerge_windows_triton(x_win, res, window_size, shift_size))

            def fmt(ms):
                return f"{ms*1e3:>8.1f}µs"

            spd_f = t_pt_f / t_tri_f
            spd_r = t_pt_r / t_tri_r
            spd_t = (t_pt_f + t_pt_r) / (t_tri_f + t_tri_r)

            print(
                f"{label:<{col_l}}"
                f"{str(dtype).split('.')[-1]:>8}"
                f"{fmt(t_pt_f)}"
                f"{fmt(t_tri_f)}"
                f"{spd_f:>6.2f}x"
                f"{fmt(t_pt_r)}"
                f"{fmt(t_tri_r)}"
                f"{spd_r:>6.2f}x"
                f"{fmt(t_pt_f+t_pt_r)}"
                f"{fmt(t_tri_f+t_tri_r)}"
                f"{spd_t:>6.2f}x"
            )
        print()

    print("Note: fwd=partition, rev=unpartition. Times in µs (microseconds).")


if __name__ == "__main__":
    main()
