"""A/B benchmark: BF16 window attention v1 (128-thread cp.async) vs v2 (160-thread TMA + DMA warp).

Goal: isolate whether the v2 heterogeneous pipeline (dedicated DMA warp, TMA loads,
160 threads) actually beats the simpler v1 (128 threads, cp.async) — especially for
the single-pass N=144 shapes that dominate real Aurora workloads, where a dedicated
DMA warp has no prefetch overlap to exploit.

Both kernels are compiled directly (bypassing the dispatch layer) so we compare the
exact same precision/numerics. SDPA is included only as a correctness anchor.

Run:
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_v1_vs_v2.py
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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.ops.cute._smem_utils import _choose_tile_n
from aurora.ops.cute._kernel_bf16 import _get_or_compile_bf16
from aurora.ops.cute._kernel_bf16_v2 import _get_or_compile_bf16_v2

WARMUP = 30
MEASURED = 1000
TRIM_FRAC = 0.05

# (Bwin, H, N, Dh, label)
SHAPES = [
    (16,   8, 144, 64, "N=144  H=8   single-pass"),
    ( 4,  32, 144, 64, "N=144  H=32  single-pass"),
    ( 8,   8, 288, 64, "N=288  H=8   single-pass"),
    ( 2,  32, 576, 64, "N=576  H=32  multi-pass (8)"),
    ( 2,   8, 400, 64, "N=400  H=8   multi-pass (5)"),
    # Realistic Aurora encoder/decoder stages (all N=144 single-pass).
    (1800,  8, 144, 64, "ERA5 S1 enc Bwin=1800 H=8"),
    ( 450, 16, 144, 64, "ERA5 S2 enc Bwin=450  H=16"),
    ( 128, 32, 144, 64, "ERA5 S3 enc Bwin=128  H=32"),
]


def make_qkv(Bwin, H, N, Dh, device="cuda"):
    g = torch.Generator(device=device).manual_seed(42)
    kw = dict(dtype=torch.bfloat16, device=device, generator=g)
    s = 0.1
    return (
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
    )


def bench(fn) -> float:
    """Return trimmed-mean latency in ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(MEASURED):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    trim = max(1, int(len(times) * TRIM_FRAC))
    core = times[trim : len(times) - trim]
    return statistics.mean(core)


def attention_flops(Bwin, H, N, Dh):
    return 4 * Bwin * H * N * N * Dh


def run_shape(Bwin, H, N, Dh, label):
    q, k, v = make_qkv(Bwin, H, N, Dh)
    scale = 1.0 / math.sqrt(Dh)
    scale_log2 = torch.tensor(0.0)  # placeholder; real value below
    from cutlass import Float32
    scale_log2 = Float32(math.log2(math.e) * scale)

    tile_m = 64
    tile_n = _choose_tile_n(N, head_dim=Dh, tile_m=tile_m)
    passes = math.ceil(N / tile_n)
    has_bias = False

    out_v1 = torch.zeros_like(q)
    out_v2 = torch.zeros_like(q)

    fn_v1 = _get_or_compile_bf16(
        head_dim=Dh, seq_len=N, has_bias=has_bias, tile_m=tile_m, tile_n=tile_n,
        q=q, k=k, v=v, o=out_v1, bias_or_none=None,
    )
    fn_v2 = _get_or_compile_bf16_v2(
        head_dim=Dh, seq_len=N, has_bias=has_bias, tile_m=tile_m, tile_n=tile_n,
        q=q, k=k, v=v, o=out_v2, bias_or_none=None,
    )

    call_v1 = lambda: fn_v1(q, k, v, out_v1, None, scale_log2)
    call_v2 = lambda: fn_v2(q, k, v, out_v2, None, scale_log2)

    # Correctness anchor: both must match SDPA.
    call_v1()
    call_v2()
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    err_v1 = (out_v1.float() - ref).abs().max().item()
    err_v2 = (out_v2.float() - ref).abs().max().item()

    t_v1 = bench(call_v1)
    t_v2 = bench(call_v2)

    flop = attention_flops(Bwin, H, N, Dh)
    tflops_v1 = flop / t_v1 / 1e9
    tflops_v2 = flop / t_v2 / 1e9
    speedup = t_v1 / t_v2  # >1 means v2 faster

    return dict(
        label=label, N=N, tile_n=tile_n, passes=passes,
        t_v1=t_v1, t_v2=t_v2, tflops_v1=tflops_v1, tflops_v2=tflops_v2,
        speedup=speedup, err_v1=err_v1, err_v2=err_v2,
    )


def main():
    print("=" * 100)
    print("BF16 window attention A/B:  v1 (128-thread cp.async)  vs  v2 (160-thread TMA + DMA warp)")
    print("=" * 100)
    print(f"{'Shape':<28}{'N':>5}{'tn':>5}{'pass':>5}"
          f"{'v1 ms':>10}{'v2 ms':>10}{'v1 TF':>8}{'v2 TF':>8}"
          f"{'v2/v1':>8}  {'err_v1':>9}{'err_v2':>9}")
    print("-" * 100)
    rows = []
    for shape in SHAPES:
        r = run_shape(*shape)
        rows.append(r)
        verdict = "v2 faster" if r["speedup"] > 1.0 else "v1 faster"
        print(f"{r['label']:<28}{r['N']:>5}{r['tile_n']:>5}{r['passes']:>5}"
              f"{r['t_v1']:>10.4f}{r['t_v2']:>10.4f}{r['tflops_v1']:>8.1f}{r['tflops_v2']:>8.1f}"
              f"{r['speedup']:>7.2f}x  {r['err_v1']:>9.2e}{r['err_v2']:>9.2e}  {verdict}")
    print("-" * 100)
    print("v2/v1 > 1.0  → v2 (DMA warp) faster;  < 1.0 → v1 (simple 128-thread) faster")

    sp = [r["speedup"] for r in rows]
    single = [r["speedup"] for r in rows if r["passes"] == 1]
    multi = [r["speedup"] for r in rows if r["passes"] > 1]
    print(f"\ngeomean v2/v1  all      : {math.exp(statistics.fmean(map(math.log, sp))):.3f}x")
    if single:
        print(f"geomean v2/v1  single-pass: {math.exp(statistics.fmean(map(math.log, single))):.3f}x")
    if multi:
        print(f"geomean v2/v1  multi-pass : {math.exp(statistics.fmean(map(math.log, multi))):.3f}x")


if __name__ == "__main__":
    main()
