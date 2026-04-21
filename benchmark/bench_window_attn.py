"""Benchmark: CuTe BF16 window attention vs torch SDPA (flash attention).

Measures forward-pass latency and TFLOPS for Aurora encoder window shapes.

Aurora uses window_size=(2, 6, 12) for all encoder and decoder stages.
The actual N = Wc×Wh×Ww depends on the spatial resolution of the input:

    N=144  (2×6×12)   standard resolution          — single-pass
    N=288  (2×6×24)   2× spatial resolution        — single-pass
    N=576  (2×12×24)  4× spatial resolution        — streaming (multi-pass)

Also performs a numerical accuracy check (CuTe vs FP32 SDPA) for every shape.

Run:
    uv run python benchmarks/bench_window_attn.py
"""

import math
import os
import statistics
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.ops.cute.window_attn_fwd import (
    _choose_tile_n,
    _CUTE_AVAILABLE,
    WinAttnPrecision,
    window_attn_fwd_cute,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP      = 20    # iterations discarded
MEASURED    = 1000  # iterations timed
TRIM_FRAC   = 0.05  # fraction trimmed from each tail before computing stats

USE_CUTE_KERNEL = os.environ.get("AURORA_CUTE_WINDOW_ATTN", "") == "1"

# ---------------------------------------------------------------------------
# Shapes: (Bwin, H, N, Dh, label)   —   N = Wc × Wh × Ww
# ---------------------------------------------------------------------------

SHAPES = [
    # (Bwin,  H,   N,  Dh,  label)
    # Bwin = num_windows × batch_size (typical inference values)
    (16,   8, 144, 64, "N=144 (2×6×12)  H=8"),
    ( 8,  16, 144, 64, "N=144 (2×6×12)  H=16"),
    ( 4,  32, 144, 64, "N=144 (2×6×12)  H=32"),
    ( 8,   8, 288, 64, "N=288 (2×6×24)  H=8   2× spatial"),
    ( 4,  16, 288, 64, "N=288 (2×6×24)  H=16  2× spatial"),
    ( 2,  32, 576, 64, "N=576 (2×12×24) H=32  4× spatial  streaming"),
]

# Realistic Aurora inference shapes (ERA5 0.25°, batch=1)
# patch_res = (latent_levels=4, 720//4, 1440//4) = (4, 180, 360)
# PatchMerging halves H and W each stage (C stays 4).
# nW = (4/window_C) * (H/window_H) * (W/window_W) with window=(2,6,12)
#   Stage 1: (4, 180, 360) → nW = 2×30×30 = 1800
#   Stage 2: (4,  90, 180) → nW = 2×15×15 =  450
#   Stage 3: (4,  45,  90) → pad→(4,48,96) → nW = 2×8×8 = 128
SHAPES_REALISTIC = [
    # (Bwin,    H,   N,  Dh,  label)
    (1800,  8, 144, 64, "ERA5 Stage1 enc  Bwin=1800 H=8"),
    ( 450, 16, 144, 64, "ERA5 Stage2 enc  Bwin=450  H=16"),
    ( 128, 32, 144, 64, "ERA5 Stage3 enc  Bwin=128  H=32"),
    ( 128, 32, 144, 64, "ERA5 Stage1 dec  Bwin=128  H=32"),
    ( 450, 16, 144, 64, "ERA5 Stage2 dec  Bwin=450  H=16"),
    (1800,  8, 144, 64, "ERA5 Stage3 dec  Bwin=1800 H=8"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def attention_flops(Bwin: int, H: int, N: int, Dh: int) -> int:
    """Forward-pass FLOPs: 2 × (Q@K^T) + 2 × (P@V), each = 2·Bwin·H·N·N·Dh."""
    return 4 * Bwin * H * N * N * Dh


def tflops(flop: int, elapsed_ms: float) -> float:
    return flop / elapsed_ms / 1e9  # ms → s · 1e3, flop → TFLOP · 1e12 → net /1e9


@dataclass
class BenchStats:
    mean:   float   # trimmed mean (ms)
    std:    float   # sample std-dev of trimmed data (ms)
    ci95:   float   # 95 % CI half-width: 1.96 · std / √n  (ms)
    cv:     float   # coefficient of variation: std / mean × 100  (%)
    median: float   # raw median (ms)
    p5:     float   # 5th percentile (ms)
    p95:    float   # 95th percentile (ms)


def bench(fn, warmup: int = WARMUP, measured: int = MEASURED) -> BenchStats:
    """Benchmark fn with rigorous statistics.

    Protocol
    --------
    1. `warmup` un-timed iterations to reach thermal/clock steady-state.
    2. `measured` CUDA-event-timed iterations, one sync per iteration
       (avoids batching multiple kernels into one timing window).
    3. Sort all samples; trim top and bottom ``TRIM_FRAC`` fraction to remove
       scheduler spikes and cache-warm artefacts.
    4. Compute trimmed mean, sample std-dev, and 95 % CI via normal
       approximation (valid here because n ≫ 30 after trimming).
    """
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

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
    n = len(times)

    median = times[n // 2]
    p5     = times[int(n * 0.05)]
    p95    = times[int(n * 0.95)]

    trim = max(1, int(n * TRIM_FRAC))
    core = times[trim : n - trim]

    tmean = statistics.mean(core)
    tstd  = statistics.stdev(core) if len(core) > 1 else 0.0
    ci95  = 1.96 * tstd / math.sqrt(len(core))
    cv    = tstd / tmean * 100 if tmean > 0 else 0.0

    return BenchStats(mean=tmean, std=tstd, ci95=ci95, cv=cv,
                      median=median, p5=p5, p95=p95)


def make_qkv(Bwin, H, N, Dh, dtype, device="cuda"):
    g = torch.Generator(device=device).manual_seed(42)
    kw = dict(dtype=dtype, device=device, generator=g)
    s = 0.1
    return (
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
    )


def fp32_reference(q_bf, k_bf, v_bf, scale):
    """FP32 reference: upcast to float, run SDPA, return BF16."""
    q32 = q_bf.float()
    k32 = k_bf.float()
    v32 = v_bf.float()
    out = F.scaled_dot_product_attention(q32, k32, v32, scale=scale)
    return out.bfloat16()


def check_accuracy(Bwin, H, N, Dh, scale, label, rtol=2e-2, atol=2e-2):
    """Run CuTe and SDPA on the same BF16 inputs, compare numerically."""
    q, k, v = make_qkv(Bwin, H, N, Dh, torch.bfloat16)

    with torch.no_grad():
        out_cute = window_attn_fwd_cute(
            q, k, v,
            precision=WinAttnPrecision.BF16_MIXED,
            scale_qk=scale,
        )
        ref = fp32_reference(q, k, v, scale)

    out_f = out_cute.float()
    ref_f = ref.float()
    abs_err = (out_f - ref_f).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    passed = torch.allclose(out_f, ref_f, rtol=rtol, atol=atol)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         max_abs={max_err:.4e}  mean_abs={mean_err:.4e}  "
          f"(atol={atol}, rtol={rtol})")
    if not passed:
        # Show worst offender location
        idx = abs_err.argmax()
        flat = abs_err.reshape(-1)
        worst = flat.argmax().item()
        total = flat.numel()
        frac = (flat > atol).float().mean().item()
        print(f"         {frac*100:.1f}% of elements exceed atol")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark.")
        return

    device = torch.cuda.current_device()
    props  = torch.cuda.get_device_properties(device)
    print(f"GPU : {props.name}  (SM{props.major}{props.minor}, "
          f"{props.total_memory // 2**20} MB)")
    print(f"CuTe kernel available : {_CUTE_AVAILABLE}")
    print(f"CuTe GEMM tiles : {'active (AURORA_CUTE_WINDOW_ATTN=1)' if USE_CUTE_KERNEL else 'disabled (stable softmax path)'}")
    print()

    scale = 1.0 / math.sqrt(64)

    # -----------------------------------------------------------------------
    # Numerical accuracy check (CuTe vs FP32 SDPA reference)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Numerical accuracy: CuTe BF16  vs  FP32 SDPA reference")
    print("=" * 60)
    all_passed = True
    for Bwin, H, N, Dh, label in SHAPES:
        ok = check_accuracy(Bwin, H, N, Dh, scale, label)
        all_passed = all_passed and ok
    print()
    print("Overall:", "ALL PASS" if all_passed else "SOME FAILED")
    print()

    # -----------------------------------------------------------------------
    # Table header
    # -----------------------------------------------------------------------
    col_label  = 36
    col_n      = 5
    col_tile   = 7
    col_pass   = 5
    col_ms     = 11   # "mean±ci" e.g. "0.024±0.000"
    col_cv     = 5    # cv%
    col_tflops = 8

    hdr = (
        f"{'Shape':<{col_label}}"
        f"{'N':>{col_n}}"
        f"{'tile_n':>{col_tile}}"
        f"{'pass':>{col_pass}}"
        f"{'CuTe mean±ci':>{col_ms+4}}"
        f"{'cv%':>{col_cv}}"
        f"{'TFLOPS':>{col_tflops}}"
        f"{'SDPA mean±ci':>{col_ms+4}}"
        f"{'cv%':>{col_cv}}"
        f"{'TFLOPS':>{col_tflops}}"
        f"{'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    with torch.no_grad():
        for Bwin, H, N, Dh, label in SHAPES:
            q_bf, k_bf, v_bf = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
            flop = attention_flops(Bwin, H, N, Dh)

            tile_n   = _choose_tile_n(N, head_dim=Dh)
            n_passes = math.ceil(N / tile_n)
            pass_str = "1" if n_passes == 1 else str(n_passes)

            # --- CuTe BF16 kernel ---
            def run_cute():
                window_attn_fwd_cute(
                    q_bf, k_bf, v_bf,
                    precision=WinAttnPrecision.BF16_MIXED,
                    scale_qk=scale,
                )

            r_cute  = bench(run_cute)
            tf_cute = tflops(flop, r_cute.mean)

            # --- torch SDPA (flash attention backend) ---
            def run_sdpa():
                F.scaled_dot_product_attention(q_bf, k_bf, v_bf, scale=scale)

            r_sdpa  = bench(run_sdpa)
            tf_sdpa = tflops(flop, r_sdpa.mean)

            speedup = r_sdpa.mean / r_cute.mean

            cute_str = f"{r_cute.mean:.3f}±{r_cute.ci95:.3f}"
            sdpa_str = f"{r_sdpa.mean:.3f}±{r_sdpa.ci95:.3f}"

            print(
                f"{label:<{col_label}}"
                f"{N:>{col_n}}"
                f"{tile_n:>{col_tile}}"
                f"{pass_str:>{col_pass}}"
                f"{cute_str:>{col_ms+4}}"
                f"{r_cute.cv:>{col_cv}.1f}"
                f"{tf_cute:>{col_tflops}.2f}"
                f"{sdpa_str:>{col_ms+4}}"
                f"{r_sdpa.cv:>{col_cv}.1f}"
                f"{tf_sdpa:>{col_tflops}.2f}"
                f"{speedup:>8.2f}x"
            )

    print()
    trim_pct = int(TRIM_FRAC * 100)
    print(f"Note: latency = trimmed mean ± 95% CI  "
          f"(top/bottom {trim_pct}% of {MEASURED} samples discarded).")
    print(f"      cv% = std/mean×100;  TFLOPS = 4·Bwin·H·N²·Dh / mean_latency.")

    # -----------------------------------------------------------------------
    # Realistic Aurora ERA5 shapes
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Realistic Aurora inference shapes  (ERA5 0.25°, batch=1)")
    print("=" * 60)
    print(hdr)
    print("-" * len(hdr))

    with torch.no_grad():
        for Bwin, H, N, Dh, label in SHAPES_REALISTIC:
            q_bf, k_bf, v_bf = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
            flop = attention_flops(Bwin, H, N, Dh)

            tile_n   = _choose_tile_n(N, head_dim=Dh)
            n_passes = math.ceil(N / tile_n)
            pass_str = "1" if n_passes == 1 else str(n_passes)

            def run_cute():
                window_attn_fwd_cute(
                    q_bf, k_bf, v_bf,
                    precision=WinAttnPrecision.BF16_MIXED,
                    scale_qk=scale,
                )

            r_cute  = bench(run_cute)
            tf_cute = tflops(flop, r_cute.mean)

            def run_sdpa():
                F.scaled_dot_product_attention(q_bf, k_bf, v_bf, scale=scale)

            r_sdpa  = bench(run_sdpa)
            tf_sdpa = tflops(flop, r_sdpa.mean)

            speedup = r_sdpa.mean / r_cute.mean

            cute_str = f"{r_cute.mean:.3f}±{r_cute.ci95:.3f}"
            sdpa_str = f"{r_sdpa.mean:.3f}±{r_sdpa.ci95:.3f}"

            print(
                f"{label:<{col_label}}"
                f"{N:>{col_n}}"
                f"{tile_n:>{col_tile}}"
                f"{pass_str:>{col_pass}}"
                f"{cute_str:>{col_ms+4}}"
                f"{r_cute.cv:>{col_cv}.1f}"
                f"{tf_cute:>{col_tflops}.2f}"
                f"{sdpa_str:>{col_ms+4}}"
                f"{r_sdpa.cv:>{col_cv}.1f}"
                f"{tf_sdpa:>{col_tflops}.2f}"
                f"{speedup:>8.2f}x"
            )

    print()
    print(f"Note: latency = trimmed mean ± 95% CI  "
          f"(top/bottom {trim_pct}% of {MEASURED} samples discarded).")
    print(f"      cv% = std/mean×100;  TFLOPS = 4·Bwin·H·N²·Dh / mean_latency.")


if __name__ == "__main__":
    main()
