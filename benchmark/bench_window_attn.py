"""Benchmark: CuTe window attention vs torch SDPA.

Measures forward-pass latency and TFLOPS for Aurora encoder window shapes.

Paths benchmarked
-----------------
* **BF16_MIXED** — BF16 I/O, FP32 softmax/accum (CuTe) vs BF16 SDPA.
* **TF32_ACC_FP32** — FP32 I/O, TF32 QK MMA (CuTe) vs torch SDPA with TF32 matmuls
  (1×TF32 apples-to-apples) and vs strict-FP32 SDPA (quality reference).

Aurora uses window_size=(2, 6, 12) for all encoder and decoder stages.
The actual N = Wc×Wh×Ww depends on the spatial resolution of the input:

    N=144  (2×6×12)   standard resolution          — single-pass
    N=288  (2×6×24)   2× spatial resolution        — single-pass / streaming
    N=576  (2×12×24)  4× spatial resolution        — streaming (multi-pass)

Run:
    uv run python benchmark/bench_window_attn.py
"""

import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402  — before torch (OMP_NUM_THREADS)

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.ops.cute.window_attn_fwd import (
    _choose_tile_n,
    _choose_tile_n_tf32,
    _CUTE_AVAILABLE,
    WinAttnPrecision,
    window_attn_fwd_cute,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP      = 20
MEASURED    = 1000
TRIM_FRAC   = 0.05

USE_CUTE_KERNEL = os.environ.get("AURORA_CUTE_WINDOW_ATTN", "") == "1"

SHAPES = [
    (16,   8, 144, 64, "N=144 (2×6×12)  H=8"),
    ( 8,  16, 144, 64, "N=144 (2×6×12)  H=16"),
    ( 4,  32, 144, 64, "N=144 (2×6×12)  H=32"),
    ( 8,   8, 288, 64, "N=288 (2×6×24)  H=8   2× spatial"),
    ( 4,  16, 288, 64, "N=288 (2×6×24)  H=16  2× spatial"),
    ( 2,  32, 576, 64, "N=576 (2×12×24) H=32  4× spatial  streaming"),
]

SHAPES_REALISTIC = [
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
    return 4 * Bwin * H * N * N * Dh


def tflops(flop: int, elapsed_ms: float) -> float:
    return flop / elapsed_ms / 1e9


@dataclass
class BenchStats:
    mean:   float
    std:    float
    ci95:   float
    cv:     float
    median: float
    p5:     float
    p95:    float


def bench(fn, warmup: int = WARMUP, measured: int = MEASURED) -> BenchStats:
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


def fp32_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    allow_tf32: bool,
) -> torch.Tensor:
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def fp32_reference_bf16(q_bf, k_bf, v_bf, scale):
    return fp32_sdpa(q_bf.float(), k_bf.float(), v_bf.float(), scale, allow_tf32=False).bfloat16()


def _print_accuracy_result(
    label: str,
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    rtol: float,
    atol: float,
) -> bool:
    out_f = candidate.float()
    ref_f = baseline.float()
    abs_err = (out_f - ref_f).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    passed = torch.allclose(out_f, ref_f, rtol=rtol, atol=atol)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         max_abs={max_err:.6e}  mean_abs={mean_err:.6e}  "
          f"(atol={atol:g}, rtol={rtol:g})")
    if not passed:
        frac = (abs_err.reshape(-1) > atol).float().mean().item()
        print(f"         {frac * 100:.1f}% of elements exceed atol")
    elif max_err > 0.0:
        print(f"         bitwise_equal={torch.equal(out_f, ref_f)}")
    return passed


def check_accuracy_bf16(Bwin, H, N, Dh, scale, label, rtol=2e-2, atol=2e-2):
    q, k, v = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
    with torch.no_grad():
        out_cute = window_attn_fwd_cute(
            q, k, v, precision=WinAttnPrecision.BF16_MIXED, scale_qk=scale,
        )
        ref = fp32_reference_bf16(q, k, v, scale)
    return _print_accuracy_result(label, out_cute, ref, rtol, atol)


def check_accuracy_tf32(
    Bwin, H, N, Dh, scale, label, *, ref_allow_tf32: bool, rtol=1e-3, atol=1e-3,
):
    if not _CUTE_AVAILABLE:
        print(f"  [SKIP] {label}  (CuTeDSL not available)")
        return True
    q, k, v = make_qkv(Bwin, H, N, Dh, torch.float32)
    with torch.no_grad():
        out_cute = window_attn_fwd_cute(
            q, k, v, precision=WinAttnPrecision.TF32_ACC_FP32, scale_qk=scale,
        )
        ref = fp32_sdpa(q, k, v, scale, allow_tf32=ref_allow_tf32)
    return _print_accuracy_result(label, out_cute, ref, rtol, atol)


_COL_LABEL  = 36
_COL_N      = 5
_COL_TILE   = 7
_COL_PASS   = 5
_COL_MS     = 11
_COL_CV     = 5
_COL_TFLOPS = 8


def _perf_header(cute_col: str, baseline_col: str) -> str:
    col_ms = _COL_MS
    return (
        f"{'Shape':<{_COL_LABEL}}"
        f"{'N':>{_COL_N}}"
        f"{'tile_n':>{_COL_TILE}}"
        f"{'pass':>{_COL_PASS}}"
        f"{cute_col:>{col_ms + 4}}"
        f"{'cv%':>{_COL_CV}}"
        f"{'TFLOPS':>{_COL_TFLOPS}}"
        f"{baseline_col:>{col_ms + 4}}"
        f"{'cv%':>{_COL_CV}}"
        f"{'TFLOPS':>{_COL_TFLOPS}}"
        f"{'speedup':>8}"
    )


def _print_perf_row(
    label: str,
    N: int,
    tile_n: int,
    pass_str: str,
    r_cute: BenchStats,
    r_base: BenchStats,
    flop: int,
) -> None:
    tf_cute = tflops(flop, r_cute.mean)
    tf_base = tflops(flop, r_base.mean)
    speedup = r_base.mean / r_cute.mean
    cute_str = f"{r_cute.mean:.3f}±{r_cute.ci95:.3f}"
    base_str = f"{r_base.mean:.3f}±{r_base.ci95:.3f}"
    print(
        f"{label:<{_COL_LABEL}}"
        f"{N:>{_COL_N}}"
        f"{tile_n:>{_COL_TILE}}"
        f"{pass_str:>{_COL_PASS}}"
        f"{cute_str:>{_COL_MS + 4}}"
        f"{r_cute.cv:>{_COL_CV}.1f}"
        f"{tf_cute:>{_COL_TFLOPS}.2f}"
        f"{base_str:>{_COL_MS + 4}}"
        f"{r_base.cv:>{_COL_CV}.1f}"
        f"{tf_base:>{_COL_TFLOPS}.2f}"
        f"{speedup:>8.2f}x"
    )


def run_perf_table(
    shapes: list[tuple[int, int, int, int, str]],
    *,
    title: str,
    dtype: torch.dtype,
    choose_tile_n: Callable[..., int],
    cute_precision: WinAttnPrecision,
    make_baseline: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], Callable[[], None]],
    cute_col: str,
    baseline_col: str,
) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    hdr = _perf_header(cute_col, baseline_col)
    print(hdr)
    print("-" * len(hdr))

    with torch.no_grad():
        for Bwin, H, N, Dh, label in shapes:
            q, k, v = make_qkv(Bwin, H, N, Dh, dtype)
            flop = attention_flops(Bwin, H, N, Dh)
            tile_n = choose_tile_n(N, head_dim=Dh)
            n_pass = math.ceil(N / tile_n)
            pass_str = "1" if n_pass == 1 else str(n_pass)
            scale = 1.0 / math.sqrt(Dh)

            def run_cute():
                window_attn_fwd_cute(
                    q, k, v, precision=cute_precision, scale_qk=scale,
                )

            r_cute = bench(run_cute)
            r_base = bench(make_baseline(q, k, v, scale))
            _print_perf_row(label, N, tile_n, pass_str, r_cute, r_base, flop)


def _print_latency_note() -> None:
    trim_pct = int(TRIM_FRAC * 100)
    print()
    print(f"Note: latency = trimmed mean ± 95% CI  "
          f"(top/bottom {trim_pct}% of {MEASURED} samples discarded).")
    print(f"      cv% = std/mean×100;  TFLOPS = 4·Bwin·H·N²·Dh / mean_latency.")


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

    print("=" * 60)
    print("Numerical accuracy: CuTe BF16  vs  FP32 SDPA  (baseline)")
    print("=" * 60)
    all_passed = True
    for Bwin, H, N, Dh, label in SHAPES:
        ok = check_accuracy_bf16(Bwin, H, N, Dh, scale, label)
        all_passed = all_passed and ok
    print()
    print("Overall:", "ALL PASS" if all_passed else "SOME FAILED")

    print()
    print("=" * 60)
    print("Numerical accuracy: CuTe TF32  vs  SDPA-TF32 baseline  (1×TF32)")
    print("=" * 60)
    tf32_sdpa_passed = True
    if not _CUTE_AVAILABLE:
        print("  [SKIP] CuTeDSL not available — TF32 kernel not built.")
    else:
        for Bwin, H, N, Dh, label in SHAPES:
            ok = check_accuracy_tf32(
                Bwin, H, N, Dh, scale, label, ref_allow_tf32=True,
            )
            tf32_sdpa_passed = tf32_sdpa_passed and ok
        print()
        print("Overall:", "ALL PASS" if tf32_sdpa_passed else "SOME FAILED")

    print()
    print("=" * 60)
    print("Numerical accuracy: CuTe TF32  vs  strict-FP32 SDPA baseline")
    print("=" * 60)
    tf32_strict_passed = True
    if not _CUTE_AVAILABLE:
        print("  [SKIP] CuTeDSL not available.")
    else:
        for Bwin, H, N, Dh, label in SHAPES:
            ok = check_accuracy_tf32(
                Bwin, H, N, Dh, scale, label, ref_allow_tf32=False,
            )
            tf32_strict_passed = tf32_strict_passed and ok
        print()
        print("Overall:", "ALL PASS" if tf32_strict_passed else "SOME FAILED")

    def sdpa_bf16(q, k, v, s):
        return lambda: F.scaled_dot_product_attention(q, k, v, scale=s)

    def sdpa_tf32_fp32(q, k, v, s):
        return lambda: fp32_sdpa(q, k, v, s, allow_tf32=True)

    def sdpa_strict_fp32(q, k, v, s):
        return lambda: fp32_sdpa(q, k, v, s, allow_tf32=False)

    run_perf_table(
        SHAPES,
        title="Performance: CuTe BF16_MIXED  vs  torch SDPA (BF16)",
        dtype=torch.bfloat16,
        choose_tile_n=_choose_tile_n,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        make_baseline=sdpa_bf16,
        cute_col="CuTe mean±ci",
        baseline_col="SDPA mean±ci",
    )
    _print_latency_note()

    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES,
            title="Performance: CuTe TF32  vs  SDPA-TF32  (1×TF32 matmul, fair)",
            dtype=torch.float32,
            choose_tile_n=_choose_tile_n_tf32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            make_baseline=sdpa_tf32_fp32,
            cute_col="CuTe-TF32",
            baseline_col="SDPA-TF32",
        )
        _print_latency_note()
        print("      speedup > 1  → CuTe TF32 kernel faster than torch SDPA @ TF32.")

        run_perf_table(
            SHAPES,
            title="Performance: CuTe TF32  vs  strict-FP32 SDPA  (quality ref)",
            dtype=torch.float32,
            choose_tile_n=_choose_tile_n_tf32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            make_baseline=sdpa_strict_fp32,
            cute_col="CuTe-TF32",
            baseline_col="SDPA-strict",
        )
        _print_latency_note()
        print("      strict baseline disables TF32; larger speedup is not apples-to-apples.")
    else:
        print()
        print("=" * 60)
        print("Performance: TF32_ACC_FP32  —  SKIPPED (CuTeDSL not available)")
        print("=" * 60)

    run_perf_table(
        SHAPES_REALISTIC,
        title="Realistic Aurora shapes: CuTe BF16_MIXED  vs  torch SDPA (BF16)",
        dtype=torch.bfloat16,
        choose_tile_n=_choose_tile_n,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        make_baseline=sdpa_bf16,
        cute_col="CuTe mean±ci",
        baseline_col="SDPA mean±ci",
    )
    _print_latency_note()

    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_REALISTIC,
            title="Realistic shapes: CuTe TF32  vs  SDPA-TF32",
            dtype=torch.float32,
            choose_tile_n=_choose_tile_n_tf32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            make_baseline=sdpa_tf32_fp32,
            cute_col="CuTe-TF32",
            baseline_col="SDPA-TF32",
        )
        _print_latency_note()


if __name__ == "__main__":
    main()
