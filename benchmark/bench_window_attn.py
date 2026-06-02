"""Benchmark: CuTe window attention vs torch SDPA.

Sections: compact accuracy summary, unmasked perf (micro + ERA5), masked Swin (-100) ERA5 enc.

Run:
    uv run python benchmark/bench_window_attn.py
    BENCH_MEASURED=200 uv run python benchmark/bench_window_attn.py  # faster
"""

from __future__ import annotations

import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable, Optional

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.ops.cute.window_attn_fwd import (
    _choose_tile_n,
    _choose_tile_n_tf32,
    _CUTE_AVAILABLE,
    _CUTE_KERNEL_VERSION,
    WinAttnPrecision,
    window_attn_fwd_cute,
    _expand_bias_for_sdpa,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
MEASURED = int(os.environ.get("BENCH_MEASURED", "1000"))
TRIM_FRAC = 0.05

SHAPES_MICRO = [
    (16, 8, 144, 64, "N144 H8"),
    (8, 8, 288, 64, "N288 H8"),
    (2, 32, 576, 64, "N576 H32 stream"),
]

SHAPES_ERA5 = [
    (1800, 8, 144, 64, "enc1 1800×8"),
    (450, 16, 144, 64, "enc2 450×16"),
    (128, 32, 144, 64, "enc3 128×32"),
]

# Swin shifted-window mask (-100); encoder shapes only (production N=144).
SHAPES_MASKED = SHAPES_ERA5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def attention_flops(Bwin: int, H: int, N: int, Dh: int) -> int:
    return 4 * Bwin * H * N * N * Dh


@dataclass
class BenchStats:
    mean: float
    ci95: float


def bench(fn, warmup: int = WARMUP, measured: int = MEASURED) -> BenchStats:
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
    n = len(times)
    trim = max(1, int(n * TRIM_FRAC))
    core = times[trim : n - trim]
    tmean = statistics.mean(core)
    tstd = statistics.stdev(core) if len(core) > 1 else 0.0
    ci95 = 1.96 * tstd / math.sqrt(len(core))
    return BenchStats(mean=tmean, ci95=ci95)


def make_qkv(Bwin, H, N, Dh, dtype, device="cuda"):
    g = torch.Generator(device=device).manual_seed(42)
    kw = dict(dtype=dtype, device=device, generator=g)
    s = 0.1
    return (
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
        torch.randn(Bwin, H, N, Dh, **kw) * s,
    )


def make_swin_bias(nW: int, N: int, device: str = "cuda") -> torch.Tensor:
    """(nW, N, N) float bias with -100 on upper-right / lower-left blocks (Swin-style)."""
    bias = torch.zeros(nW, N, N, dtype=torch.float32, device=device)
    bias[:, N // 2 :, : N // 2] = -100.0
    return bias


def fp32_sdpa(q, k, v, scale, bias: Optional[torch.Tensor] = None):
    Bwin, H, N, _ = q.shape
    mask = _expand_bias_for_sdpa(bias, Bwin, H, N) if bias is not None else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def check_accuracy_batch(
    shapes: list[tuple[int, int, int, int, str]],
    *,
    masked: bool,
    rtol_bf16: float = 2e-2,
    atol_bf16: float = 2e-2,
    rtol_tf32: float = 1e-3,
    atol_tf32: float = 1e-3,
) -> tuple[bool, float, float]:
    """Return (all_pass, worst_bf16_err, worst_tf32_err)."""
    scale = 1.0 / math.sqrt(64)
    nW = 1
    worst_bf, worst_tf = 0.0, 0.0
    all_ok = True
    if not _CUTE_AVAILABLE:
        return True, 0.0, 0.0

    with torch.no_grad():
        for Bwin, H, N, Dh, _ in shapes:
            bias = make_swin_bias(nW, N) if masked else None
            q_bf, k_bf, v_bf = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
            ref_bf = fp32_sdpa(
                q_bf.float(), k_bf.float(), v_bf.float(), scale, bias=bias,
            ).bfloat16()
            out_bf = window_attn_fwd_cute(
                q_bf, k_bf, v_bf, bias=bias,
                precision=WinAttnPrecision.BF16_MIXED, scale_qk=scale,
            )
            err_bf = _max_abs(out_bf, ref_bf)
            worst_bf = max(worst_bf, err_bf)
            if not torch.allclose(out_bf.float(), ref_bf.float(), rtol=rtol_bf16, atol=atol_bf16):
                all_ok = False

            q, k, v = make_qkv(Bwin, H, N, Dh, torch.float32)
            ref_tf = fp32_sdpa(q, k, v, scale, bias=bias)
            out_tf = window_attn_fwd_cute(
                q, k, v, bias=bias,
                precision=WinAttnPrecision.TF32_ACC_FP32, scale_qk=scale,
            )
            err_tf = _max_abs(out_tf, ref_tf)
            worst_tf = max(worst_tf, err_tf)
            if not torch.allclose(out_tf, ref_tf, rtol=rtol_tf32, atol=atol_tf32):
                all_ok = False
    return all_ok, worst_bf, worst_tf


_COL_LABEL = 22
_COL_MS = 8


def _print_perf_header(title: str, baseline: str) -> None:
    print(f"\n{title}")
    print(
        f"{'shape':<{_COL_LABEL}}"
        f"{'cute_ms':>{_COL_MS}}"
        f"{baseline:>{_COL_MS}}"
        f"{'vs':>6}"
    )
    print("-" * (_COL_LABEL + _COL_MS * 2 + 6))


def _print_perf_row(
    label: str,
    r_cute: BenchStats,
    r_base: BenchStats,
) -> None:
    vs = r_base.mean / r_cute.mean
    print(
        f"{label:<{_COL_LABEL}}"
        f"{r_cute.mean:>{_COL_MS}.3f}"
        f"{r_base.mean:>{_COL_MS}.3f}"
        f"{vs:>6.2f}x"
    )


def run_perf_table(
    shapes: list[tuple[int, int, int, int, str]],
    *,
    title: str,
    dtype: torch.dtype,
    cute_precision: WinAttnPrecision,
    baseline_col: str,
    make_baseline: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]],
        Callable[[], None],
    ],
    bias: Optional[torch.Tensor] = None,
) -> None:
    _print_perf_header(title, baseline_col)
    scale = 1.0 / math.sqrt(64)
    with torch.no_grad():
        for Bwin, H, N, Dh, label in shapes:
            q, k, v = make_qkv(Bwin, H, N, Dh, dtype)
            b = bias
            if bias is not None and bias.shape[-1] != N:
                b = make_swin_bias(bias.shape[0], N, device=str(q.device))

            def run_cute():
                window_attn_fwd_cute(
                    q, k, v, bias=b, precision=cute_precision, scale_qk=scale,
                )

            r_cute = bench(run_cute)
            r_base = bench(make_baseline(q, k, v, scale, b))
            _print_perf_row(label, r_cute, r_base)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark.")
        return

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    cute_on = os.environ.get("AURORA_CUTE_WINDOW_ATTN", "1") != "0"
    print(
        f"GPU {props.name} SM{props.major}{props.minor} | "
        f"CuTe={_CUTE_AVAILABLE} kernel={_CUTE_KERNEL_VERSION} "
        f"cute_attn={cute_on} | measured={MEASURED}"
    )

    acc_shapes = SHAPES_MICRO + SHAPES_ERA5
    ok_nomask, wbf, wtf = check_accuracy_batch(acc_shapes, masked=False)
    ok_mask, wbf_m, wtf_m = check_accuracy_batch(SHAPES_MASKED, masked=True)
    tag = lambda ok: "PASS" if ok else "FAIL"
    print(
        f"Accuracy vs FP32 SDPA: nomask {tag(ok_nomask)} "
        f"(max_abs BF16={wbf:.2e} TF32={wtf:.2e}) | "
        f"masked -100 {tag(ok_mask)} "
        f"(max_abs BF16={wbf_m:.2e} TF32={wtf_m:.2e})"
    )

    def sdpa_bf16(q, k, v, s, bias=None):
        mask = None
        if bias is not None:
            mask = _expand_bias_for_sdpa(
                bias, q.shape[0], q.shape[1], q.shape[2],
            ).to(dtype=q.dtype)
        return lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=s)

    def sdpa_fp32(q, k, v, s, bias=None):
        return lambda: fp32_sdpa(q, k, v, s, bias=bias)

    # --- Unmasked perf (compact) ---
    run_perf_table(
        SHAPES_MICRO,
        title="No mask — micro shapes (BF16 CuTe vs BF16 SDPA)",
        dtype=torch.bfloat16,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        baseline_col="sdpa_ms",
        make_baseline=sdpa_bf16,
    )
    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_MICRO,
            title="No mask — micro shapes (TF32 CuTe vs FP32 SDPA)",
            dtype=torch.float32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            baseline_col="sdpa_ms",
            make_baseline=sdpa_fp32,
        )

    run_perf_table(
        SHAPES_ERA5,
        title="No mask — ERA5 windows (BF16)",
        dtype=torch.bfloat16,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        baseline_col="sdpa_ms",
        make_baseline=sdpa_bf16,
    )
    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_ERA5,
            title="No mask — ERA5 windows (TF32; SDPA is true FP32 matmul)",
            dtype=torch.float32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            baseline_col="sdpa_ms",
            make_baseline=sdpa_fp32,
        )

    # --- Masked Swin bias -100 (ERA5 encoder) ---
    print("\nMasked Swin bias -100 (ERA5 encoder, nW=1)")
    bias144 = make_swin_bias(1, 144)
    run_perf_table(
        SHAPES_MASKED,
        title="Masked — BF16 CuTe vs BF16 SDPA + attn_mask",
        dtype=torch.bfloat16,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        baseline_col="sdpa_ms",
        make_baseline=sdpa_bf16,
        bias=bias144,
    )
    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_MASKED,
            title="Masked — TF32 CuTe vs FP32 SDPA + attn_mask",
            dtype=torch.float32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            baseline_col="sdpa_ms",
            make_baseline=sdpa_fp32,
            bias=bias144,
        )

    print(
        f"\nLatency: trimmed mean of {MEASURED} runs "
        f"(drop {int(TRIM_FRAC * 100)}% tails). vs = baseline/cute (>1 faster)."
    )


if __name__ == "__main__":
    main()
