"""Benchmark: CuTe window attention vs torch SDPA.

Sections: checkpoint shape coverage, accuracy (all variants), unmasked perf, masked Swin (-100).

Run:
    uv run python benchmark/bench_window_attn.py
    BENCH_MEASURED=200 uv run python benchmark/bench_window_attn.py  # faster
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from _aurora_attn_shapes import (
    CHECKPOINT_VARIANTS,
    SHAPES_ALL_CHECKPOINTS,
    SHAPES_ERA5_025,
    all_unique_attn_shapes,
)
from flash_aurora.aurora.ops.cute.window_attn_fwd import (
    _best_tile_m,
    _choose_tile_n,
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

# All unique (Bwin, H, N, Dh) across the seven Aurora checkpoints (enc + dec).
SHAPES_ALL = SHAPES_ALL_CHECKPOINTS

# 0.25° family encoder stages - kept for focused ERA5 perf titles.
SHAPES_ERA5 = SHAPES_ERA5_025


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


@dataclass
class ShapeCheckResult:
    label: str
    ok: bool
    max_abs: float
    error: str | None = None


def _scale_for_dh(dh: int) -> float:
    return 1.0 / math.sqrt(dh)


def _check_one_shape(
    Bwin: int,
    H: int,
    N: int,
    Dh: int,
    *,
    masked: bool,
    precision: WinAttnPrecision,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> ShapeCheckResult:
    label = f"Bwin={Bwin} H={H} N={N} Dh={Dh}"
    if not _CUTE_AVAILABLE:
        return ShapeCheckResult(label=label, ok=True, max_abs=0.0)

    scale = _scale_for_dh(Dh)
    bias = make_swin_bias(1, N) if masked else None
    try:
        with torch.no_grad():
            q, k, v = make_qkv(Bwin, H, N, Dh, dtype)
            if precision == WinAttnPrecision.BF16_MIXED:
                ref = fp32_sdpa(
                    q.float(), k.float(), v.float(), scale, bias=bias,
                ).bfloat16()
            else:
                ref = fp32_sdpa(q, k, v, scale, bias=bias)
            out = window_attn_fwd_cute(
                q, k, v, bias=bias, precision=precision, scale_qk=scale,
            )
            err = _max_abs(out, ref)
            ok = torch.allclose(
                out.float(), ref.float(), rtol=rtol, atol=atol,
            )
            return ShapeCheckResult(label=label, ok=ok, max_abs=err)
    except Exception as exc:
        return ShapeCheckResult(label=label, ok=False, max_abs=float("inf"), error=str(exc))


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
    worst_bf, worst_tf = 0.0, 0.0
    all_ok = True
    if not _CUTE_AVAILABLE:
        return True, 0.0, 0.0

    for Bwin, H, N, Dh, _ in shapes:
        r_bf = _check_one_shape(
            Bwin, H, N, Dh,
            masked=masked,
            precision=WinAttnPrecision.BF16_MIXED,
            dtype=torch.bfloat16,
            rtol=rtol_bf16,
            atol=atol_bf16,
        )
        r_tf = _check_one_shape(
            Bwin, H, N, Dh,
            masked=masked,
            precision=WinAttnPrecision.TF32_ACC_FP32,
            dtype=torch.float32,
            rtol=rtol_tf32,
            atol=atol_tf32,
        )
        worst_bf = max(worst_bf, r_bf.max_abs)
        worst_tf = max(worst_tf, r_tf.max_abs)
        all_ok = all_ok and r_bf.ok and r_tf.ok
    return all_ok, worst_bf, worst_tf


def run_checkpoint_coverage() -> bool:
    """Per-shape kernel smoke + accuracy for every unique checkpoint geometry."""
    unique = all_unique_attn_shapes()
    n_variants = len(CHECKPOINT_VARIANTS)
    print(
        f"\nCheckpoint shape coverage: {len(unique)} unique geometries "
        f"from {n_variants} variants × enc/dec stages"
    )
    print(
        f"{'shape':<32} {'tile_m/n':>10} {'bf16':>5} {'+mask':>6} "
        f"{'tf32':>5} {'+mask':>6} {'max_abs':>10}  checkpoints"
    )
    print("-" * 110)

    all_ok = True
    if not _CUTE_AVAILABLE:
        print("  CuTe unavailable — skipping coverage.")
        return True

    for shape in unique:
        Bwin, H, N, Dh = shape.bwin, shape.heads, shape.n_tokens, shape.head_dim
        tile_m = _best_tile_m(is_bf16=True, has_bias=False)
        tile_n = _choose_tile_n(N, head_dim=Dh, tile_m=tile_m)
        tile_s = f"{tile_m}/{tile_n}"

        modes = (
            ("bf16", False, WinAttnPrecision.BF16_MIXED, torch.bfloat16, 2e-2, 2e-2),
            ("+mask", True, WinAttnPrecision.BF16_MIXED, torch.bfloat16, 2e-2, 2e-2),
            ("tf32", False, WinAttnPrecision.TF32_ACC_FP32, torch.float32, 1e-3, 1e-3),
            ("+mask", True, WinAttnPrecision.TF32_ACC_FP32, torch.float32, 1e-3, 1e-3),
        )
        cells: list[str] = []
        worst = 0.0
        for _tag, masked, precision, dtype, rtol, atol in modes:
            r = _check_one_shape(
                Bwin, H, N, Dh,
                masked=masked,
                precision=precision,
                dtype=dtype,
                rtol=rtol,
                atol=atol,
            )
            worst = max(worst, r.max_abs)
            if r.error:
                cells.append("ERR")
                all_ok = False
            elif r.ok:
                cells.append("OK")
            else:
                cells.append("FAIL")
                all_ok = False

        variant_short = ", ".join(sorted({v.split("/")[0] for v in shape.variants}))
        if len(variant_short) > 36:
            variant_short = variant_short[:33] + "..."
        print(
            f"{shape.label:<32} {tile_s:>10} "
            f"{cells[0]:>5} {cells[1]:>6} {cells[2]:>5} {cells[3]:>6} "
            f"{worst:>10.2e}  {variant_short}"
        )
        if any(c == "ERR" for c in cells):
            for _tag, masked, precision, dtype, rtol, atol in modes:
                r = _check_one_shape(
                    Bwin, H, N, Dh,
                    masked=masked,
                    precision=precision,
                    dtype=dtype,
                    rtol=rtol,
                    atol=atol,
                )
                if r.error:
                    print(f"    error ({precision.name}, mask={masked}): {r.error}")

    tag = "PASS" if all_ok else "FAIL"
    print(f"\nCoverage summary: {tag} ({len(unique)} shapes × BF16/TF32 × mask/unmask)")
    return all_ok


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
    with torch.no_grad():
        for Bwin, H, N, Dh, label in shapes:
            scale = _scale_for_dh(Dh)
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


def _sdpa_backend_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    bias: Optional[torch.Tensor],
    backend: SDPBackend,
) -> Callable[[], None]:
    mask = None
    if bias is not None:
        mask = _expand_bias_for_sdpa(bias, q.shape[0], q.shape[1], q.shape[2]).to(dtype=q.dtype)

    def run() -> None:
        with sdpa_kernel(backends=[backend]):
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    return run


def _bench_optional(fn: Callable[[], None]) -> tuple[BenchStats | None, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fn()
            torch.cuda.synchronize()
            return bench(fn), ""
    except Exception as exc:
        torch.cuda.synchronize()
        return None, type(exc).__name__


def run_sdpa_backend_probe(
    shapes: list[tuple[int, int, int, int, str]],
    *,
    title: str,
    masked: bool,
) -> None:
    """Force each PyTorch SDPA backend on Aurora BF16 ERA5 shapes."""
    backends = (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_eff", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    )
    print(f"\n{title}")
    print(
        f"{'shape':<{_COL_LABEL}}"
        f"{'cute':>{_COL_MS}}"
        + "".join(f"{name:>{_COL_MS}}" for name, _backend in backends)
    )
    print("-" * (_COL_LABEL + _COL_MS * (1 + len(backends))))
    with torch.no_grad():
        for Bwin, H, N, Dh, label in shapes:
            scale = _scale_for_dh(Dh)
            q, k, v = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
            bias = make_swin_bias(1, N) if masked else None

            def run_cute() -> None:
                window_attn_fwd_cute(
                    q, k, v, bias=bias,
                    precision=WinAttnPrecision.BF16_MIXED,
                    scale_qk=scale,
                )

            cute_stats = bench(run_cute)
            cells = [f"{label:<{_COL_LABEL}}", f"{cute_stats.mean:>{_COL_MS}.3f}"]
            for _name, backend in backends:
                stats, err = _bench_optional(_sdpa_backend_runner(q, k, v, scale, bias, backend))
                if stats is None:
                    cells.append(f"{'n/a':>{_COL_MS}}")
                else:
                    cells.append(f"{stats.mean:>{_COL_MS}.3f}")
            print("".join(cells))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark.")
        return

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"GPU {props.name} SM{props.major}{props.minor} | "
        f"CuTe={_CUTE_AVAILABLE} kernel={_CUTE_KERNEL_VERSION} | measured={MEASURED}"
    )

    coverage_ok = run_checkpoint_coverage()

    acc_shapes = SHAPES_MICRO + SHAPES_ALL
    ok_nomask, wbf, wtf = check_accuracy_batch(acc_shapes, masked=False)
    ok_mask, wbf_m, wtf_m = check_accuracy_batch(SHAPES_ALL, masked=True)
    tag = lambda ok: "PASS" if ok else "FAIL"
    print(
        f"\nAccuracy vs FP32 SDPA (micro + all checkpoints): "
        f"nomask {tag(ok_nomask)} "
        f"(max_abs BF16={wbf:.2e} TF32={wtf:.2e}) | "
        f"masked -100 {tag(ok_mask)} "
        f"(max_abs BF16={wbf_m:.2e} TF32={wtf_m:.2e})"
    )
    if not coverage_ok:
        print("WARNING: checkpoint coverage reported failures above.")

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
        SHAPES_ALL,
        title="No mask — all checkpoint shapes (BF16)",
        dtype=torch.bfloat16,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        baseline_col="sdpa_ms",
        make_baseline=sdpa_bf16,
    )
    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_ALL,
            title="No mask — all checkpoint shapes (TF32; SDPA is true FP32 matmul)",
            dtype=torch.float32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            baseline_col="sdpa_ms",
            make_baseline=sdpa_fp32,
        )

    # --- Masked Swin bias -100 (all checkpoint shapes) ---
    print("\nMasked Swin bias -100 (all checkpoint shapes, nW=1)")
    bias144 = make_swin_bias(1, 144)
    run_perf_table(
        SHAPES_ALL,
        title="Masked — BF16 CuTe vs BF16 SDPA + attn_mask",
        dtype=torch.bfloat16,
        cute_precision=WinAttnPrecision.BF16_MIXED,
        baseline_col="sdpa_ms",
        make_baseline=sdpa_bf16,
        bias=bias144,
    )
    if _CUTE_AVAILABLE:
        run_perf_table(
            SHAPES_ALL,
            title="Masked — TF32 CuTe vs FP32 SDPA + attn_mask",
            dtype=torch.float32,
            cute_precision=WinAttnPrecision.TF32_ACC_FP32,
            baseline_col="sdpa_ms",
            make_baseline=sdpa_fp32,
            bias=bias144,
        )

    run_sdpa_backend_probe(
        SHAPES_ERA5,
        title="Forced SDPA backend probe — BF16 0.25° ERA5 enc (no mask)",
        masked=False,
    )
    run_sdpa_backend_probe(
        SHAPES_ERA5,
        title="Forced SDPA backend probe — BF16 0.25° ERA5 enc (masked -100)",
        masked=True,
    )

    print(
        f"\nLatency: trimmed mean of {MEASURED} runs "
        f"(drop {int(TRIM_FRAC * 100)}% tails). vs = baseline/cute (>1 faster)."
    )


if __name__ == "__main__":
    main()
