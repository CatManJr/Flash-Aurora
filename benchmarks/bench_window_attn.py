"""Benchmark: CuTe BF16 window attention vs torch SDPA (flash attention).

Measures forward-pass latency and TFLOPS for representative window shapes:

  Aurora encoder (window_size = 2×6×12):
    N=144  all encoder/decoder stages at standard resolution
    N=288  2× spatial resolution
    N=576  4× spatial resolution — streaming (multi-pass) on 99 KB SMEM

  Swin3D (window_size = 2×7×7 / 4×7×7 / 7×7×7):
    N=98   partial-tile (not a multiple of 16) — exercises predicated loads
    N=196  two full tiles — clean multi-pass
    N=343  streaming — large window, comparable to N=576 enc case

Also performs a numerical accuracy check (CuTe vs FP32 SDPA) for every shape.

Run:
    uv run python benchmarks/bench_window_attn.py
"""

import math
import os
import sys

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

WARMUP   = 20   # iterations discarded
MEASURED = 100  # iterations timed

USE_CUTE_KERNEL = os.environ.get("AURORA_CUTE_WINDOW_ATTN", "") == "1"

# ---------------------------------------------------------------------------
# Shapes: (Bwin, H, N, Dh, label)
#
#  N = Wc × Wh × Ww
#
#  Aurora encoder  (window_size = 2×6×12, all stages use the same window)
#   144 = 2×6×12   standard resolution
#   288 = 2×6×24   2× spatial resolution
#   576 = 2×12×24  4× spatial resolution — streaming on 99 KB SMEM
#
#  Swin3D  (common standalone backbone configurations)
#    98 = 2×7×7    partial-tile: N not a multiple of 16 → predicated loads
#   196 = 4×7×7    two full tiles: clean two-pass path
#   343 = 7×7×7    large window, streaming (comparable to enc N=576)
# ---------------------------------------------------------------------------

SHAPES = [
    # (Bwin,  H,   N,  Dh,  label)
    # ── Aurora encoder ──────────────────────────────────────────────────────
    (16,   8, 144, 64, "aurora  H=8  N=144 (2×6×12)"),
    ( 8,  16, 144, 64, "aurora  H=16 N=144 (2×6×12)"),
    ( 4,  32, 144, 64, "aurora  H=32 N=144 (2×6×12)"),
    ( 8,   8, 288, 64, "aurora  H=8  N=288 (2×6×24) 2× spatial"),
    ( 4,  16, 288, 64, "aurora  H=16 N=288 (2×6×24) 2× spatial"),
    ( 2,  32, 576, 64, "aurora  H=32 N=576 (2×12×24) streaming"),
    # ── Swin3D ──────────────────────────────────────────────────────────────
    (16,   8,  98, 64, "swin3d  H=8  N=98  (2×7×7)  partial-tile"),
    ( 8,  16,  98, 64, "swin3d  H=16 N=98  (2×7×7)  partial-tile"),
    ( 4,  32,  98, 64, "swin3d  H=32 N=98  (2×7×7)  partial-tile"),
    ( 8,   8, 196, 64, "swin3d  H=8  N=196 (4×7×7)  two-pass"),
    ( 4,  16, 196, 64, "swin3d  H=16 N=196 (4×7×7)  two-pass"),
    ( 2,  32, 343, 64, "swin3d  H=32 N=343 (7×7×7)  streaming"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def attention_flops(Bwin: int, H: int, N: int, Dh: int) -> int:
    """Forward-pass FLOPs: 2 × (Q@K^T) + 2 × (P@V), each = 2·Bwin·H·N·N·Dh."""
    return 4 * Bwin * H * N * N * Dh


def tflops(flop: int, elapsed_ms: float) -> float:
    return flop / elapsed_ms / 1e9  # ms → s · 1e3, flop → TFLOP · 1e12 → net /1e9


def bench(fn, warmup: int = WARMUP, measured: int = MEASURED) -> float:
    """Return median latency in ms over `measured` iterations."""
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
    return times[len(times) // 2]  # median


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

    # Header
    col_label   = 36
    col_n       = 5
    col_tile    = 9
    col_ms      = 9
    col_tflops  = 9

    hdr = (
        f"{'Shape':<{col_label}}"
        f"{'N':>{col_n}}"
        f"{'tile_n':>{col_tile}}"
        f"{'pass':>6}"
        f"{'CuTe ms':>{col_ms}}"
        f"{'TFLOPS':>{col_tflops}}"
        f"{'SDPA ms':>{col_ms}}"
        f"{'TFLOPS':>{col_tflops}}"
        f"{'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    with torch.no_grad():
        for Bwin, H, N, Dh, label in SHAPES:
            q_bf, k_bf, v_bf = make_qkv(Bwin, H, N, Dh, torch.bfloat16)
            flop = attention_flops(Bwin, H, N, Dh)

            tile_n    = _choose_tile_n(N, head_dim=Dh)
            n_passes  = math.ceil(N / tile_n)
            pass_str  = "1" if n_passes == 1 else str(n_passes)

            # --- CuTe BF16 kernel ---
            def run_cute():
                window_attn_fwd_cute(
                    q_bf, k_bf, v_bf,
                    precision=WinAttnPrecision.BF16_MIXED,
                    scale_qk=scale,
                )

            ms_cute   = bench(run_cute)
            tf_cute   = tflops(flop, ms_cute)

            # --- torch SDPA (flash attention backend) ---
            def run_sdpa():
                F.scaled_dot_product_attention(q_bf, k_bf, v_bf, scale=scale)

            ms_sdpa   = bench(run_sdpa)
            tf_sdpa   = tflops(flop, ms_sdpa)

            speedup   = ms_sdpa / ms_cute

            print(
                f"{label:<{col_label}}"
                f"{N:>{col_n}}"
                f"{tile_n:>{col_tile}}"
                f"{pass_str:>6}"
                f"{ms_cute:>{col_ms}.3f}"
                f"{tf_cute:>{col_tflops}.2f}"
                f"{ms_sdpa:>{col_ms}.3f}"
                f"{tf_sdpa:>{col_tflops}.2f}"
                f"{speedup:>8.2f}x"
            )

    print()
    print("Note: TFLOPS = 4·Bwin·H·N²·Dh / latency  (forward pass FLOPs only).")


if __name__ == "__main__":
    main()
