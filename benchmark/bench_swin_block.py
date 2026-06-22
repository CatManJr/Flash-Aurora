"""Benchmark: Swin3DTransformerBlock - pure PyTorch vs Triton/CuTe optimizations.

Accuracy (same weights; error = |candidate − baseline|):
  A. BF16 OPT  vs  BF16 BL
  B. BF16 OPT  vs  FP32+autocast BL
  C. BF16 OPT  vs  FP32 strict SDPA BL
  D. TF32 OPT  vs  FP32 strict SDPA BL
  E. TF32 OPT  vs  SDPA-TF32 BL  (1xTF32 fair compare)

Timing & memory (latency in µs):
  FP32-strict | SDPA-TF32 | TF32-OPT | T32/TF32 | T32/str | BF16-BL | BF16-OPT

Run:
    uv run python benchmark/bench_swin_block.py
"""

import copy
import contextlib
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
from aurora.ops.cute.window_attn_fwd import _CUTE_AVAILABLE

WARMUP    = 20
MEASURED  = 100
WINDOW_SIZE = (2, 6, 12)
B = 1

# Aurora ERA5 shapes
# Stage1: (4,180,360) D=512  H=8
# Stage2: (4,90,180)  D=1024 H=16
# Stage3: (4,45,90)   D=2048 H=32
SHAPES = [
    (4, 180, 360,  512,  8, "Stage1"),
    (4,  90, 180, 1024, 16, "Stage2"),
    (4,  45,  90, 2048, 32, "Stage3"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bench(fn):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(MEASURED):
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    trim = max(1, int(len(times) * 0.05))
    return statistics.mean(times[trim:-trim])  # ms


def peak_mem(fn) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def make_block(D, num_heads, shift, optimized: bool, dtype=torch.float32):
    blk = Swin3DTransformerBlock(
        dim=D, num_heads=num_heads, time_dim=D,
        window_size=WINDOW_SIZE,
        shift_size=(1, 3, 6) if shift else (0, 0, 0),
        mlp_ratio=4.0,
        use_triton_layout=optimized,
        use_triton_adaln=optimized,
        use_triton_mlp=optimized,
        use_cute_window_attn=optimized,
    ).to(device="cuda", dtype=dtype).eval()
    return blk


@contextlib.contextmanager
def _matmul_tf32_ctx(allow_tf32: bool):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def fwd(blk, x, c, res, pool=None, autocast=False, matmul_tf32: bool | None = None):
    if pool is not None:
        blk._layout_pool = pool
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    if matmul_tf32 is None:
        mm = contextlib.nullcontext()
    else:
        mm = _matmul_tf32_ctx(matmul_tf32)
    with torch.no_grad(), ac, mm:
        return blk(x, c, res, rollout_step=0)


def fmt_mb(n): return f"{n/1024**2:.1f} MB"


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def verify_accuracy():
    print("=" * 80)
    print("NUMERICAL ACCURACY  (same weights; |candidate − baseline|)")
    print("=" * 80)
    if not _CUTE_AVAILABLE:
        print("  [note] CuTeDSL unavailable — TF32 rows skipped.\n")

    col_l = 14
    col_c = 38
    col_e = 14
    hdr = (f"{'Shape':<{col_l}}"
           f"{'candidate (baseline)':<{col_c}}"
           f"{'max|err|':>{col_e}}"
           f"{'mean|err|':>{col_e}}"
           f"{'cosine':>{col_e}}")
    print(hdr)
    print("-" * len(hdr))

    def row(candidate, baseline, lshape, label, err_fmt: str = ".6e"):
        cand = candidate.float()
        base = baseline.float()
        d = (cand - base).abs()
        cos = torch.nn.functional.cosine_similarity(
            cand.reshape(1, -1), base.reshape(1, -1)).item()
        print(f"{lshape:<{col_l}}{label:<{col_c}}"
              f"{d.max().item():>{col_e}{err_fmt}}"
              f"{d.mean().item():>{col_e}{err_fmt}}"
              f"{cos:>{col_e}.8f}")
        if d.max().item() == 0.0:
            print(f"{'':<{col_l}}{'  (bitwise equal to baseline)':<{col_c}}")

    for sw_lbl, shift in [("W", False), ("SW", True)]:
        for C, H, W, D, nh, slbl in SHAPES:
            res = (C, H, W)
            L   = C * H * W
            lbl = f"{slbl} {sw_lbl}"

            torch.manual_seed(0)
            x32 = torch.randn(B, L, D, device="cuda")
            c32 = torch.randn(B, D,   device="cuda")
            x16 = x32.bfloat16()
            c16 = c32.bfloat16()

            # All three share the same random weights.
            # ref_fp32 is the source of truth; BF16 copies cast its weights.
            ref_fp32 = make_block(D, nh, shift, optimized=False, dtype=torch.float32)
            bl_bf16  = copy.deepcopy(ref_fp32).to(torch.bfloat16)
            op_bf16  = make_block(D, nh, shift, optimized=True, dtype=torch.bfloat16)
            op_bf16.load_state_dict(bl_bf16.state_dict())
            op_tf32  = make_block(D, nh, shift, optimized=True, dtype=torch.float32)
            op_tf32.load_state_dict(ref_fp32.state_dict())
            pool = InferenceWorkspacePool()

            with torch.no_grad():
                out_bl16 = bl_bf16(x16, c16, res, rollout_step=0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ac = ref_fp32(x32, c32, res, rollout_step=0)
                with _matmul_tf32_ctx(False):
                    out_fp32_strict = ref_fp32(x32, c32, res, rollout_step=0)
                with _matmul_tf32_ctx(True):
                    out_sdpa_tf32 = ref_fp32(x32, c32, res, rollout_step=0)

                op_bf16._layout_pool = pool
                out_bf16_opt = op_bf16(x16, c16, res, rollout_step=0)
                row(out_bf16_opt, out_bl16, lbl, "BF16 OPT  (BL: BF16 PyTorch)")

                row(out_bf16_opt, out_ac, lbl, "BF16 OPT  (BL: FP32+autocast)")

                row(out_bf16_opt, out_fp32_strict, lbl, "BF16 OPT  (BL: FP32 strict)")

                if _CUTE_AVAILABLE:
                    op_tf32._layout_pool = pool
                    with _matmul_tf32_ctx(True):
                        out_tf32_opt = op_tf32(x32, c32, res, rollout_step=0)
                    row(
                        out_tf32_opt, out_fp32_strict, lbl,
                        "TF32 OPT  (BL: FP32 strict SDPA)",
                        err_fmt=".6e",
                    )
                    row(
                        out_tf32_opt, out_sdpa_tf32, lbl,
                        "TF32 OPT  (BL: SDPA-TF32)",
                        err_fmt=".6e",
                    )

        print()


# ---------------------------------------------------------------------------
# timing + memory
# ---------------------------------------------------------------------------

def run_bench():
    print("=" * 80)
    print(f"TIMING & PEAK MEMORY  (B={B}  warmup={WARMUP}  measured={MEASURED})")
    print("=" * 80)

    col_l = 18
    col_t = 11
    col_m = 11
    col_s = 7
    hdr = (
        f"{'Shape':<{col_l}}"
        f"{'FP32-str':>{col_t}}"
        f"{'FP32-TF32*':>{col_t}}"
        f"{'TF32-OPT':>{col_t}}"
        f"{'T/TF32*':>{col_s}}"
        f"{'T/str':>{col_s}}"
        f"{'BF16-BL':>{col_t}}"
        f"{'BF16-OPT':>{col_t}}"
        f"{'B16/BL':>{col_s}}"
        f"{'TF32 mem':>{col_m}}"
        f"{'OPT mem':>{col_m}}"
    )
    sep = "-" * len(hdr)

    for sw_lbl, shift in [("W  (no shift)", False), ("SW (shifted)", True)]:
        print(f"\n=== {sw_lbl} ===")
        print(hdr); print(sep)

        for C, H, W, D, nh, slbl in SHAPES:
            res  = (C, H, W)
            L    = C * H * W
            lbl  = f"{slbl} {sw_lbl}"
            x32  = torch.randn(B, L, D, device="cuda", dtype=torch.float32)
            c32  = torch.randn(B, D,   device="cuda", dtype=torch.float32)
            x16  = x32.bfloat16()
            c16  = c32.bfloat16()

            blk_fp32 = make_block(D, nh, shift, optimized=False, dtype=torch.float32)

            fwd(blk_fp32, x32, c32, res, matmul_tf32=False)
            t_fp32_strict = bench(
                lambda b=blk_fp32: fwd(b, x32, c32, res, matmul_tf32=False)
            )

            fwd(blk_fp32, x32, c32, res, matmul_tf32=True)
            t_sdpa_tf32 = bench(
                lambda b=blk_fp32: fwd(b, x32, c32, res, matmul_tf32=True)
            )

            pool = InferenceWorkspacePool()
            if _CUTE_AVAILABLE:
                blk_tf32 = make_block(D, nh, shift, optimized=True, dtype=torch.float32)
                blk_tf32.load_state_dict(blk_fp32.state_dict())
                fwd(blk_tf32, x32, c32, res, pool, matmul_tf32=True)
                t_tf32_opt = bench(
                    lambda b=blk_tf32, p=pool: fwd(b, x32, c32, res, p, matmul_tf32=True)
                )
                m_tf32 = peak_mem(
                    lambda b=blk_tf32, p=pool: fwd(b, x32, c32, res, p, matmul_tf32=True)
                )
                spd_tf32 = t_sdpa_tf32 / t_tf32_opt
                spd_strict = t_fp32_strict / t_tf32_opt
                t_tf32_us = t_tf32_opt * 1000
                m_tf32_s = fmt_mb(m_tf32)
            else:
                t_tf32_us = float("nan")
                spd_tf32 = float("nan")
                spd_strict = float("nan")
                m_tf32_s = "n/a"

            # BF16 baseline
            blk_bl16 = make_block(D, nh, shift, optimized=False, dtype=torch.bfloat16)
            fwd(blk_bl16, x16, c16, res)
            t_bl16 = bench(lambda b=blk_bl16: fwd(b, x16, c16, res))
            m_bl16 = peak_mem(lambda b=blk_bl16: fwd(b, x16, c16, res))

            # BF16 optimized
            blk_opt = make_block(D, nh, shift, optimized=True, dtype=torch.bfloat16)
            fwd(blk_opt, x16, c16, res, pool)
            t_opt = bench(lambda b=blk_opt, p=pool: fwd(b, x16, c16, res, p))
            m_opt = peak_mem(lambda b=blk_opt, p=pool: fwd(b, x16, c16, res, p))

            if _CUTE_AVAILABLE:
                tf32_time_s = f"{t_tf32_us:>{col_t}.1f}"
                spd_tf32_s = f"{spd_tf32:>{col_s}.2f}x"
                spd_strict_s = f"{spd_strict:>{col_s}.2f}x"
            else:
                tf32_time_s = f"{'n/a':>{col_t}}"
                spd_tf32_s = f"{'n/a':>{col_s}}"
                spd_strict_s = f"{'n/a':>{col_s}}"

            print(
                f"{lbl:<{col_l}}"
                f"{t_fp32_strict*1000:>{col_t}.1f}"
                f"{t_sdpa_tf32*1000:>{col_t}.1f}"
                f"{tf32_time_s}"
                f"{spd_tf32_s}"
                f"{spd_strict_s}"
                f"{t_bl16*1000:>{col_t}.1f}"
                f"{t_opt*1000:>{col_t}.1f}"
                f"{t_bl16/t_opt:>{col_s}.2f}x"
                f"{m_tf32_s:>{col_m}}"
                f"{fmt_mb(m_opt):>{col_m}}"
            )

    print()
    print("Time in µs. FP32-str = PyTorch BL, TF32 off. FP32-TF32* = PyTorch BL, allow_tf32=True.")
    print("* Note: allow_tf32 speeds up linear projections (Triton/cuBLAS) but has NO effect on")
    print("  FP32 SDPA itself — PyTorch selects mem-efficient/math backend which ignores the flag.")
    print("  TF32-OPT uses CuTe TF32 attention kernel, which IS faster than FP32 SDPA.")
    print("TF32-OPT = CuTe TF32_ACC_FP32 + Triton (allow_tf32=True for projections).")
    print("T/TF32* = OPT speedup vs FP32-TF32* (projections at same speed; attn kernel differs).")
    print("T/str = OPT vs strict FP32 (quality ref).  Mem = peak extra alloc.")


# ---------------------------------------------------------------------------

def main():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"GPU : {props.name}  (SM{props.major}{props.minor})")
    print(f"CuTe TF32 window attn : {_CUTE_AVAILABLE}\n")
    verify_accuracy()
    run_bench()


if __name__ == "__main__":
    main()
