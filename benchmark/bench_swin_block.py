"""Benchmark: Swin3DTransformerBlock — pure PyTorch vs all-Triton/CuTe optimizations.

Accuracy comparisons (same weights):
  A. BF16 Baseline  vs  BF16 Optimized        → kernel fidelity
  B. FP32+autocast Baseline  vs  BF16 Optimized → autocast vs explicit BF16
  C. FP32 strict Baseline  vs  BF16 Optimized   → full FP32 reference

Timing & memory columns:
  FP32-strict  |  BF16-BL  |  BF16-OPT

Run:
    uv run python benchmark/bench_swin_block.py
"""

import copy
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aurora"))

from aurora.model.swin3d import Swin3DTransformerBlock
from aurora.model.workspace_pool import InferenceWorkspacePool

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


def fwd(blk, x, c, res, pool=None, autocast=False):
    if pool is not None:
        blk._layout_pool = pool
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else _nullctx()
    with torch.no_grad(), ac:
        return blk(x, c, res, rollout_step=0)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def fmt_mb(n): return f"{n/1024**2:.1f} MB"


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def verify_accuracy():
    print("=" * 80)
    print("NUMERICAL ACCURACY  (same weights, ref = Optimized BF16)")
    print("=" * 80)

    col_l = 14
    col_c = 38
    col_e = 12
    hdr = (f"{'Shape':<{col_l}}"
           f"{'Comparison':<{col_c}}"
           f"{'max|err|':>{col_e}}"
           f"{'mean|err|':>{col_e}}"
           f"{'cosine':>{col_e}}")
    print(hdr)
    print("-" * len(hdr))

    def row(a, b, lshape, lcmp):
        a, b = a.float(), b.float()
        d = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)).item()
        print(f"{lshape:<{col_l}}{lcmp:<{col_c}}"
              f"{d.max().item():>{col_e}.3e}"
              f"{d.mean().item():>{col_e}.3e}"
              f"{cos:>{col_e}.8f}")

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
            bl_bf16  = copy.deepcopy(ref_fp32).to(torch.bfloat16)  # same arch, BF16 weights
            op_bf16  = make_block(D, nh, shift, optimized=True, dtype=torch.bfloat16)
            op_bf16.load_state_dict(bl_bf16.state_dict())           # same weights, opt kernels
            pool = InferenceWorkspacePool()

            with torch.no_grad():
                op_bf16._layout_pool = pool
                out_opt  = op_bf16(x16, c16, res, rollout_step=0)

                # A: BF16 Baseline vs BF16 Optimized  (kernel fidelity)
                out_bl16 = bl_bf16(x16, c16, res, rollout_step=0)
                row(out_bl16, out_opt, lbl, "A: BF16 BL  vs  BF16 OPT")

                # B: FP32+autocast Baseline vs BF16 Optimized
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ac = ref_fp32(x32, c32, res, rollout_step=0)
                row(out_ac, out_opt, lbl, "B: FP32+autocast BL  vs  BF16 OPT")

                # C: Strict FP32 Baseline vs BF16 Optimized  (full FP32 reference)
                out_fp32 = ref_fp32(x32, c32, res, rollout_step=0)
                row(out_fp32, out_opt, lbl, "C: FP32 strict BL  vs  BF16 OPT")

        print()


# ---------------------------------------------------------------------------
# timing + memory
# ---------------------------------------------------------------------------

def run_bench():
    print("=" * 80)
    print(f"TIMING & PEAK MEMORY  (B={B}  warmup={WARMUP}  measured={MEASURED})")
    print("=" * 80)

    col_l = 20
    col_t = 14
    col_m = 13
    col_s = 9
    hdr = (
        f"{'Shape':<{col_l}}"
        f"{'FP32-BL':>{col_t}}"
        f"{'BF16-BL':>{col_t}}"
        f"{'BF16-OPT':>{col_t}}"
        f"{'OPT/FP32':>{col_s}}"
        f"{'OPT/BL16':>{col_s}}"
        f"{'FP32 mem':>{col_m}}"
        f"{'BL16 mem':>{col_m}}"
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

            # FP32 strict baseline
            blk_fp32 = make_block(D, nh, shift, optimized=False, dtype=torch.float32)
            fwd(blk_fp32, x32, c32, res)
            t_fp32 = bench(lambda b=blk_fp32: fwd(b, x32, c32, res))
            m_fp32 = peak_mem(lambda b=blk_fp32: fwd(b, x32, c32, res))

            # BF16 baseline
            blk_bl16 = make_block(D, nh, shift, optimized=False, dtype=torch.bfloat16)
            fwd(blk_bl16, x16, c16, res)
            t_bl16 = bench(lambda b=blk_bl16: fwd(b, x16, c16, res))
            m_bl16 = peak_mem(lambda b=blk_bl16: fwd(b, x16, c16, res))

            # BF16 optimized + pool
            blk_opt = make_block(D, nh, shift, optimized=True,  dtype=torch.bfloat16)
            pool = InferenceWorkspacePool()
            fwd(blk_opt, x16, c16, res, pool)
            t_opt  = bench(lambda b=blk_opt, p=pool: fwd(b, x16, c16, res, p))
            m_opt  = peak_mem(lambda b=blk_opt, p=pool: fwd(b, x16, c16, res, p))

            print(
                f"{lbl:<{col_l}}"
                f"{t_fp32*1000:>{col_t}.1f}"
                f"{t_bl16*1000:>{col_t}.1f}"
                f"{t_opt*1000:>{col_t}.1f}"
                f"{t_fp32/t_opt:>{col_s-1}.2f}x"
                f"{t_bl16/t_opt:>{col_s-1}.2f}x"
                f"{fmt_mb(m_fp32):>{col_m}}"
                f"{fmt_mb(m_bl16):>{col_m}}"
                f"{fmt_mb(m_opt):>{col_m}}"
            )

    print()
    print("Time in µs. OPT/FP32 = speedup vs FP32-strict; OPT/BL16 = speedup vs BF16-baseline.")
    print("Mem = peak extra alloc above model weights.")


# ---------------------------------------------------------------------------

def main():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"GPU : {props.name}  (SM{props.major}{props.minor})\n")
    verify_accuracy()
    run_bench()


if __name__ == "__main__":
    main()
