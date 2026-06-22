#!/usr/bin/env python3
"""Micro-benchmark: D2 fused AdaLN + residual (:func:`adaptive_layernorm_film_add_residual_forward`)
vs composing ``residual + adaptive_layernorm_film_forward`` (extra global write + add).

Run from the repository root::

    PYTHONPATH=aurora uv run python aurora/profiling_adaln.py
    PYTHONPATH=aurora uv run python aurora/profiling_adaln.py --preset aurora_s2 --l 2048
    PYTHONPATH=aurora uv run python aurora/profiling_adaln.py --sweep-l 512,2048,8192 --report-out profiling/adaln_d2.md

Requires CUDA + float32.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _bench_cuda(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    import torch

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(repeat):
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        times_ms.append(ev0.elapsed_time(ev1))
    med = float(statistics.median(times_ms))
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return med, peak_mb


def main() -> None:
    import torch

    from flash_aurora.aurora.ops.triton_adaln import (
        adaptive_layernorm_film_add_residual_forward,
        adaptive_layernorm_film_forward,
    )

    p = argparse.ArgumentParser(description="Profile AdaLN D2 fused residual vs composed path.")
    p.add_argument(
        "--preset",
        choices=("none", "aurora_s0", "aurora_s1", "aurora_s2"),
        default="aurora_s0",
        help="Aurora backbone dim D: s0=512, s1=1024, s2=2048 (ignores --d unless none).",
    )
    p.add_argument("--b", type=int, default=1, help="Batch size B.")
    p.add_argument("--l", type=int, default=2048, help="Sequence length L (tokens).")
    p.add_argument("--d", type=int, default=512, help="Channel dim D (preset=none).")
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--repeat", type=int, default=80)
    p.add_argument("--report-out", type=str, default="", help="Optional Markdown path.")
    p.add_argument(
        "--sweep-l",
        type=str,
        default="",
        help="Comma-separated L values (e.g. 512,2048,8192); overrides --l.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    if args.preset == "aurora_s0":
        d_dim = 512
    elif args.preset == "aurora_s1":
        d_dim = 1024
    elif args.preset == "aurora_s2":
        d_dim = 2048
    else:
        d_dim = args.d

    sweep_l: list[int] = []
    if args.sweep_l.strip():
        sweep_l = [int(x.strip()) for x in args.sweep_l.split(",") if x.strip()]
        if not sweep_l:
            print("--sweep-l had no integers", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda")
    scale_bias = 0.0
    eps = 1e-5

    def run_shapes(B: int, L: int, D: int) -> dict[str, float]:
        torch.manual_seed(0)
        residual = torch.randn(B, L, D, device=device, dtype=torch.float32)
        x = torch.randn(B, L, D, device=device, dtype=torch.float32)
        scale = torch.randn(B, 1, D, device=device, dtype=torch.float32)
        shift = torch.randn(B, 1, D, device=device, dtype=torch.float32)

        out_fused = torch.empty_like(x)

        def composed() -> None:
            nonlocal out_fused
            h = adaptive_layernorm_film_forward(x, scale, shift, scale_bias, eps)
            out_fused = residual + h

        def fused() -> None:
            nonlocal out_fused
            out_fused = adaptive_layernorm_film_add_residual_forward(
                residual, x, scale, shift, scale_bias, eps
            )

        with torch.no_grad():
            composed()
            fused()
            ref = residual + adaptive_layernorm_film_forward(x, scale, shift, scale_bias, eps)
            out_fused = adaptive_layernorm_film_add_residual_forward(
                residual, x, scale, shift, scale_bias, eps
            )
            err = (out_fused - ref).abs().max().item()

        torch.cuda.reset_peak_memory_stats()
        med_c, peak_c = _bench_cuda(composed, warmup=args.warmup, repeat=args.repeat)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        med_f, peak_f = _bench_cuda(fused, warmup=args.warmup, repeat=args.repeat)

        return {
            "err": float(err),
            "composed_ms": med_c,
            "fused_ms": med_f,
            "peak_c_mb": peak_c,
            "peak_f_mb": peak_f,
        }

    lines: list[str] = []
    if sweep_l:
        print(f"[sweep] preset={args.preset} D={d_dim}, B={args.b}, L in {sweep_l}")
        lines.append("| L | composed ms | fused ms | composed/fused | max|err| | peak CUDA MB (c / f) |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for L in sweep_l:
            s = run_shapes(args.b, L, d_dim)
            ratio = s["composed_ms"] / s["fused_ms"] if s["fused_ms"] > 0 else float("nan")
            lines.append(
                f"| {L} | {s['composed_ms']:.4f} | {s['fused_ms']:.4f} | {ratio:.3f}x | {s['err']:.3e} | "
                f"{s['peak_c_mb']:.1f} / {s['peak_f_mb']:.1f} |"
            )
            print(
                f"L={L:6d}  composed={s['composed_ms']:.4f}ms  fused={s['fused_ms']:.4f}ms  "
                f"ratio={ratio:.3f}x  max|err|={s['err']:.3e}"
            )
        print("\n--- Markdown ---\n")
        print("\n".join(lines))
    else:
        B, L = args.b, args.l
        print(f"[run] preset={args.preset} B={B} L={L} D={d_dim}")
        s = run_shapes(B, L, d_dim)
        ratio = s["composed_ms"] / s["fused_ms"] if s["fused_ms"] > 0 else float("nan")
        print(f"  composed (film + torch add): median {s['composed_ms']:.4f} ms, peak CUDA {s['peak_c_mb']:.1f} MB")
        print(f"  fused (film_add_residual):    median {s['fused_ms']:.4f} ms, peak CUDA {s['peak_f_mb']:.1f} MB")
        print(f"  composed/fused = {ratio:.3f}x  (max|err| check: {s['err']:.3e})")
        lines = [
            f"- B={B}, L={L}, D={d_dim}",
            f"- composed: {s['composed_ms']:.4f} ms",
            f"- fused: {s['fused_ms']:.4f} ms",
            f"- ratio: {ratio:.3f}x",
            f"- max|err|: {s['err']:.3e}",
        ]

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import torch as _t

        body = "\n".join(lines) if sweep_l else "\n".join(f"- {x}" for x in lines)
        text = "\n".join(
            [
                "# AdaLN D2 profiling (Triton)",
                "",
                f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
                f"- PyTorch: {_t.__version__}",
                f"- preset: `{args.preset}`, warmup: {args.warmup}, repeat: {args.repeat}",
                "",
                "**Variants**",
                "",
                "- **composed**: `residual + adaptive_layernorm_film_forward(x, ...)` (Triton AdaLN then PyTorch add).",
                "- **fused**: `adaptive_layernorm_film_add_residual_forward(residual, x, ...)` (single kernel).",
                "",
                "## Results",
                "",
                body,
                "",
            ]
        )
        out.write_text(text, encoding="utf-8")
        print(f"\n[report] {out.resolve()}")


if __name__ == "__main__":
    main()
