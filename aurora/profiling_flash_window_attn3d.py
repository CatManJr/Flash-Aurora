#!/usr/bin/env python3
"""Profile flash_window_attn3d with Aurora-standard tensor shapes.

Run from repository root:
    uv run python aurora/profiling_flash_window_attn3d.py
    uv run python aurora/profiling_flash_window_attn3d.py --preset aurora
    uv run python aurora/profiling_flash_window_attn3d.py --batch-size 2 --with-mask
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aurora.ops.flash_window_attn3d import (  # noqa: E402
    torch_window_attention_3d_reference,
    flash_window_attn_3d_forward,
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _infer_bwin(batch_size: int, patch_res: tuple[int, int, int], window: tuple[int, int, int]) -> int:
    c, h, w = patch_res
    wc, wh, ww = window
    windows_per_sample = _ceil_div(c, wc) * _ceil_div(h, wh) * _ceil_div(w, ww)
    return batch_size * windows_per_sample


def _bench_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(repeat):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / repeat


def _bench_ms_mixed(fn_nomask, fn_mask, mask_ratio: float, warmup: int, repeat: int) -> float:
    # Deterministic schedule: every 100 calls, first K are masked.
    k = int(round(max(0.0, min(1.0, mask_ratio)) * 100))
    for i in range(warmup):
        if (i % 100) < k:
            fn_mask()
        else:
            fn_nomask()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for i in range(repeat):
        if (i % 100) < k:
            fn_mask()
        else:
            fn_nomask()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / repeat


def _clear_cuda_cache() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _choose_window_size(args_window: tuple[int, int, int], *, heads: int, dim: int, device: torch.device) -> tuple[int, int, int]:
    # If user explicitly provides non-default window, respect it.
    if args_window != (2, 6, 12):
        return args_window
    # Auto-switch: prefer Aurora default N=144; fallback to N=96 on lower smem devices.
    dh = dim // heads
    if dh <= 0:
        return args_window
    props = torch.cuda.get_device_properties(device)
    smem = int(getattr(props, "shared_memory_per_block_optin", 0) or getattr(props, "shared_memory_per_block", 0) or 0)
    # Conservative threshold for the short-seq full-tile path.
    return (2, 6, 12) if smem >= 120000 else (2, 4, 12)


def _tflops(bwin: int, heads: int, n: int, dh: int, ms: float) -> float:
    # Rough forward FLOPs: QK^T + PV ~= 4 * Bwin * H * N * N * Dh.
    flops = 4.0 * float(bwin) * float(heads) * float(n) * float(n) * float(dh)
    return flops / (ms * 1e-3) / 1e12


def _torch_window_attention_3d_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scale_qk: float,
) -> torch.Tensor:
    # q, k, v: (Bwin, H, N, Dh)
    attn = torch.matmul(q * scale_qk, k.transpose(-2, -1))
    if bias is not None:
        attn = attn + bias
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def main() -> None:
    p = argparse.ArgumentParser(description="Profile flash_window_attn3d with Aurora-like shapes.")
    p.add_argument("--preset", choices=("small", "aurora"), default="small")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--patch-res", type=int, nargs=3, default=(4, 32, 64), metavar=("C", "H", "W"))
    p.add_argument("--window-size", type=int, nargs=3, default=(2, 6, 12), metavar=("Wc", "Wh", "Ww"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--with-mask", action="store_true")
    p.add_argument(
        "--mode",
        choices=("auto", "nomask", "mask", "mixed"),
        default="auto",
        help="Profiling mode. auto follows --with-mask unless --mixed-mask-ratio is set.",
    )
    p.add_argument(
        "--mixed-mask-ratio",
        type=float,
        default=-1.0,
        help="If in (0,1], run mixed mask/no-mask schedule with this masked ratio (e.g. 0.5).",
    )
    p.add_argument("--report-out", type=str, default="", help="Write markdown report to this path.")
    p.add_argument(
        "--no-empty-cache-between-runs",
        action="store_true",
        help="Disable torch.cuda.empty_cache() before each benchmark run.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler.")

    if args.preset == "small":
        dim, heads = 256, 4
    else:
        dim, heads = 512, 8
    dh = dim // heads
    device = "cuda"
    ws = _choose_window_size(tuple(args.window_size), heads=heads, dim=dim, device=torch.device(device))
    n = ws[0] * ws[1] * ws[2]
    bwin = _infer_bwin(args.batch_size, tuple(args.patch_res), ws)
    dtype = torch.float32
    q = torch.randn(bwin, heads, n, dh, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    mask = None
    mask = torch.zeros(1, 1, n, n, device=device, dtype=dtype)
    mask[..., n // 2 :, : n // 2] = -1000.0

    out = torch.empty_like(q)
    scale_qk = float(1.0 / math.sqrt(dh))

    if args.mode == "mixed":
        run_mode = "mixed"
    elif args.mode == "mask":
        run_mode = "mask"
    elif args.mode == "nomask":
        run_mode = "nomask"
    else:
        if 0.0 < args.mixed_mask_ratio <= 1.0:
            run_mode = "mixed"
        else:
            run_mode = "mask" if args.with_mask else "nomask"

    if run_mode == "mixed":
        mask_ratio = args.mixed_mask_ratio if 0.0 < args.mixed_mask_ratio <= 1.0 else 0.5
    else:
        mask_ratio = 1.0 if run_mode == "mask" else 0.0

    print(
        f"[shape] preset={args.preset}, patch_res={tuple(args.patch_res)}, window={ws}, "
        f"Bwin={bwin}, H={heads}, N={n}, Dh={dh}, mode={run_mode}, mask_ratio={mask_ratio:.2f}"
    )

    def naive_nomask() -> torch.Tensor:
        return _torch_window_attention_3d_naive(q, k, v, None, scale_qk=scale_qk)

    def naive_mask() -> torch.Tensor:
        return _torch_window_attention_3d_naive(q, k, v, mask.expand(bwin, heads, n, n), scale_qk=scale_qk)

    def sdpa_nomask() -> torch.Tensor:
        return torch_window_attention_3d_reference(q, k, v, None)

    def sdpa_mask() -> torch.Tensor:
        return torch_window_attention_3d_reference(q, k, v, mask)

    def triton_nomask() -> torch.Tensor:
        return flash_window_attn_3d_forward(q, k, v, None)

    def triton_mask() -> torch.Tensor:
        return flash_window_attn_3d_forward(q, k, v, mask)

    if not args.no_empty_cache_between_runs:
        _clear_cuda_cache()
    if run_mode == "mixed":
        torch_naive_ms = _bench_ms_mixed(naive_nomask, naive_mask, mask_ratio, args.warmup, args.repeat)
    elif run_mode == "mask":
        torch_naive_ms = _bench_ms(naive_mask, args.warmup, args.repeat)
    else:
        torch_naive_ms = _bench_ms(naive_nomask, args.warmup, args.repeat)
    torch_naive_tflops = _tflops(bwin, heads, n, dh, torch_naive_ms)
    print(f"[baseline-naive] torch matmul+softmax: {torch_naive_ms:.4f} ms/iter, {torch_naive_tflops:.2f} TFLOPS")

    if not args.no_empty_cache_between_runs:
        _clear_cuda_cache()
    if run_mode == "mixed":
        torch_sdpa_ms = _bench_ms_mixed(sdpa_nomask, sdpa_mask, mask_ratio, args.warmup, args.repeat)
    elif run_mode == "mask":
        torch_sdpa_ms = _bench_ms(sdpa_mask, args.warmup, args.repeat)
    else:
        torch_sdpa_ms = _bench_ms(sdpa_nomask, args.warmup, args.repeat)
    torch_sdpa_tflops = _tflops(bwin, heads, n, dh, torch_sdpa_ms)
    print(f"[baseline-sdpa] torch_window_attention_3d_reference: {torch_sdpa_ms:.4f} ms/iter, {torch_sdpa_tflops:.2f} TFLOPS")

    if not args.no_empty_cache_between_runs:
        _clear_cuda_cache()
    if run_mode == "mixed":
        triton_auto_ms = _bench_ms_mixed(triton_nomask, triton_mask, mask_ratio, args.warmup, args.repeat)
    elif run_mode == "mask":
        triton_auto_ms = _bench_ms(triton_mask, args.warmup, args.repeat)
    else:
        triton_auto_ms = _bench_ms(triton_nomask, args.warmup, args.repeat)
    triton_auto_tflops = _tflops(bwin, heads, n, dh, triton_auto_ms)
    print(f"[triton-autotune] flash_window_attn_3d_forward: {triton_auto_ms:.4f} ms/iter, {triton_auto_tflops:.2f} TFLOPS")

    best_speedup_vs_naive = torch_naive_ms / triton_auto_ms
    best_speedup_vs_sdpa = torch_sdpa_ms / triton_auto_ms
    best_kind = "autotune"
    print(
        f"[best-vs-baselines] best={best_kind}, "
        f"speedup_vs_naive={best_speedup_vs_naive:.3f}x, speedup_vs_sdpa={best_speedup_vs_sdpa:.3f}x"
    )

    if args.report_out:
        lines = [
            "# flash_window_attn3d profiling",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Torch: {torch.__version__}",
            f"- Device: {torch.cuda.get_device_name(0)}",
            (
                f"- Shape: preset={args.preset}, patch_res={tuple(args.patch_res)}, "
                f"window={ws}, Bwin={bwin}, H={heads}, N={n}, Dh={dh}, mode={run_mode}, mask_ratio={mask_ratio:.2f}"
            ),
            f"- Warmup/Repeat: {args.warmup}/{args.repeat}",
            "",
            "## Baselines (Torch)",
            "",
            f"- `torch matmul+softmax (naive)`: {torch_naive_ms:.4f} ms/iter, {torch_naive_tflops:.2f} TFLOPS",
            f"- `torch_window_attention_3d_reference (sdpa/flash backend)`: {torch_sdpa_ms:.4f} ms/iter, {torch_sdpa_tflops:.2f} TFLOPS",
            "",
            "## Triton Autotune",
            "",
            f"- `flash_window_attn_3d_forward`: {triton_auto_ms:.4f} ms/iter, {triton_auto_tflops:.2f} TFLOPS",
            "",
            "## Triton Result",
            "",
            f"- `flash_window_attn_3d_forward (autotune)`: {triton_auto_ms:.4f} ms/iter, {triton_auto_tflops:.2f} TFLOPS",
        ]
        lines.extend(
            [
                "",
                "## Best vs Baselines",
                "",
                f"- Best path: `{best_kind}`",
                f"- Speedup vs naive: **{best_speedup_vs_naive:.3f}x**",
                f"- Speedup vs sdpa: **{best_speedup_vs_sdpa:.3f}x**",
                "",
            ]
        )
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[report] {out_path.resolve()}")


if __name__ == "__main__":
    main()

