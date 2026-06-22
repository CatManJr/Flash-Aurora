#!/usr/bin/env python3
"""Profile a single :class:`Swin3DTransformerBlock` to localize bottlenecks (SDPA vs GEMM vs
AdaLN vs layout) without encoder/decoder/time-MLP noise from the full backbone.

On CUDA, KPI buckets sum only ``aten::`` and memcpy rows so GEMM/FMHA **kernel** lines are not
double-counted with their parent ATen ops. SDPA focus uses ``_efficient_attention_forward`` etc.,
not only ``scaled_dot_product`` (PyTorch naming varies by backend).

Run from the repository root::

    uv run python aurora/profiling_swin3d_block.py
    uv run python aurora/profiling_swin3d_block.py --preset aurora --patch-h 180 --patch-w 360
    uv run python aurora/profiling_swin3d_block.py --shifted --report-out profiling/swin3d_block.md
    uv run python aurora/profiling_swin3d_block.py --use-triton-layout --use-triton-adaln
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import warnings
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse table helpers from the backbone profiler (same torch.profiler aggregation model).
from profiling_swin3d import (
    _extract_addmm_stats,
    _shorten,
    _top_ops_ms,
)
from flash_aurora.aurora.model.cuda_graph import CudaGraphSwin3DBlockRunner
from flash_aurora.aurora.model.inference_precision import apply_inference_config


_MATRIX_CONFIGS = [
    ("baseline_sdpa", []),
    ("triton_layout_sdpa", ["--use-triton-layout"]),
    ("triton_layout_cute", ["--use-triton-layout", "--use-cute-window-attn", "--autocast"]),
    (
        "triton_all_cute",
        [
            "--use-triton-layout",
            "--use-triton-adaln",
            "--use-triton-mlp",
            "--use-cute-window-attn",
            "--autocast",
        ],
    ),
]


def _timing_stats(samples: list[float]) -> dict[str, float]:
    samples = sorted(samples)
    trim = max(1, int(len(samples) * 0.05)) if len(samples) >= 20 else 0
    core = samples[trim : len(samples) - trim] if trim else samples
    mean = statistics.mean(core)
    return {
        "mean": mean,
        "median": samples[len(samples) // 2],
        "p5": samples[int(len(samples) * 0.05)],
        "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


def _cuda_stage_timer(fn) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    return result, start, end


def _measure_block_stages(
    block: Any,
    x: Any,
    c: Any,
    patch_res: tuple[int, int, int],
    *,
    rollout_step: int,
    warped: bool,
    autocast: bool,
    device: str,
) -> dict[str, float]:
    """Run one block forward manually and return CUDA-event stage timings."""
    import torch
    import torch.nn.functional as F

    from flash_aurora.aurora.model.swin3d import (
        compute_3d_shifted_window_mask,
        crop_3d,
        maybe_adjust_windows,
        pad_3d,
        window_partition_3d,
        window_reverse_3d,
    )

    events: list[tuple[str, Any, Any]] = []

    def timed(name: str, fn):
        result, start, end = _cuda_stage_timer(fn)
        events.append((name, start, end))
        return result

    ctx: Any
    if autocast and device.startswith("cuda"):
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        with torch.inference_mode():
            C, H, W = patch_res
            B, L, D = x.shape
            ws, ss = maybe_adjust_windows(block.window_size, block.shift_size, patch_res)
            shortcut = x
            x_5d = x.view(B, C, H, W, D)
            if not all(s == 0 for s in ss):
                attn_mask, _ = timed(
                    "shifted_mask",
                    lambda: compute_3d_shifted_window_mask(
                        C, H, W, ws, ss, x.device, x.dtype, warped=warped
                    ),
                )
            else:
                attn_mask = None

            if block.use_triton_layout:
                from flash_aurora.aurora.ops.triton_swin3d_layout import (
                    crop_roll_unmerge_windows_triton,
                    roll_pad_partition_windows_triton,
                )

                x_windows = timed(
                    "layout_partition",
                    lambda: roll_pad_partition_windows_triton(
                        x_5d, patch_res, block.window_size, block.shift_size, pool=block._layout_pool
                    ),
                )
            else:
                def _torch_partition():
                    shifted_x = (
                        torch.roll(x_5d, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
                        if not all(s == 0 for s in ss)
                        else x_5d
                    )
                    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
                    shifted_x = pad_3d(shifted_x, pad_size)
                    windows = window_partition_3d(shifted_x, ws)
                    return windows.view(-1, ws[0] * ws[1] * ws[2], D), shifted_x.shape, pad_size

                x_windows, shifted_shape, pad_size = timed("layout_partition", _torch_partition)

            attn = block.attn
            qkv = timed(
                "qkv_linear",
                lambda: attn._linear_with_optional_lora_merge(
                    x_windows, attn.qkv, attn.lora_qkv, step=rollout_step, cache_name="qkv"
                )
                if hasattr(attn, "_linear_with_optional_lora_merge")
                else attn.qkv(x_windows),
            )

            Bwin, N, _ = qkv.shape
            head_dim = D // attn.num_heads

            attn_dropout = attn.attn_drop if attn.training else 0.0
            use_cute = (
                attn.use_cute_window_attn
                and (not attn.training)
                and attn_dropout == 0.0
                and qkv.is_cuda
                and qkv.dtype in (torch.float32, torch.bfloat16)
                and (not torch.is_grad_enabled())
            )
            use_cute_qkvpacked = (
                use_cute
                and qkv.is_contiguous()
                and qkv.shape[-1] == 3 * attn.num_heads * head_dim
                and qkv.dtype in (torch.bfloat16, torch.float32)
            )
            if not use_cute_qkvpacked:
                def _qkv_layout():
                    qkv_view = qkv.view(Bwin, N, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv_view[0], qkv_view[1], qkv_view[2]
                    return q, k, v

                q, k, v = timed("qkv_rearrange_split", _qkv_layout)
                if use_cute:
                    q, k, v = timed(
                        "qkv_contiguous",
                        lambda: (
                            q.contiguous() if not q.is_contiguous() else q,
                            k.contiguous() if not k.is_contiguous() else k,
                            v.contiguous() if not v.is_contiguous() else v,
                        ),
                    )

            def _attention():
                if use_cute_qkvpacked:
                    from flash_aurora.aurora.ops.cute import window_attn_fwd_cute_qkvpacked

                    bias = None if attn_mask is None else attn_mask.to(dtype=torch.float32, device=qkv.device)
                    if bias is not None and not bias.is_contiguous():
                        bias = bias.contiguous()
                    return window_attn_fwd_cute_qkvpacked(
                        qkv, attn.num_heads, bias=bias, output_layout="bnc"
                    )
                if use_cute:
                    from flash_aurora.aurora.ops.cute import WinAttnPrecision, window_attn_fwd_cute

                    precision = (
                        WinAttnPrecision.BF16_MIXED
                        if q.dtype == torch.bfloat16
                        else WinAttnPrecision.TF32_ACC_FP32
                    )
                    bias = None if attn_mask is None else attn_mask.to(dtype=torch.float32, device=q.device)
                    if bias is not None and not bias.is_contiguous():
                        bias = bias.contiguous()
                    return window_attn_fwd_cute(q, k, v, bias=bias, precision=precision)
                if attn_mask is not None:
                    mask = attn_mask.unsqueeze(1).unsqueeze(0)
                    batch = q.shape[0] // mask.shape[1]
                    mask = mask.repeat(batch, 1, 1, 1, 1).reshape(-1, *mask.shape[2:])
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=attn_dropout)
                return F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout)

            attn_out = timed("attention_core", _attention)
            if use_cute_qkvpacked:
                x_attn = attn_out
            else:
                x_attn = timed(
                    "attn_output_layout",
                    lambda: attn_out.permute(0, 2, 1, 3).reshape(Bwin, N, D),
                )
            x_attn = timed(
                "proj_linear",
                lambda: attn._linear_with_optional_lora_merge(
                    x_attn, attn.proj, attn.lora_proj, step=rollout_step, cache_name="proj"
                )
                if hasattr(attn, "_linear_with_optional_lora_merge")
                else attn.proj(x_attn),
            )
            x_attn = attn.proj_drop(x_attn)

            if block.use_triton_layout:
                x_after_layout = timed(
                    "layout_unmerge",
                    lambda: crop_roll_unmerge_windows_triton(
                        x_attn, patch_res, block.window_size, block.shift_size, pool=block._layout_pool
                    ),
                )
            else:
                def _torch_unmerge():
                    attn_windows = x_attn.view(-1, ws[0], ws[1], ws[2], D)
                    _, pad_C, pad_H, pad_W, _ = shifted_shape
                    shifted_x = window_reverse_3d(attn_windows, ws, pad_C, pad_H, pad_W)
                    shifted_x = crop_3d(shifted_x, pad_size)
                    if not all(s == 0 for s in ss):
                        return torch.roll(shifted_x, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
                    return shifted_x

                x_after_layout = timed("layout_unmerge", _torch_unmerge)
            x_attn_flat = timed("layout_flatten", lambda: x_after_layout.reshape(B, C * H * W, D))

            use_d2_norm1 = (
                block.norm1.use_triton
                and not block.training
                and shortcut.is_cuda
                and shortcut.dtype in (torch.float32, torch.bfloat16)
                and isinstance(block.drop_path, torch.nn.Identity)
            )
            x_normed = timed(
                "residual_adaln1",
                lambda: block.norm1.forward_add_residual(shortcut, x_attn_flat, c)
                if use_d2_norm1
                else shortcut + block.drop_path(block.norm1(x_attn_flat, c)),
            )

            mlp = block.mlp
            h = timed("mlp_fc1", lambda: mlp.fc1(x_normed))
            h = timed(
                "mlp_gelu",
                lambda: __import__("aurora.ops.triton_gelu", fromlist=["gelu_forward_triton"]).gelu_forward_triton(h)
                if (
                    mlp.use_triton_gelu
                    and (not mlp.training)
                    and mlp.drop.p == 0.0
                    and h.is_cuda
                    and h.dtype in (torch.float32, torch.bfloat16)
                )
                else mlp.act(h),
            )
            h = mlp.drop(h)
            h = timed("mlp_fc2", lambda: mlp.fc2(h))
            h = mlp.drop(h)

            use_d2_norm2 = (
                block.norm2.use_triton
                and not block.training
                and x_normed.is_cuda
                and x_normed.dtype in (torch.float32, torch.bfloat16)
                and isinstance(block.drop_path, torch.nn.Identity)
            )
            _ = timed(
                "residual_adaln2",
                lambda: block.norm2.forward_add_residual(x_normed, h, c)
                if use_d2_norm2
                else x_normed + block.drop_path(block.norm2(h, c)),
            )

    torch.cuda.synchronize()
    return {name: start.elapsed_time(end) for name, start, end in events}


def _measure_stage_stats(
    block: Any,
    x: Any,
    c: Any,
    patch_res: tuple[int, int, int],
    *,
    rollout_step: int,
    warped: bool,
    autocast: bool,
    device: str,
    warmup: int,
    repeat: int,
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        _measure_block_stages(
            block, x, c, patch_res, rollout_step=rollout_step,
            warped=warped, autocast=autocast, device=device,
        )
    buckets: dict[str, list[float]] = {}
    for _ in range(repeat):
        row = _measure_block_stages(
            block, x, c, patch_res, rollout_step=rollout_step,
            warped=warped, autocast=autocast, device=device,
        )
        for name, ms in row.items():
            buckets.setdefault(name, []).append(ms)
    return {name: _timing_stats(values) for name, values in buckets.items()}


def _print_stage_stats(stage_stats: dict[str, dict[str, float]]) -> list[str]:
    total = sum(v["mean"] for v in stage_stats.values())
    lines = ["", "--- Explicit CUDA-event stage timings ---"]
    lines.append(f"{'stage':<24}{'mean ms':>10}{'%':>8}{'median':>10}{'p5':>10}{'p95':>10}")
    lines.append("-" * 72)
    for name, stats in sorted(stage_stats.items(), key=lambda item: -item[1]["mean"]):
        pct = 100.0 * stats["mean"] / total if total > 0 else 0.0
        lines.append(
            f"{name:<24}{stats['mean']:>10.4f}{pct:>7.1f}%"
            f"{stats['median']:>10.4f}{stats['p5']:>10.4f}{stats['p95']:>10.4f}"
        )
    lines.append("-" * 72)
    lines.append(f"{'stage sum':<24}{total:>10.4f}")
    return lines


def _bucket_swin3d_block(name: str) -> str:
    """Coarse buckets tuned for one Swin3D block (W-MSA / SW-MSA + AdaLN + MLP)."""
    n = name.lower()
    if "memcpy" in n or "dtoh" in n or "htod" in n or "memset" in n:
        return "memcpy"
    if (
        "scaled_dot_product" in n
        or "sdpa" in n
        or "efficient_attention" in n
        or "flash" in n
        or "fmha" in n
    ):
        return "attention (SDPA / FMHA)"
    if (
        "addmm" in n
        or "aten::mm" in n
        or "magma" in n
        or "cutlass" in n
        or n.endswith("::mm")
        or "cublas" in n
        or "gemm" in n
    ):
        return "GEMM (Linear / matmul)"
    if "native_layer_norm" in n or "layer_norm" in n:
        return "LayerNorm"
    if "gelu" in n:
        return "GELU"
    if "silu" in n:
        return "SiLU (AdaLN modulation MLP)"
    if "aten::copy_" in n or "copy_kernel" in n or "direct_copy" in n:
        return "copy / scatter"
    if "roll" in n:
        return "roll / pad / window layout"
    if "triton" in n or "compiledfxgraph" in n or ("compiled" in n and "fx" in n):
        return "triton / torch.compile"
    if (
        "elementwise" in n
        or "::mul" in n
        or "::add" in n
        or "::div" in n
        or "where" in n
        or "masked_fill" in n
    ):
        return "elementwise (other)"
    return "other"


def _include_row_in_cuda_bucket_sum(key: str) -> bool:
    """Avoid double-counting: PyTorch lists both ``aten::addmm`` and ``magma_*`` / ``fmha_*`` with
    nearly identical self-CUDA; summing all rows inflates totals (~2x). Buckets use ATen + memcpy
    rows only."""
    k = str(key)
    if k.startswith("aten::"):
        return True
    if "memcpy" in k.lower():
        return True
    return False


def _aggregate_block_kpis(prof: Any, *, use_cuda: bool) -> tuple[dict[str, float], float]:
    buckets: dict[str, float] = {}
    total_ms = 0.0
    for e in prof.key_averages():
        key = str(e.key)
        if use_cuda and not _include_row_in_cuda_bucket_sum(key):
            continue
        if use_cuda:
            t_us = float(
                getattr(e, "self_cuda_time_total", 0)
                or getattr(e, "self_device_time_total", 0)
                or 0
            )
        else:
            t_us = float(getattr(e, "self_cpu_time_total", 0) or 0)
        if t_us <= 0:
            continue
        t_ms = t_us / 1000.0
        total_ms += t_ms
        b = _bucket_swin3d_block(key)
        buckets[b] = buckets.get(b, 0.0) + t_ms
    return buckets, total_ms


def _extract_sdpa_stats(prof: Any, *, use_cuda: bool) -> tuple[int, float]:
    """ATen-level SDPA / memory-efficient attention (matches table rows like ``_efficient_attention_forward``)."""
    calls = 0
    self_ms = 0.0
    for e in prof.key_averages():
        key = str(e.key)
        kl = key.lower()
        if not key.startswith("aten::"):
            continue
        if (
            "scaled_dot_product" not in kl
            and "sdpa" not in kl
            and "efficient_attention" not in kl
        ):
            continue
        calls += int(getattr(e, "count", 0) or 0)
        if use_cuda:
            t_us = float(
                getattr(e, "self_cuda_time_total", 0)
                or getattr(e, "self_device_time_total", 0)
                or 0
            )
        else:
            t_us = float(getattr(e, "self_cpu_time_total", 0) or 0)
        self_ms += t_us / 1000.0
    return calls, self_ms


def _print_bottleneck_summary(
    buckets: dict[str, float], total_ms: float, *, use_cuda: bool
) -> list[str]:
    lines = [
        "",
        "--- Swin3D block bottleneck (CUDA: ATen+memcpy rows only — no GEMM/FMHA kernel double-count) ---"
        if use_cuda
        else "--- Swin3D block bottleneck (aggregate self-time, all profiler rows) ---",
    ]
    if total_ms <= 0:
        lines.append("  (no CUDA/CPU self-time recorded)")
        return lines
    for b in sorted(buckets.keys(), key=lambda k: -buckets[k]):
        ms = buckets[b]
        lines.append(f"  {b}: {ms:.2f} ms ({100.0 * ms / total_ms:.1f}%)")
    lines.append(f"  total: {total_ms:.2f} ms")
    return lines


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from flash_aurora.aurora.model.swin3d import Swin3DTransformerBlock

    p = argparse.ArgumentParser(
        description="Profile one Swin3DTransformerBlock (isolate block-level bottlenecks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=("small", "aurora", "none"),
        default="small",
        help=(
            "small: dim=256, heads=4 (matches AuroraSmall-like stage 0); "
            "aurora: dim=512, heads=8 (full Aurora 1.3B stage 0). "
            "'none' uses --dim / --num-heads explicitly."
        ),
    )
    p.add_argument("--dim", type=int, default=256, help="Block channel dim (ignored if preset≠none).")
    p.add_argument("--num-heads", type=int, default=4, help="Attention heads (ignored if preset≠none).")
    p.add_argument(
        "--time-dim",
        type=int,
        default=0,
        help="AdaLN context dim (default: same as dim).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--latent-levels", type=int, default=4, help="C in patch_res (C, H, W).")
    p.add_argument("--patch-h", type=int, default=32)
    p.add_argument("--patch-w", type=int, default=64)
    p.add_argument(
        "--window-size",
        type=int,
        nargs=3,
        default=(2, 6, 12),
        metavar=("Wc", "Wh", "Ww"),
        help="3D window size (default 2 6 12, Aurora).",
    )
    p.add_argument(
        "--shifted",
        action="store_true",
        help="Use shifted-window (SW-MSA) attention mask path; default is W-MSA (shift 0).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--table-rows", type=int, default=40)
    p.add_argument("--plot-out", type=str, default="")
    p.add_argument("--plot-top", type=int, default=30)
    p.add_argument("--report-out", type=str, default="")
    p.add_argument("--trace-out", type=str, default="", help="Export torch profiler Chrome trace JSON.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--autocast",
        action="store_true",
        help="Run block under BF16 autocast (not default FP32).",
    )
    p.add_argument(
        "--input-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Synthetic x/c input dtype. Use bfloat16 for Triton AdaLN + CuTe BF16 profiling.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile this block only (fixed shape recommended).",
    )
    p.add_argument(
        "--inference-precision",
        choices=("fp32", "pytorch_autocast", "fast_fp32", "tf32", "bf16_mixed", "bf16"),
        default=None,
        help=(
            "Apply one of the five inference presets (overrides scattered flags): "
            "fp32, pytorch_autocast, fast_fp32, tf32, bf16_mixed, bf16."
        ),
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture and replay the fixed-shape block with torch.cuda.CUDAGraph.",
    )
    p.add_argument(
        "--use-triton-layout",
        action="store_true",
        help="Fused roll/pad/window Triton path (CUDA float32).",
    )
    p.add_argument(
        "--use-triton-adaln",
        action="store_true",
        help="Fused AdaLN Triton path (CUDA float32).",
    )
    p.add_argument(
        "--use-triton-mlp",
        action="store_true",
        help="Triton GELU in MLP (CUDA float32, eval).",
    )
    p.add_argument(
        "--use-cute-window-attn",
        action="store_true",
        help="Use CuTe window attention inside the block (requires CUDA inference path).",
    )
    p.add_argument(
        "--stage-timing",
        action="store_true",
        help="Print explicit CUDA-event timings for block stages in addition to torch profiler.",
    )
    p.add_argument(
        "--stage-only",
        action="store_true",
        help="Only run full-block and explicit stage timings; skip torch.profiler table.",
    )
    p.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA in attention (extra GEMMs vs baseline).",
    )
    p.add_argument(
        "--use-lora-merged",
        action="store_true",
        help="Merge LoRA into linear at inference (requires --use-lora).",
    )
    args = p.parse_args()

    if args.inference_precision is not None:
        preset = apply_inference_config(args.inference_precision)
        args.use_triton_layout = preset["use_triton_layout"]
        args.use_triton_adaln = preset["use_triton_adaln"]
        args.use_triton_mlp = preset["use_triton_mlp"]
        args.use_cute_window_attn = preset["use_cute_window_attn"]
        args.input_dtype = (
            "bfloat16" if preset["backbone_compute_dtype"] == "bfloat16" else "float32"
        )
        args.autocast = preset["autocast"]

    if args.cuda_graph and args.compile:
        raise SystemExit("--cuda-graph and --compile are mutually exclusive in this profiler.")
    if args.cuda_graph and args.stage_timing:
        raise SystemExit("--cuda-graph cannot be combined with --stage-timing.")
    if args.cuda_graph and not args.device.startswith("cuda"):
        raise SystemExit("--cuda-graph requires a CUDA device.")

    if args.preset == "small":
        dim, num_heads = 256, 4
    elif args.preset == "aurora":
        dim, num_heads = 512, 8
    else:
        dim, num_heads = args.dim, args.num_heads

    time_dim = args.time_dim if args.time_dim > 0 else dim
    ws = tuple(args.window_size)

    C, H, W = args.latent_levels, args.patch_h, args.patch_w
    if C % ws[0] != 0:
        raise SystemExit(f"latent-levels ({C}) must be divisible by window[0] ({ws[0]}).")
    L = C * H * W

    shift = (ws[0] // 2, ws[1] // 2, ws[2] // 2) if args.shifted else (0, 0, 0)

    block = Swin3DTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        time_dim=time_dim,
        window_size=ws,
        shift_size=shift,
        mlp_ratio=4.0,
        drop_path=0.0,
        use_triton_layout=args.use_triton_layout,
        use_triton_adaln=args.use_triton_adaln,
        use_triton_mlp=args.use_triton_mlp,
        use_lora=args.use_lora,
        use_lora_merged_inference=args.use_lora_merged,
        use_cute_window_attn=args.use_cute_window_attn,
    )
    block.eval()
    input_dtype = torch.bfloat16 if args.input_dtype == "bfloat16" else torch.float32
    block.to(device=args.device, dtype=input_dtype)

    if args.compile:
        block = torch.compile(block, dynamic=False)

    x = torch.randn(args.batch_size, L, dim, device=args.device, dtype=input_dtype)
    c = torch.randn(args.batch_size, time_dim, device=args.device, dtype=input_dtype)
    patch_res = (C, H, W)

    graph_runner: CudaGraphSwin3DBlockRunner | None = None

    def run_once_eager() -> None:
        ctx: Any
        if args.autocast and args.device.startswith("cuda"):
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif args.autocast and torch.xpu.is_available() and args.device == "xpu":
            ctx = torch.autocast(device_type="xpu", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            with torch.inference_mode():
                _ = block(x, c, patch_res, rollout_step=0, warped=True)

    def run_once() -> None:
        if graph_runner is not None:
            _ = graph_runner(x, c)
        else:
            run_once_eager()

    print(
        f"[config] preset={args.preset}, dim={dim}, heads={num_heads}, time_dim={time_dim}, "
        f"patch_res={patch_res}, L={L}, window={ws}, shift={shift}, "
        f"input_dtype={args.input_dtype}, autocast={args.autocast}, compile={args.compile}, "
        f"inference_precision={args.inference_precision}, "
        f"cuda_graph={args.cuda_graph}, "
        f"use_triton_layout={args.use_triton_layout}, use_triton_adaln={args.use_triton_adaln}, "
        f"use_triton_mlp={args.use_triton_mlp}, use_cute_window_attn={args.use_cute_window_attn}, "
        f"use_lora={args.use_lora}, use_lora_merged={args.use_lora_merged}"
    )

    if not args.device.startswith("cuda") and torch.cuda.is_available():
        warnings.warn(f"CUDA is available but device={args.device!r}; profiling may be CPU-only.")

    for _ in range(args.warmup):
        run_once_eager()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    if args.cuda_graph:
        graph_runner = CudaGraphSwin3DBlockRunner(
            block,
            x,
            c,
            patch_res,
            rollout_step=0,
            warped=True,
            autocast=args.autocast,
        )
        print("[cuda-graph] captured fixed-shape Swin3D block")

    timing_line = ""
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(args.repeat):
            run_once()
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1)
        timing_line = f"GPU: {ms:.2f} ms for {args.repeat} iters -> {ms / args.repeat:.3f} ms/iter"
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"[timing] {timing_line}")
        print(
            f"[mem] peak CUDA allocated: {peak_alloc:.1f} MB, "
            f"peak reserved: {peak_reserved:.1f} MB"
        )
    else:
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            run_once()
        ms = (time.perf_counter() - t0) * 1e3
        timing_line = f"CPU: {ms:.2f} ms for {args.repeat} iters"
        print(f"[timing] {timing_line}")

    stage_stats: dict[str, dict[str, float]] = {}
    if args.stage_timing:
        if not args.device.startswith("cuda"):
            warnings.warn("--stage-timing requires CUDA; skipping explicit stage timings.")
        else:
            stage_stats = _measure_stage_stats(
                block,
                x,
                c,
                patch_res,
                rollout_step=0,
                warped=True,
                autocast=args.autocast,
                device=args.device,
                warmup=max(1, args.warmup),
                repeat=max(1, args.repeat),
            )
            for line in _print_stage_stats(stage_stats):
                print(line)

    if args.stage_only:
        return

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        acc_events=True,
    ) as prof:
        for _ in range(args.repeat):
            run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    sort_by = "self_cuda_time_total" if args.device.startswith("cuda") else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=args.table_rows)
    print("\n" + table)

    if args.trace_out:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print(f"[trace] {trace_path.resolve()}")

    use_cuda = args.device.startswith("cuda")
    names, tms = _top_ops_ms(prof, use_cuda=use_cuda, top_k=args.plot_top)
    buckets, total_kpi_ms = _aggregate_block_kpis(prof, use_cuda=use_cuda)
    for line in _print_bottleneck_summary(buckets, total_kpi_ms, use_cuda=use_cuda):
        print(line)

    addmm_calls, addmm_self_ms = _extract_addmm_stats(prof, use_cuda=use_cuda)
    sdpa_calls, sdpa_self_ms = _extract_sdpa_stats(prof, use_cuda=use_cuda)
    st = "self_cuda" if use_cuda else "self_cpu"
    print(
        f"\n[focus] ATen SDPA / efficient_attention calls={sdpa_calls}, {st}≈{sdpa_self_ms:.3f} ms"
    )
    print(f"[focus] aten::addmm calls={addmm_calls}, {st}≈{addmm_self_ms:.3f} ms")

    if args.plot_out:
        timing_safe = timing_line.replace("\n", " ")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(names)
        fig_h = max(4.0, 0.38 * n + 1.2)
        fig, ax = plt.subplots(figsize=(10.5, fig_h), layout="constrained")
        y = range(n)
        ax.barh(y, tms, color="#8d4c2c", alpha=0.9)
        ax.set_yticks(list(y))
        ax.set_yticklabels([_shorten(x) for x in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Self time (ms)")
        ax.set_title("Swin3DTransformerBlock — top ops (self CUDA)")
        fig.text(0.02, 0.02, timing_safe, fontsize=8, family="monospace", color="#333333")
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {Path(args.plot_out).resolve()}")

    if args.report_out:
        lines = [
            "# Swin3DTransformerBlock profiling",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Torch: {torch.__version__}",
            f"- Config: preset={args.preset}, dim={dim}, heads={num_heads}, patch_res={patch_res}, "
            f"window={ws}, shift={shift}, input_dtype={args.input_dtype}, "
            f"autocast={args.autocast}, compile={args.compile}, cuda_graph={args.cuda_graph}, "
            f"Triton layout/AdaLN/MLP={args.use_triton_layout}/{args.use_triton_adaln}/{args.use_triton_mlp}, "
            f"CuTe attention={args.use_cute_window_attn}",
            "",
            "## Timer",
            "",
            timing_line,
            "",
            "## Bottleneck buckets (block-local)",
            "",
            "| Bucket | Self (ms) | % |",
            "| --- | ---: | ---: |",
        ]
        if total_kpi_ms > 0:
            for b in sorted(buckets.keys(), key=lambda k: -buckets[k]):
                ms = buckets[b]
                pct = 100.0 * ms / total_kpi_ms
                lines.append(f"| {b} | {ms:.3f} | {pct:.1f} |")
        if stage_stats:
            stage_total = sum(v["mean"] for v in stage_stats.values())
            lines.extend(
                [
                    "",
                    "## Explicit Stage Timings",
                    "",
                    "| Stage | Mean (ms) | % of stage sum | Median | p5 | p95 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, stats in sorted(stage_stats.items(), key=lambda item: -item[1]["mean"]):
                pct = 100.0 * stats["mean"] / stage_total if stage_total > 0 else 0.0
                lines.append(
                    f"| {name} | {stats['mean']:.4f} | {pct:.1f} | "
                    f"{stats['median']:.4f} | {stats['p5']:.4f} | {stats['p95']:.4f} |"
                )
        lines.extend(
            [
                "",
                "## Focus",
                "",
                f"- ATen `scaled_dot_product*` / `_efficient_attention_forward`: calls≈{sdpa_calls}, "
                f"self-time≈{sdpa_self_ms:.3f} ms",
                f"- `aten::addmm`: calls={addmm_calls}, self-time≈{addmm_self_ms:.3f} ms",
                "",
                "## Top operators",
                "",
                "| Rank | Operator | Self (ms) |",
                "| ---: | --- | ---: |",
            ]
        )
        for i, (n, t) in enumerate(zip(names, tms, strict=True), start=1):
            lines.append(f"| {i} | {n.replace('|', '\\|')} | {t:.3f} |")
        lines.extend(["", "## Full profiler table", "", "```text", table.rstrip(), "```", ""])
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] {Path(args.report_out).resolve()}")


if __name__ == "__main__":
    main()
