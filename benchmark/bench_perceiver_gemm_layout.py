#!/usr/bin/env python3
"""Catalog and microbench **level_decoder** GEMM layouts (512↔1024 MLP + QKV).

Use after ``bench_small_pretrained.py`` confirms E2E/accuracy; this script isolates
:class:`~aurora.model.perceiver.PerceiverResampler` tensor geometry and cuBLAS/CUTLASS
problem sizes for layout work (flat M, weight TN, fusion candidates).

Examples::

    CUTE_DSL_ARCH=sm_120a AURORA_HF_LOCAL_DIR=/root/autodl-tmp/aurora \\
      uv run python benchmark/bench_perceiver_gemm_layout.py

    uv run python benchmark/bench_perceiver_gemm_layout.py --micro-iters 100
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_AURORA_PKG = _REPO / "aurora"
if _AURORA_PKG.is_dir():
    sys.path.insert(0, str(_AURORA_PKG))


@dataclass(frozen=True)
class GemmSpec:
    name: str
    m: int
    k: int
    n: int
    notes: str = ""


def _bench_ms(fn: Callable[[], None], warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0 / iters


def _linear_gemm_specs(
    name: str,
    in_features: int,
    out_features: int,
    rows: int,
    *,
    notes: str = "",
) -> GemmSpec:
    """PyTorch ``F.linear(x, w)`` → GEMM with M=rows, K=in, N=out (TN on weight)."""
    return GemmSpec(name, rows, in_features, out_features, notes)


def _capture_decoder_geometry(model, batch) -> dict:
    info: dict = {}
    dec = model.decoder

    orig = dec.deaggregate_levels

    def wrapped(le, x, ld):
        info["level_embed"] = tuple(le.shape)
        info["x_ctx"] = tuple(x.shape)
        info["BL"] = le.shape[0] * le.shape[1]
        info["L1"] = le.shape[2]
        info["L2"] = x.shape[2]
        info["D"] = le.shape[3]
        return orig(le, x, ld)

    dec.deaggregate_levels = wrapped  # type: ignore[method-assign]

    with torch.inference_mode():
        model.forward(batch)
    return info


def _build_specs(info: dict, attn, mlp) -> list[GemmSpec]:
    bl = info["BL"]
    l1, l2, d = info["L1"], info["L2"], info["D"]
    mlp_h = mlp.net[0].out_features
    inner = attn.inner_dim
    specs: list[GemmSpec] = [
        _linear_gemm_specs("attn.to_q", d, inner, bl * l1, notes=f"latent (BL,L1,D) → M=BL*L1={bl*l1}"),
        _linear_gemm_specs("attn.to_kv", d, 2 * inner, bl * l2, notes=f"context (BL,L2,D) → M=BL*L2={bl*l2}"),
        _linear_gemm_specs("attn.to_out", inner, d, bl * l1),
        _linear_gemm_specs("mlp.fc1", d, mlp_h, bl * l1, notes="512→1024 expansion"),
        _linear_gemm_specs("mlp.fc2", mlp_h, d, bl * l1, notes="1024→512 contraction"),
    ]
    h, dh = attn.num_heads, attn.head_dim
    _ = (h, dh, l1, l2)  # FMHA: B=BL, seqlen L1/L2 — not a GEMM row; tracked separately
    return specs


def _print_layout_table(latents: torch.Tensor, ctx: torch.Tensor) -> None:
    print("\n=== Tensor layout (level_decoder inputs) ===")
    for name, t in ("latents", latents), ("context", ctx):
        print(
            f"  {name:8s} shape={tuple(t.shape)} stride={t.stride()} "
            f"contig={t.is_contiguous()} dtype={t.dtype}"
        )
    flat = latents.reshape(-1, latents.shape[-1])
    print(
        f"  latents→2D shape={tuple(flat.shape)} stride={flat.stride()} "
        f"(view, no copy when contiguous)"
    )


def _print_gemm_catalog(specs: list[GemmSpec], attn, info: dict) -> None:
    bl, l1, l2 = info["BL"], info["L1"], info["L2"]
    print("\n=== GEMM catalog (bf16_mixed E/D: activations FP32, TN weight) ===")
    print(f"  batch_eff BL={bl}, L1={l1}, L2={l2}, D={info['D']}, heads={attn.num_heads}")
    print(f"  {'name':16s} {'M':>8s} {'K':>6s} {'N':>6s}  notes")
    print("  " + "-" * 62)
    for s in specs:
        print(f"  {s.name:16s} {s.m:8d} {s.k:6d} {s.n:6d}  {s.notes}")
    fmha_mac = bl * attn.num_heads * l1 * l2 * attn.head_dim
    print(
        f"\n  FMHA (SDPA, not GEMM): B={bl}, H={attn.num_heads}, Lq={l1}, Lk={l2}, "
        f"dh={attn.head_dim}  ~{2*fmha_mac/1e6:.1f} MFLOP/layer"
    )


def _run_microbenches(
    latents: torch.Tensor,
    ctx: torch.Tensor,
    attn,
    mlp,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> None:
    print("\n=== Microbench (isolated F.linear / matmul, FP32) ===")
    lat2 = latents.reshape(-1, latents.shape[-1])
    ctx2 = ctx.reshape(-1, ctx.shape[-1])

    w_q = attn.to_q.weight
    w_kv = attn.to_kv.weight
    w_o = attn.to_out.weight
    w1, w2 = mlp.net[0].weight, mlp.net[2].weight
    with torch.inference_mode():
        hidden = F.gelu(F.linear(lat2, w1), approximate="none")

    cases: list[tuple[str, Callable[[], None]]] = [
        ("to_q  nested", lambda: F.linear(latents, w_q)),
        ("to_q  flat", lambda: F.linear(lat2, w_q)),
        ("to_kv nested", lambda: F.linear(ctx, w_kv)),
        ("to_kv flat", lambda: F.linear(ctx2, w_kv)),
        ("to_out", lambda: F.linear(lat2, w_o)),  # after attn — use flat
        ("fc1", lambda: F.linear(lat2, w1)),
        ("fc2 only", lambda: F.linear(hidden, w2)),
        ("mlp full", lambda: mlp.net(lat2)),
    ]

    for name, fn in cases:
        ms = _bench_ms(fn, warmup, iters, device)
        print(f"  {name:18s} {ms:7.3f} ms")


def main() -> None:
    from bench_small_pretrained import _build_model, _load_batch

    p = argparse.ArgumentParser(description="level_decoder GEMM layout catalog + microbench")
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("AURORA_HF_LOCAL_DIR", "/root/autodl-tmp/aurora")))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--precision", default="bf16_mixed")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--micro-iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    ckpt = args.checkpoint or (args.data_dir / "aurora-0.25-small-pretrained.ckpt")
    batch = _load_batch(args.data_dir).to(device)

    gc.collect()
    torch.cuda.empty_cache()
    model = _build_model(args.precision, ckpt, device)
    model.eval()

    info = _capture_decoder_geometry(model, batch)
    dec = model.decoder
    attn, mlp, _, _ = dec.level_decoder.layers[0]
    specs = _build_specs(info, attn, mlp)

    B, L, l1, D = info["level_embed"]
    _, _, l2, _ = info["x_ctx"]
    latents = torch.randn(B, L, l1, D, device=device, dtype=torch.float32)
    ctx = torch.randn(B, L, l2, D, device=device, dtype=torch.float32)

    print(f"checkpoint={ckpt}")
    print(f"precision={args.precision}  device={torch.cuda.get_device_name(0)}")
    _print_layout_table(latents, ctx)
    _print_gemm_catalog(specs, attn, info)

    def full_decoder():
        with torch.inference_mode():
            dec.deaggregate_levels(
                latents,
                ctx,
                dec.level_decoder,
            )

    ms_full = _bench_ms(full_decoder, args.warmup, args.micro_iters, device)
    print(f"\n=== Full level_decoder (deaggregate_levels) ===")
    print(f"  {ms_full:.3f} ms/iter")

    _run_microbenches(
        latents, ctx, attn, mlp, warmup=args.warmup, iters=args.micro_iters, device=device
    )

    print("\n=== MLP roadmap (two tracks only; no Triton MLP) ===")
    print("  A. fc2 fast: M=140k K=1024 N=512 — prepack TN, decoder BF16 TC, CuTe/cuDNN GEMM")
    print("  B. fc1+GELU+fc2 fused graph — CuTeDSL or cuDNN frontend (gemm_* + epilogue)")
    print("     https://github.com/NVIDIA/cudnn-frontend/tree/develop/python/cudnn")
    print("  Skip: fc1+GELU-only fusion (low ROI vs fc2); standalone Triton MLP removed")


if __name__ == "__main__":
    main()
