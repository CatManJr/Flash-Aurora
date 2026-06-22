#!/usr/bin/env python3
"""Benchmark Aurora **Perceiver** stacks via PyTorch SDPA.

The encoder / decoder paths exercised here:

1. **Encoder** ``level_agg`` - the sole :class:`~aurora.model.perceiver.PerceiverResampler` inside
   ``aggregate_levels``.
2. **Decoder** ``level_decoder`` - primary ``deaggregate_levels`` path for pressure-level fusion.
3. **Decoder** ``level_decoder_alternate`` - optional second stack when ``separate_perceiver`` is
   non-empty (enable with ``--decoder-alternate``).
4. **Full Perceiver strip** - ``aggregate_levels``, then ``deaggregate_levels(main)``, then (if
   enabled) ``deaggregate_levels(alternate)``, matching the chained calls in a realistic forward.

The TFLOPS column uses a dense cross-attention forward estimate:
``4 * B_eff * num_heads * L_q * L_k * head_dim x depth`` (MLP/LayerNorm excluded).

Examples::

    PYTHONPATH=aurora uv run python benchmark/bench_perceiver_tflops.py
    PYTHONPATH=aurora uv run python benchmark/bench_perceiver_tflops.py --decoder-alternate --warmup 10
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

_REPO = Path(__file__).resolve().parents[1]


def estimate_cross_attn_fwd_flops(
    b_eff: int,
    num_heads: int,
    head_dim: int,
    seq_q: int,
    seq_kv: int,
    depth: int,
) -> float:
    """Dense cross-attention forward FLOPs (two matmuls, MAC->FLOPs convention)."""
    per_layer = 4.0 * b_eff * num_heads * seq_q * seq_kv * head_dim
    return per_layer * float(depth)


def bench_loop(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters


def iter_perceiver_modules(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Module]]:
    yield "encoder.level_agg", encoder.level_agg
    yield "decoder.level_decoder", decoder.level_decoder
    if getattr(decoder, "level_decoder_alternate", None) is not None:
        yield "decoder.level_decoder_alternate", decoder.level_decoder_alternate


def report_ms_tflops(name: str, t_sec: float, flops_attn_approx: float) -> None:
    ms = t_sec * 1000
    tf = (flops_attn_approx / t_sec) / 1e12 if t_sec > 0 else float("nan")
    print(f"  {name}: {ms:.4f} ms/iter  (~{tf:.3f} TFLOPS_attn_est)")


def main() -> None:
    from flash_aurora.aurora.model.decoder import Perceiver3DDecoder
    from flash_aurora.aurora.model.encoder import Perceiver3DEncoder

    p = argparse.ArgumentParser(
        description="Aurora multi-Perceiver (encoder + decoder [+ alternate]) SDPA TFLOPS"
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--l-grid", type=int, default=512, dest="l_grid", help="sequence length L (e.g. H*W)")
    p.add_argument("--latent-levels", type=int, default=4, dest="latent_levels")
    p.add_argument("--embed-dim", type=int, default=256, dest="embed_dim")
    p.add_argument("--dec-embed-mult", type=int, default=2, help="decoder embed_dim = embed_dim * this")
    p.add_argument("--num-heads", type=int, default=16, dest="num_heads")
    p.add_argument("--depth-enc", type=int, default=2, dest="depth_enc")
    p.add_argument("--depth-dec", type=int, default=2, dest="depth_dec")
    p.add_argument(
        "--decoder-alternate",
        action="store_true",
        help="Instantiate decoder with separate_perceiver=('z',) so level_decoder_alternate exists.",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--warmup", type=int, default=15, help="Dry-run iterations before timing (each target).")
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    B, L = args.batch, args.l_grid
    latent_levels = args.latent_levels
    c_agg = latent_levels - 1

    surf_vars = ("t2m",)
    atmos_vars = ("z", "u", "v")
    num_atmos_levels = len(atmos_vars)
    embed_dec = args.embed_dim * args.dec_embed_mult
    head_dim_enc = max(1, args.embed_dim // args.num_heads)
    head_dim_dec = max(1, embed_dec // args.num_heads)

    sep = ("z",) if args.decoder_alternate else ()

    encoder = Perceiver3DEncoder(
        surf_vars=surf_vars,
        static_vars=None,
        atmos_vars=atmos_vars,
        latent_levels=latent_levels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=head_dim_enc,
        depth=args.depth_enc,
        drop_rate=0.0,
    ).to(device=device, dtype=dtype)

    decoder = Perceiver3DDecoder(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        embed_dim=embed_dec,
        depth=args.depth_dec,
        head_dim=head_dim_dec,
        num_heads=args.num_heads,
        drop_rate=0.0,
        separate_perceiver=sep,
    ).to(device=device, dtype=dtype)

    encoder.eval()
    decoder.eval()

    x_enc = torch.randn(B, c_agg, L, args.embed_dim, device=device, dtype=dtype)
    level_embed = torch.randn(B, L, num_atmos_levels, embed_dec, device=device, dtype=dtype)
    x_dec_context = torch.randn(B, L, c_agg, embed_dec, device=device, dtype=dtype)

    b_eff = B * L

    flops_enc = estimate_cross_attn_fwd_flops(
        b_eff,
        args.num_heads,
        head_dim_enc,
        c_agg,
        c_agg,
        args.depth_enc,
    )
    flops_dec = estimate_cross_attn_fwd_flops(
        b_eff,
        args.num_heads,
        head_dim_dec,
        num_atmos_levels,
        c_agg,
        args.depth_dec,
    )

    modules = list(iter_perceiver_modules(encoder, decoder))
    flops_strip = flops_enc + flops_dec * (2 if args.decoder_alternate else 1)

    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype}")
    print(
        f"encoder aggregate_levels: x ({B},{c_agg},{L},{args.embed_dim}) depth={args.depth_enc}"
    )
    print(
        f"decoder deaggregate_levels: level_embed ({B},{L},{num_atmos_levels},{embed_dec}), "
        f"context ({B},{L},{c_agg},{embed_dec}) depth={args.depth_dec}"
    )
    print(f"decoder alternate Perceiver: {'yes (separate_perceiver=z)' if args.decoder_alternate else 'no'}")
    print(f"registered Perceiver stacks: {[n for n, _ in modules]}")

    def enc_only():
        with torch.no_grad():
            encoder.aggregate_levels(x_enc)

    def dec_main_only():
        with torch.no_grad():
            decoder.deaggregate_levels(level_embed, x_dec_context, decoder.level_decoder)

    def dec_alt_only():
        with torch.no_grad():
            decoder.deaggregate_levels(
                level_embed, x_dec_context, decoder.level_decoder_alternate
            )

    def strip_all_perceivers():
        with torch.no_grad():
            encoder.aggregate_levels(x_enc)
            decoder.deaggregate_levels(level_embed, x_dec_context, decoder.level_decoder)
            if args.decoder_alternate:
                decoder.deaggregate_levels(
                    level_embed,
                    x_dec_context,
                    decoder.level_decoder_alternate,
                )

    targets: list[tuple[str, Callable[[], None], float]] = [
        ("encoder.level_agg (aggregate_levels)", enc_only, flops_enc),
        ("decoder.level_decoder (deaggregate main)", dec_main_only, flops_dec),
    ]
    if args.decoder_alternate:
        targets.append(
            (
                "decoder.level_decoder_alternate",
                dec_alt_only,
                flops_dec,
            )
        )
    targets.append(
        (
            "strip: enc + dec" + (" + dec_alt" if args.decoder_alternate else ""),
            strip_all_perceivers,
            flops_strip,
        )
    )

    for title, fn, flops in targets:
        t_sdpa = bench_loop(fn, args.warmup, args.iters, device)
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n## {title}")
        print(f"  approx_attn_flops={flops:.3e}")
        report_ms_tflops("SDPA", t_sdpa, flops)


if __name__ == "__main__":
    main()
