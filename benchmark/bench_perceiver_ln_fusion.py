#!/usr/bin/env python3
"""Benchmark **Triton LayerNorm + residual fusion** in :class:`~aurora.model.perceiver.PerceiverResampler`
(``use_triton_ln_residual_fusion`` / encoder & decoder ``use_triton_perceiver_ln_fusion``).

Compared to the eager PyTorch path (``LayerNorm(x)`` then ``+ residual``), the fused kernel aims to
reduce peak activation traffic (fewer intermediate tensors).

Reports:

- Wall time per forward (``ms/iter``) - baseline (fusion off) vs fused (fusion on).
- Peak CUDA allocated memory during the timed loop - ``torch.cuda.max_memory_allocated``.

Examples::

    PYTHONPATH=aurora uv run python benchmark/bench_perceiver_ln_fusion.py --mode micro
    PYTHONPATH=aurora uv run python benchmark/bench_perceiver_ln_fusion.py --mode strip --warmup 10 --iters 60

Use ``--dtype fp32`` for stricter numerical debugging (slower).

Requires CUDA. Triton must import successfully for the fused path to diverge from baseline; otherwise
:class:`~aurora.model.perceiver.PerceiverResampler` falls back to eager and timings match.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

_REPO = Path(__file__).resolve().parents[1]


def bench_loop(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters


def bench_peak_allocated_mb(fn, warmup: int, iters: int, device: torch.device) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def main() -> None:
    from flash_aurora.aurora.model.perceiver import PerceiverResampler
    import flash_aurora.aurora.model.perceiver as perceiver_mod

    p = argparse.ArgumentParser(
        description="Perceiver Triton LN+residual fusion vs eager (latency + peak CUDA MB)"
    )
    p.add_argument(
        "--mode",
        choices=("micro", "encoder", "decoder", "strip"),
        default="micro",
        help=(
            "micro = PerceiverResampler only; encoder / decoder / strip = "
            "aggregate_levels / deaggregate(main) / both chained (same tensors as TFLOPS bench)."
        ),
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--l-grid", type=int, default=512, dest="l_grid")
    p.add_argument("--latent-levels", type=int, default=4, dest="latent_levels")
    p.add_argument("--embed-dim", type=int, default=256, dest="embed_dim")
    p.add_argument("--dec-embed-mult", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=16, dest="num_heads")
    p.add_argument("--depth-enc", type=int, default=2, dest="depth_enc")
    p.add_argument("--depth-dec", type=int, default=2, dest="depth_dec")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    B, L = args.batch, args.l_grid
    c_agg = args.latent_levels - 1
    surf_vars = ("t2m",)
    atmos_vars = ("z", "u", "v")
    num_atmos_levels = len(atmos_vars)
    embed_dec = args.embed_dim * args.dec_embed_mult
    head_dim_enc = max(1, args.embed_dim // args.num_heads)
    head_dim_dec = max(1, embed_dec // args.num_heads)

    triton_ln_ok = getattr(perceiver_mod, "_TRITON_LN_RESIDUAL_AVAILABLE", False)
    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype}")
    print(f"Triton LN fusion available (import path): {triton_ln_ok}")
    print(f"mode={args.mode}")

    def build_encoder_decoder(use_fusion: bool):
        from flash_aurora.aurora.model.decoder import Perceiver3DDecoder
        from flash_aurora.aurora.model.encoder import Perceiver3DEncoder

        enc = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=None,
            atmos_vars=atmos_vars,
            latent_levels=args.latent_levels,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            head_dim=head_dim_enc,
            depth=args.depth_enc,
            drop_rate=0.0,
            use_triton_perceiver_ln_fusion=use_fusion,
        ).to(device=device, dtype=dtype)

        dec = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            embed_dim=embed_dec,
            depth=args.depth_dec,
            head_dim=head_dim_dec,
            num_heads=args.num_heads,
            drop_rate=0.0,
            use_triton_perceiver_ln_fusion=use_fusion,
        ).to(device=device, dtype=dtype)
        enc.eval()
        dec.eval()
        return enc, dec

    x_enc = torch.randn(B, c_agg, L, args.embed_dim, device=device, dtype=dtype)
    level_embed = torch.randn(B, L, num_atmos_levels, embed_dec, device=device, dtype=dtype)
    x_dec_context = torch.randn(B, L, c_agg, embed_dec, device=device, dtype=dtype)

    enc_off, dec_off = build_encoder_decoder(False)
    enc_on, dec_on = build_encoder_decoder(True)
    enc_on.load_state_dict(enc_off.state_dict())
    dec_on.load_state_dict(dec_off.state_dict())

    # Micro: standalone resampler - same weights for fair compare
    rs_off = PerceiverResampler(
        latent_dim=args.embed_dim,
        context_dim=args.embed_dim,
        depth=args.depth_enc,
        head_dim=head_dim_enc,
        num_heads=args.num_heads,
        use_triton_ln_residual_fusion=False,
    ).to(device=device, dtype=dtype)
    rs_on = PerceiverResampler(
        latent_dim=args.embed_dim,
        context_dim=args.embed_dim,
        depth=args.depth_enc,
        head_dim=head_dim_enc,
        num_heads=args.num_heads,
        use_triton_ln_residual_fusion=True,
    ).to(device=device, dtype=dtype)
    rs_on.load_state_dict(rs_off.state_dict())
    rs_off.eval()
    rs_on.eval()
    lat_micro = torch.randn(B * L, c_agg, args.embed_dim, device=device, dtype=dtype)
    ctx_micro = torch.randn(B * L, c_agg, args.embed_dim, device=device, dtype=dtype)

    def enc_only(enc_mod):
        def _():
            with torch.no_grad():
                enc_mod.aggregate_levels(x_enc)

        return _

    def dec_only(dec_mod):
        def _():
            with torch.no_grad():
                dec_mod.deaggregate_levels(level_embed, x_dec_context, dec_mod.level_decoder)

        return _

    def strip(enc_mod, dec_mod):
        def _():
            with torch.no_grad():
                enc_mod.aggregate_levels(x_enc)
                dec_mod.deaggregate_levels(level_embed, x_dec_context, dec_mod.level_decoder)

        return _

    def micro(rs):
        def _():
            with torch.no_grad():
                rs(lat_micro, ctx_micro)

        return _

    targets: list[tuple[str, object]] = []
    if args.mode == "micro":
        targets.append(("PerceiverResampler (micro)", micro(rs_off), micro(rs_on)))
    elif args.mode == "encoder":
        targets.append(("encoder.aggregate_levels", enc_only(enc_off), enc_only(enc_on)))
    elif args.mode == "decoder":
        targets.append(("decoder.deaggregate_levels (main)", dec_only(dec_off), dec_only(dec_on)))
    else:
        targets.append(("strip: enc agg + dec main", strip(enc_off, dec_off), strip(enc_on, dec_on)))

    for title, fn_off, fn_on in targets:
        # Baseline (eager LN + add)
        t_off = bench_loop(fn_off, args.warmup, args.iters, device)
        mb_off = bench_peak_allocated_mb(fn_off, args.warmup, args.iters, device)
        gc.collect()
        torch.cuda.empty_cache()

        # Fused LN + residual (when Triton path active)
        t_on = bench_loop(fn_on, args.warmup, args.iters, device)
        mb_on = bench_peak_allocated_mb(fn_on, args.warmup, args.iters, device)
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n## {title}")
        print(f"  eager (fusion off): {t_off * 1000:.4f} ms/iter   peak_cuda_alloc≈{mb_off:.1f} MiB")
        print(f"  fused (fusion on):  {t_on * 1000:.4f} ms/iter   peak_cuda_alloc≈{mb_on:.1f} MiB")
        if t_off > 0 and t_on > 0:
            print(f"  speedup (walltime): {t_off / t_on:.3f}x")
        if mb_off > 0:
            print(f"  peak_alloc ratio (fused/eager): {mb_on / mb_off:.3f}x  (Δ {(mb_on - mb_off):+.1f} MiB)")


if __name__ == "__main__":
    main()
