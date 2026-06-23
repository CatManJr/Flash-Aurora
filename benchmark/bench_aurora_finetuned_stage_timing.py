#!/usr/bin/env python3
"""Finetuned forward stage timing: encoder / backbone / decoder breakdown.

Uses real ingress IC (same as ``bench_aurora_finetuned_lora.py``). Default: ``hres_t0_finetuned``,
``lora_merged``, and a focused tier set to quantify how much E/D dilutes backbone tier gaps.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_finetuned_stage_timing.py

    uv run python benchmark/bench_aurora_finetuned_stage_timing.py \\
        --presets hres_t0_finetuned hres_0.1 --tiers bf16_mixed@fp32 tf32@fp32

    uv run python benchmark/bench_aurora_finetuned_stage_timing.py --profile-casts
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if os.path.join(_REPO, "profiling") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "profiling"))
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _finetuned_ic import (  # noqa: E402
    FINETUNED_LORA_PRESETS,
    checkpoint_path,
    load_preset_batch,
)
from _pretrained_era5 import purge_gpu, pytorch_reference_tiers, tier_entry  # noqa: E402
from _stage_timing import StageTiming, time_forward_stages  # noqa: E402

import torch

_DEFAULT_STAGE_TIERS: tuple[str, ...] = (
    "bf16_mixed@fp32",
    "bf16_mixed@tf32",
    "tf32@fp32",
    "fp32@fp32",
    "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
    "pytorch_backbone_fp32_encoder_decoder_fp32",
)


def resolve_tier_specs(names: list[str]) -> list[tuple[str, str]]:
    pytorch_map = {label: precision for label, precision, _desc in pytorch_reference_tiers()}
    resolved: list[tuple[str, str]] = []
    for name in names:
        if name in pytorch_map:
            resolved.append((name, pytorch_map[name]))
            continue
        try:
            label, precision, _desc = tier_entry(name)
            resolved.append((label, precision))
        except ValueError:
            resolved.append((name, name))
    return resolved


def build_finetuned(
    config,
    ckpt: Path,
    *,
    precision: str,
    device: torch.device,
):
    from flash_aurora.engine.core.model_registry import ModelFactory

    variant = config.variant
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        use_lora_merged_inference=True,
        inference_precision=precision,
    )
    model.load_checkpoint_local(str(ckpt), strict=variant.strict_checkpoint)
    model.eval()
    return model.to(device)


def _prepare_backbone_input(model: Any, batch: Any) -> tuple[tuple[int, int, int], Any, int]:
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

    _, transformed, patch_res = model._prepare_encoder_batch(batch)
    with torch.inference_mode():
        x = run_with_encoder_decoder_routing(
            model.encoder,
            transformed,
            autocast_bf16=model.autocast_encoder_decoder,
            use_tensor_core=model.encoder_decoder_use_tensor_core,
            lead_time=model.timestep,
        )
    return patch_res, x, batch.metadata.rollout_step


def profile_backbone_cast_buckets(
    model: Any,
    batch: Any,
    *,
    repeat: int,
    device: torch.device,
) -> dict[str, float]:
    """Self-CUDA buckets on backbone-only loop (cast_dtype vs gemm, etc.)."""
    from torch.profiler import ProfilerActivity, profile

    from profile_precision_tiers import _aggregate_buckets

    patch_res, backbone_x, rollout_step = _prepare_backbone_input(model, batch)

    def run_bb() -> None:
        with torch.inference_mode():
            _ = model._run_backbone(
                backbone_x,
                lead_time=model.timestep,
                patch_res=patch_res,
                rollout_step=rollout_step,
            )

    for _ in range(2):
        run_bb()
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(repeat):
            run_bb()
        torch.cuda.synchronize(device)

    buckets, _total = _aggregate_buckets(prof, use_cuda=True)
    return buckets


def print_stage_table(
    preset: str,
    rows: list[tuple[str, StageTiming]],
    *,
    baseline_tier: str | None,
) -> None:
    print(f"\n=== {preset} — stage timing (lora_merged) ===")
    hdr = (
        f"{'tier':<44} {'total':>8} {'enc':>8} {'bb':>8} {'dec':>8} "
        f"{'post':>6} {'bb%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    baseline_bb: float | None = None
    baseline_total: float | None = None
    for tier, t in rows:
        if tier == baseline_tier:
            baseline_bb = t.backbone_ms
            baseline_total = t.total_ms
        print(
            f"{tier:<44} {t.total_ms:8.1f} {t.encoder_ms:8.1f} {t.backbone_ms:8.1f} "
            f"{t.decoder_ms:8.1f} {t.post_ms:6.1f} {t.backbone_pct:5.1f}%"
        )

    if baseline_tier and baseline_bb and baseline_total:
        print(f"\n  vs {baseline_tier} (backbone / total speedup):")
        for tier, t in rows:
            if tier == baseline_tier:
                continue
            bb_ratio = baseline_bb / t.backbone_ms if t.backbone_ms > 0 else float("nan")
            tot_ratio = baseline_total / t.total_ms if t.total_ms > 0 else float("nan")
            print(
                f"    {tier:<40} backbone {bb_ratio:5.2f}x  e2e {tot_ratio:5.2f}x  "
                f"(bb gap {t.backbone_ms - baseline_bb:+.1f} ms, "
                f"e2e gap {t.total_ms - baseline_total:+.1f} ms)"
            )


def print_cast_comparison(
    preset: str,
    cast_rows: list[tuple[str, dict[str, float]]],
) -> None:
    keys = ("cast_dtype", "copy_tensor", "memcpy", "attention_cute_window", "linear", "addmm", "gemm_other")
    print(f"\n=== {preset} — backbone profiler buckets (ms, backbone-only) ===")
    hdr = f"{'tier':<44}" + "".join(f"{k:>12}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for tier, buckets in cast_rows:
        cols = "".join(f"{buckets.get(k, 0.0):12.2f}" for k in keys)
        print(f"{tier:<44}{cols}")


def write_markdown_report(
    path: Path,
    *,
    asset_root: Path,
    gpu_name: str,
    warmup: int,
    repeat: int,
    all_rows: list[tuple[str, str, StageTiming]],
    cast_rows: list[tuple[str, str, dict[str, float]]] | None,
    baseline_tier: str | None,
) -> None:
    lines = [
        "# Finetuned stage timing (real ingress)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- GPU: {gpu_name}",
        f"- Asset root: `{asset_root}`",
        f"- Warmup: {warmup}, repeat: {repeat}",
        "- Mode: `lora_merged` only",
        "",
        "## Per-stage ms (CUDA events)",
        "",
        "| preset | tier | total | encoder | backbone | decoder | post | backbone % |",
        "|--------|------|------:|--------:|---------:|--------:|-----:|-----------:|",
    ]
    for preset, tier, t in all_rows:
        lines.append(
            f"| {preset} | {tier} | {t.total_ms:.1f} | {t.encoder_ms:.1f} | "
            f"{t.backbone_ms:.1f} | {t.decoder_ms:.1f} | {t.post_ms:.1f} | "
            f"{t.backbone_pct:.1f} |"
        )

    if baseline_tier:
        lines.extend(["", f"## Speedup vs `{baseline_tier}`", ""])
        by_preset: dict[str, list[tuple[str, StageTiming]]] = {}
        for preset, tier, t in all_rows:
            by_preset.setdefault(preset, []).append((tier, t))
        lines.append("| preset | tier | backbone speedup | e2e speedup |")
        lines.append("|--------|------|-----------------:|------------:|")
        for preset in sorted(by_preset):
            base = next((t for tier, t in by_preset[preset] if tier == baseline_tier), None)
            if base is None:
                continue
            for tier, t in by_preset[preset]:
                if tier == baseline_tier:
                    continue
                bb = base.backbone_ms / t.backbone_ms if t.backbone_ms > 0 else 0.0
                e2e = base.total_ms / t.total_ms if t.total_ms > 0 else 0.0
                lines.append(f"| {preset} | {tier} | {bb:.2f}x | {e2e:.2f}x |")

    if cast_rows:
        lines.extend(
            [
                "",
                "## Backbone cast / GEMM buckets (profiler, backbone-only)",
                "",
                "| preset | tier | cast_dtype | copy | memcpy | cute_attn | linear | addmm | gemm_other |",
                "|--------|------|----------:|-----:|-------:|----------:|-------:|------:|-----------:|",
            ]
        )
        for preset, tier, buckets in cast_rows:
            lines.append(
                f"| {preset} | {tier} | {buckets.get('cast_dtype', 0):.2f} | "
                f"{buckets.get('copy_tensor', 0):.2f} | {buckets.get('memcpy', 0):.2f} | "
                f"{buckets.get('attention_cute_window', 0):.2f} | {buckets.get('linear', 0):.2f} | "
                f"{buckets.get('addmm', 0):.2f} | {buckets.get('gemm_other', 0):.2f} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["hres_t0_finetuned"],
        choices=FINETUNED_LORA_PRESETS + ("tc_tracking",),
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(_DEFAULT_STAGE_TIERS),
        help="Tier names (default: focused dilution set)",
    )
    parser.add_argument(
        "--baseline-tier",
        default="bf16_mixed@fp32",
        help="Reference tier for speedup table",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--profile-casts",
        action="store_true",
        help="Run torch.profiler backbone-only cast/GEMM buckets per tier",
    )
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    gpu_name = torch.cuda.get_device_name(device)
    tier_specs = resolve_tier_specs(args.tiers)

    print(f"[gpu] {gpu_name}")
    print(f"[asset] {asset_root}")
    print(f"[presets] {', '.join(args.presets)}")
    print(f"[tiers] {', '.join(label for label, _ in tier_specs)}")
    print(f"[warmup] {args.warmup}  [repeat] {args.repeat}")

    all_rows: list[tuple[str, str, StageTiming]] = []
    all_cast_rows: list[tuple[str, str, dict[str, float]]] = []

    for preset in args.presets:
        batch, config = load_preset_batch(preset, asset_root)
        ckpt = checkpoint_path(config, asset_root)
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint missing for {preset}: {ckpt}")
        h, w = batch.spatial_shape
        print(
            f"\n[preset] {preset}  grid={h}x{w}  ckpt={ckpt.name}",
            flush=True,
        )

        preset_rows: list[tuple[str, StageTiming]] = []
        cast_rows: list[tuple[str, dict[str, float]]] = []

        for tier_label, precision in tier_specs:
            print(f"[run] {preset}  {tier_label}...", flush=True)
            model = build_finetuned(config, ckpt, precision=precision, device=device)
            dev_batch = batch.to(device)
            try:
                timing, _pred = time_forward_stages(
                    model,
                    dev_batch,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    device=device,
                )
                preset_rows.append((tier_label, timing))
                all_rows.append((preset, tier_label, timing))
                print(
                    f"       -> total={timing.total_ms:.1f} ms  "
                    f"enc={timing.encoder_ms:.1f} bb={timing.backbone_ms:.1f} "
                    f"dec={timing.decoder_ms:.1f} post={timing.post_ms:.1f}  "
                    f"bb%={timing.backbone_pct:.1f}%",
                    flush=True,
                )

                if args.profile_casts:
                    buckets = profile_backbone_cast_buckets(
                        model,
                        dev_batch,
                        repeat=args.profile_repeat,
                        device=device,
                    )
                    cast_rows.append((tier_label, buckets))
                    all_cast_rows.append((preset, tier_label, buckets))
                    print(
                        f"       [casts] cast_dtype={buckets.get('cast_dtype', 0):.2f} ms  "
                        f"copy={buckets.get('copy_tensor', 0):.2f} ms",
                        flush=True,
                    )
            finally:
                purge_gpu(model, dev_batch)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        print_stage_table(preset, preset_rows, baseline_tier=args.baseline_tier)
        if args.profile_casts and cast_rows:
            print_cast_comparison(preset, cast_rows)

    report_path = args.report_out
    if report_path is None:
        report_path = Path("benchmark") / f"finetuned_stage_{datetime.now():%Y%m%d_%H%M%S}.md"
    write_markdown_report(
        report_path,
        asset_root=asset_root,
        gpu_name=gpu_name,
        warmup=args.warmup,
        repeat=args.repeat,
        all_rows=all_rows,
        cast_rows=all_cast_rows if args.profile_casts else None,
        baseline_tier=args.baseline_tier,
    )


if __name__ == "__main__":
    main()
