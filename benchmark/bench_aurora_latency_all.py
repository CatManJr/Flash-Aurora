#!/usr/bin/env python3
"""Unified end-to-end forward latency (all presets except ``wave``).

One table per preset: every inference tier, with ``lora_eager`` vs ``lora_merged`` on
finetuned models (``forward`` only on pretrained presets). Includes ``era5_pretrained``,
``small_pretrained``, ``tc_tracking``, etc.

Default tiers exclude ``bf16@*`` (no speed win; worse precision drift).

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_latency_all.py \\
        --asset-root /root/autodl-tmp/aurora

    uv run python benchmark/bench_aurora_latency_all.py \\
        --presets hres_t0_finetuned tc_tracking --warmup 2 --repeat 5
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _latency_bench import (  # noqa: E402
    DEFAULT_LATENCY_TIERS,
    PYTORCH_FP32_REF_TIER,
    resolve_tier_specs,
    run_tier_lora_modes,
)
from _preset_ic import PRECISION_PRESETS, checkpoint_path, load_preset_batch  # noqa: E402

import torch


def _jit_warmup(asset_root: Path, device: torch.device) -> None:
    """One untimed forward so CuTe/Triton JIT is not charged to the first benchmark cell."""
    preset = "hres_t0_finetuned"
    precision = "bf16_mixed@fp32"
    print(f"[jit-warmup] {preset} {precision} lora_merged (untimed)...", flush=True)
    batch, config = load_preset_batch(preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    run_tier_lora_modes(
        config=config,
        ckpt=ckpt,
        precision=precision,
        batch=batch,
        device=device,
        warmup=1,
        repeat=1,
    )
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("[jit-warmup] done", flush=True)


def _cuda_version() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    return torch.version.cuda or "unknown"


def _format_eager_merged(
    use_lora: bool,
    timings: dict[str, tuple[float, float, float]],
) -> tuple[str, str, str]:
    if use_lora:
        eager_ms = timings["lora_eager"][0]
        merged_ms = timings["lora_merged"][0]
        ratio = f"{eager_ms / merged_ms:.2f}x" if merged_ms > 0 else "—"
        return f"{eager_ms:.1f}", f"{merged_ms:.1f}", ratio
    fwd = timings["forward"][0]
    return "—", f"{fwd:.1f}", "—"


def print_preset_latency_table(
    preset: str,
    *,
    use_lora: bool,
    grid: str,
    rows: list[tuple[str, str, str, str, str]],
) -> None:
    print(f"\n=== {preset} ({grid}) ===")
    hdr = f"{'tier':<44} {'eager':>10} {'merged':>10} {'eager/merged':>12} {'vs ref':>8}"
    print(hdr)
    print("-" * len(hdr))
    for tier, eager_s, merged_s, ratio_s, vs_ref in rows:
        print(f"{tier:<44} {eager_s:>10} {merged_s:>10} {ratio_s:>12} {vs_ref:>8}")


def write_markdown_report(
    path: Path,
    *,
    asset_root: Path,
    gpu_name: str,
    torch_version: str,
    cuda_version: str,
    cute_arch: str | None,
    warmup: int,
    repeat: int,
    preset_tables: dict[str, list[tuple[str, str, str, str, str]]],
    preset_meta: dict[str, tuple[bool, str]],
) -> None:
    lines = [
        "# Aurora end-to-end latency (all presets except wave)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- GPU: {gpu_name}",
        f"- PyTorch: `{torch_version}`",
        f"- CUDA: `{cuda_version}`",
    ]
    if cute_arch:
        lines.append(f"- `CUTE_DSL_ARCH`: `{cute_arch}`")
    lines.extend(
        [
            f"- Asset root: `{asset_root}`",
            f"- Warmup: {warmup}, repeat: {repeat}",
            "- Finetuned presets: `lora_eager` vs `lora_merged` (engine default)",
            "- Pretrained presets: single forward in **merged** column (`eager` = —)",
            f"- Reference tier for speedup: `{PYTORCH_FP32_REF_TIER}`",
            "- Excluded: `wave` (MARS ingress)",
            "- Tiers exclude `bf16@*` (see README)",
            "",
        ]
    )

    for preset in sorted(preset_tables):
        use_lora, grid = preset_meta[preset]
        lines.extend([f"## {preset} ({grid})", ""])
        lines.append(
            "| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |"
        )
        lines.append("|------|----------------:|-----------------:|-------------:|--------------------:|")
        for tier, eager_s, merged_s, ratio_s, vs_ref in preset_tables[preset]:
            lines.append(
                f"| {tier} | {eager_s} | {merged_s} | {ratio_s} | {vs_ref} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(PRECISION_PRESETS),
        choices=PRECISION_PRESETS,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(DEFAULT_LATENCY_TIERS),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    gpu_name = torch.cuda.get_device_name(device)
    torch_version = torch.__version__
    cuda_version = _cuda_version()
    cute_arch = os.environ.get("CUTE_DSL_ARCH")
    tier_specs = resolve_tier_specs(args.tiers)

    print(f"[gpu] {gpu_name}")
    print(f"[torch] {torch_version}  cuda={cuda_version}")
    if cute_arch:
        print(f"[cute] CUTE_DSL_ARCH={cute_arch}")
    print(f"[asset] {asset_root}")
    print(f"[presets] {', '.join(args.presets)}")
    print(f"[tiers] {', '.join(args.tiers)}")
    print(f"[warmup] {args.warmup}  [repeat] {args.repeat}")

    _jit_warmup(asset_root, device)

    preset_tables: dict[str, list[tuple[str, str, str, str, str]]] = {}
    preset_meta: dict[str, tuple[bool, str]] = {}
    ref_merged_ms: dict[str, float] = {}

    for preset in args.presets:
        print(f"\n[preset] loading IC: {preset}...", flush=True)
        batch, config = load_preset_batch(preset, asset_root)
        ckpt = checkpoint_path(config, asset_root)
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint missing for {preset}: {ckpt}")
        h, w = batch.spatial_shape
        grid = f"{h}x{w}"
        use_lora = config.variant.use_lora
        preset_meta[preset] = (use_lora, grid)
        print(
            f"  model={config.variant.model_class}  grid={grid}  "
            f"use_lora={use_lora}  ckpt={ckpt.name}",
            flush=True,
        )

        tier_timings: dict[str, dict[str, tuple[float, float, float]]] = {}
        for tier_label, precision in tier_specs:
            print(f"  [run] {tier_label}...", flush=True)
            tier_timings[tier_label] = run_tier_lora_modes(
                config=config,
                ckpt=ckpt,
                precision=precision,
                batch=batch,
                device=device,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if PYTORCH_FP32_REF_TIER in tier_timings:
            if use_lora:
                ref_merged_ms[preset] = tier_timings[PYTORCH_FP32_REF_TIER]["lora_merged"][0]
            else:
                ref_merged_ms[preset] = tier_timings[PYTORCH_FP32_REF_TIER]["forward"][0]

        rows: list[tuple[str, str, str, str, str]] = []
        for tier_label, _precision in tier_specs:
            eager_s, merged_s, ratio_s = _format_eager_merged(
                use_lora, tier_timings[tier_label]
            )
            if tier_label == PYTORCH_FP32_REF_TIER:
                vs_ref = "base"
            elif preset in ref_merged_ms:
                merged_val = float(merged_s) if merged_s != "—" else 0.0
                ref = ref_merged_ms[preset]
                vs_ref = f"{ref / merged_val:.2f}x" if merged_val > 0 else "—"
            else:
                vs_ref = "—"
            rows.append((tier_label, eager_s, merged_s, ratio_s, vs_ref))
            if use_lora:
                print(
                    f"       {tier_label}: eager={eager_s} merged={merged_s} "
                    f"ratio={ratio_s} vs_ref={vs_ref}",
                    flush=True,
                )
            else:
                print(f"       {tier_label}: forward={merged_s} vs_ref={vs_ref}", flush=True)

        preset_tables[preset] = rows
        print_preset_latency_table(preset, use_lora=use_lora, grid=grid, rows=rows)

    report_path = args.report_out
    if report_path is None:
        report_path = Path("benchmark") / f"latency_all_{datetime.now():%Y%m%d_%H%M%S}.md"
    write_markdown_report(
        report_path,
        asset_root=asset_root,
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        cute_arch=cute_arch,
        warmup=args.warmup,
        repeat=args.repeat,
        preset_tables=preset_tables,
        preset_meta=preset_meta,
    )


if __name__ == "__main__":
    main()
