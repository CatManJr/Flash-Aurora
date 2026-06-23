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

    # Fair cross-tier numbers (one fresh process per tier; default for README reports):
    uv run python benchmark/bench_aurora_latency_all.py \\
        --asset-root /root/autodl-tmp/aurora --isolate-tiers \\
        --report-out benchmark/latency_all_latest.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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
    order_tier_specs_for_timing,
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
    if use_lora:
        hdr = f"{'tier':<44} {'eager':>10} {'merged':>10} {'eager/merged':>12} {'vs ref':>8}"
        print(hdr)
        print("-" * len(hdr))
        for tier, eager_s, merged_s, ratio_s, vs_ref in rows:
            print(f"{tier:<44} {eager_s:>10} {merged_s:>10} {ratio_s:>12} {vs_ref:>8}")
    else:
        hdr = f"{'tier':<44} {'forward':>10} {'vs ref':>8}"
        print(hdr)
        print("-" * len(hdr))
        for tier, _eager_s, merged_s, _ratio_s, vs_ref in rows:
            print(f"{tier:<44} {merged_s:>10} {vs_ref:>8}")


_BENCH_WORKER = Path(__file__).resolve().parent / "_latency_tier_worker.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_tier_isolated(
    *,
    preset: str,
    tier_label: str,
    precision: str,
    asset_root: Path,
    warmup: int,
    repeat: int,
) -> dict[str, tuple[float, float, float]]:
    """Spawn a clean process so cuDNN autotune from other tiers cannot skew timing."""
    cmd = [
        sys.executable,
        str(_BENCH_WORKER),
        "--preset",
        preset,
        "--tier-label",
        tier_label,
        "--precision",
        precision,
        "--asset-root",
        str(asset_root),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"isolated tier failed preset={preset!r} tier={tier_label!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    payload = json.loads(proc.stdout.strip())
    return {
        key: (vals["ms"], vals["peak_alloc_mb"], vals["peak_reserved_mb"])
        for key, vals in payload["timings"].items()
    }


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
    isolate_tiers: bool,
    defer_ref: bool = False,
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
            f"- Tier isolation: **{'subprocess per tier' if isolate_tiers else 'single process'}**",
            (
                f"- PyTorch FP32 ref: **timed after custom tiers (--defer-ref)**"
                if defer_ref
                else (
                    "- PyTorch FP32 ref: **timed before custom tiers**"
                    if not isolate_tiers
                    else "- PyTorch FP32 ref: **fresh subprocess per tier**"
                )
            ),
            "- Finetuned presets: `lora_eager` vs `lora_merged` (engine default)",
            "- Pretrained presets: single `forward` column (no LoRA)",
            f"- Reference tier for speedup: `{PYTORCH_FP32_REF_TIER}`",
            "- Excluded: `wave` (MARS ingress)",
            "- Tiers exclude `bf16@*` (see README)",
            "",
        ]
    )

    for preset in sorted(preset_tables):
        use_lora, grid = preset_meta[preset]
        lines.extend([f"## {preset} ({grid})", ""])
        if use_lora:
            lines.append(
                "| Tier | lora_eager (ms) | lora_merged (ms) | eager/merged | vs PyTorch FP32 ref |"
            )
            lines.append(
                "|------|----------------:|-----------------:|-------------:|--------------------:|"
            )
            for tier, eager_s, merged_s, ratio_s, vs_ref in preset_tables[preset]:
                lines.append(
                    f"| {tier} | {eager_s} | {merged_s} | {ratio_s} | {vs_ref} |"
                )
        else:
            lines.append("| Tier | forward (ms) | vs PyTorch FP32 ref |")
            lines.append("|------|-------------:|--------------------:|")
            for tier, _eager_s, merged_s, _ratio_s, vs_ref in preset_tables[preset]:
                lines.append(f"| {tier} | {merged_s} | {vs_ref} |")
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
    parser.add_argument(
        "--isolate-tiers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each preset×tier in a fresh subprocess (fair cuDNN state; default: on)",
    )
    parser.add_argument(
        "--defer-ref",
        action="store_true",
        help="With --no-isolate-tiers: time PyTorch FP32 ref after custom tiers "
        "(reproduces cuDNN cross-tier warmup artifact)",
    )
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
    print(f"[isolate] {args.isolate_tiers}")
    if not args.isolate_tiers:
        print(f"[defer-ref] {args.defer_ref}")

    ref_tier_specs, other_tier_specs = order_tier_specs_for_timing(tier_specs)
    jit_warmup_done = False

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

        def _run_tier(tier_label: str, precision: str) -> None:
            print(f"  [run] {tier_label}...", flush=True)
            if args.isolate_tiers:
                tier_timings[tier_label] = _run_tier_isolated(
                    preset=preset,
                    tier_label=tier_label,
                    precision=precision,
                    asset_root=asset_root,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            else:
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

        if args.isolate_tiers:
            for tier_label, precision in tier_specs:
                _run_tier(tier_label, precision)
        else:
            if args.defer_ref:
                if not jit_warmup_done and other_tier_specs:
                    _jit_warmup(asset_root, device)
                    jit_warmup_done = True
                for tier_label, precision in other_tier_specs:
                    _run_tier(tier_label, precision)
                for tier_label, precision in ref_tier_specs:
                    _run_tier(tier_label, precision)
            else:
                for tier_label, precision in ref_tier_specs:
                    _run_tier(tier_label, precision)
                if not jit_warmup_done and other_tier_specs:
                    _jit_warmup(asset_root, device)
                    jit_warmup_done = True
                for tier_label, precision in other_tier_specs:
                    _run_tier(tier_label, precision)

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
        stem = "latency_all_isolated" if args.isolate_tiers else "latency_all_single_process"
        report_path = Path("benchmark") / f"{stem}.md"
    write_markdown_report(
        report_path,
        asset_root=asset_root,
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        cute_arch=cute_arch,
        warmup=args.warmup,
        repeat=args.repeat,
        isolate_tiers=args.isolate_tiers,
        defer_ref=args.defer_ref,
        preset_tables=preset_tables,
        preset_meta=preset_meta,
    )


if __name__ == "__main__":
    main()
