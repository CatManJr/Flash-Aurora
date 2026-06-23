#!/usr/bin/env python3
"""Cross-preset inference precision accuracy (seed=42, all presets except ``wave``).

Runs each tier once per preset, compares outputs to the PyTorch FP32 baseline
(``pytorch_backbone_fp32_encoder_decoder_fp32``), and reports per-variable mean
relative error vs upstream ``tests/aurora/test_model.py`` tolerances.

Finetuned presets use ``lora_merged`` (engine default). RNG is fixed at seed 42
before each model load and forward for reproducibility.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_precision_all.py \\
        --asset-root "$AURORA_ASSET_ROOT"

    uv run python benchmark/bench_aurora_precision_all.py \\
        --presets era5_pretrained hres_t0_finetuned --tiers bf16_mixed@fp32 tf32@fp32
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _preset_ic import (  # noqa: E402
    PRECISION_PRESETS,
    checkpoint_path,
    load_preset_batch,
    output_var_tolerances,
)
from _pretrained_era5 import (  # noqa: E402
    _PYTORCH_BASELINE_KEY,
    prediction_tensors,
    purge_gpu,
    pytorch_reference_tiers,
    tier_entry,
)

import torch

_BENCHMARK_SEED = 42

_DEFAULT_PRECISION_TIERS: tuple[str, ...] = (
    _PYTORCH_BASELINE_KEY,
    "bf16_mixed@fp32",
    "bf16_mixed@tf32",
    "tf32@fp32",
    "tf32@tf32",
    "fp32@fp32",
    "bf16@fp32",
    "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
)


@dataclass(frozen=True)
class VarAccuracy:
    group: str
    name: str
    mean_rel: float
    tol: float
    max_abs: float
    ok: bool

    @property
    def key(self) -> str:
        return f"{self.group}.{self.name}"


@dataclass(frozen=True)
class TierAccuracy:
    tier: str
    vars: tuple[VarAccuracy, ...]

    @property
    def passed(self) -> int:
        return sum(1 for v in self.vars if v.ok)

    @property
    def total(self) -> int:
        return len(self.vars)

    @property
    def all_ok(self) -> bool:
        return self.passed == self.total


def set_benchmark_seed(seed: int = _BENCHMARK_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model(
    config,
    ckpt: Path,
    *,
    precision: str,
    device: torch.device,
):
    from flash_aurora.engine.core.model_registry import ModelFactory

    set_benchmark_seed()
    variant = config.variant
    kwargs: dict[str, Any] = {"inference_precision": precision}
    if variant.use_lora:
        kwargs["use_lora_merged_inference"] = True
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        **kwargs,
    )
    model.load_checkpoint_local(str(ckpt), strict=variant.strict_checkpoint)
    model.eval()
    return model.to(device)


def run_forward_tensors(model: Any, batch: Any, *, device: torch.device) -> dict[str, torch.Tensor]:
    set_benchmark_seed()
    dev_batch = batch.to(device)
    with torch.inference_mode():
        pred = model.forward(dev_batch)
    return prediction_tensors(pred)


def accuracy_rows(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    var_specs: tuple[tuple[str, str, float], ...],
) -> tuple[VarAccuracy, ...]:
    rows: list[VarAccuracy] = []
    for group, name, tol in var_specs:
        key = f"{group}.{name}"
        ref = reference[key]
        cand = candidate[key]
        err = (cand - ref).abs()
        mean_rel = float(err.mean().item() / ref.abs().mean().clamp_min(1e-8).item())
        max_abs = float(err.max().item())
        rows.append(
            VarAccuracy(
                group=group,
                name=name,
                mean_rel=mean_rel,
                tol=tol,
                max_abs=max_abs,
                ok=mean_rel <= tol,
            )
        )
    return tuple(rows)


def print_preset_report(
    preset: str,
    results: list[TierAccuracy],
    *,
    baseline_tier: str,
    seed: int,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"PRESET: {preset}")
    print(f"{'=' * 72}")
    print(f"  baseline: {baseline_tier}")
    print(f"  seed: {seed}")
    print(f"  metric: mean(|out-ref|) / mean(|ref|)")

    failures: list[tuple[str, str, float, float]] = []
    for tr in results:
        if tr.tier == baseline_tier:
            print(f"\n  [{tr.tier}] baseline ({tr.total} vars)")
            continue
        status = "PASS" if tr.all_ok else "FAIL"
        print(f"\n  [{tr.tier}] {status} {tr.passed}/{tr.total} vars within tol")
        for v in tr.vars:
            mark = "ok" if v.ok else "FAIL"
            print(
                f"    {v.name:<8} mean_rel={v.mean_rel:10.4e}  tol={v.tol:.0e}  "
                f"max_abs={v.max_abs:.4g}  [{mark}]"
            )
            if not v.ok:
                failures.append((tr.tier, v.name, v.mean_rel, v.tol))

    if failures:
        print(f"\n  ** Failures ({len(failures)}) **")
        for tier, name, mean_rel, tol in failures:
            ratio = mean_rel / tol if tol > 0 else float("inf")
            print(f"    {tier:<44} {name:<8} mean_rel={mean_rel:.4e}  ({ratio:.1f}x tol)")
    else:
        non_base = [r for r in results if r.tier != baseline_tier]
        if non_base:
            print("\n  All compared tiers within tolerance on every output variable.")


def write_markdown_report(
    path: Path,
    *,
    asset_root: Path,
    gpu_name: str,
    torch_version: str,
    seed: int,
    baseline_tier: str,
    all_results: list[tuple[str, tuple[tuple[str, str, float], ...], list[TierAccuracy]]],
) -> None:
    lines = [
        "# Aurora precision suite (all presets except wave)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- GPU: {gpu_name}",
        f"- PyTorch: `{torch_version}`",
        f"- Asset root: `{asset_root}`",
        f"- Seed: **{seed}** (torch / cuda / numpy / random)",
        f"- Baseline: `{baseline_tier}` (PyTorch backbone FP32, E/D FP32, no Triton/CuTe)",
        "- Finetuned presets: `lora_merged`",
        f"- Pollution vars (CAMS): heuristic tol `5e-3` (same as wind/q)",
        "",
    ]

    # Global failure summary
    lines.append("## Failure summary")
    lines.append("")
    any_fail = False
    for preset, _specs, results in all_results:
        for tr in results:
            if tr.tier == baseline_tier:
                continue
            for v in tr.vars:
                if not v.ok:
                    any_fail = True
                    ratio = v.mean_rel / v.tol if v.tol > 0 else 0.0
                    lines.append(
                        f"- **{preset}** / `{tr.tier}` / `{v.name}`: "
                        f"mean_rel={v.mean_rel:.4e} (tol={v.tol:.0e}, **{ratio:.1f}x**)"
                    )
    if not any_fail:
        lines.append("- No failures: every tier passed official/heuristic tolerance on all variables.")

    for preset, var_specs, results in all_results:
        var_names = [name for _g, name, _t in var_specs]
        lines.extend(["", f"## {preset}", ""])
        lines.append(
            "| tier | pass | " + " | ".join(var_names) + " |",
        )
        lines.append("|------|-----:|" + "|".join("---:" for _ in var_names) + "|")
        for tr in results:
            if tr.tier == baseline_tier:
                continue
            by_name = {v.name: v for v in tr.vars}
            cells = []
            for name in var_names:
                v = by_name.get(name)
                if v is None:
                    cells.append("—")
                elif v.ok:
                    cells.append(f"{v.mean_rel:.2e}")
                else:
                    cells.append(f"**{v.mean_rel:.2e}**")
            lines.append(
                f"| {tr.tier} | {tr.passed}/{tr.total} | " + " | ".join(cells) + " |"
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
        default=list(PRECISION_PRESETS),
        choices=PRECISION_PRESETS,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(_DEFAULT_PRECISION_TIERS),
        help="Tier names (baseline + candidates)",
    )
    parser.add_argument(
        "--baseline-tier",
        default=_PYTORCH_BASELINE_KEY,
        help="Tier used as reference (must be in --tiers)",
    )
    parser.add_argument("--seed", type=int, default=_BENCHMARK_SEED)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    set_benchmark_seed(args.seed)

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    gpu_name = torch.cuda.get_device_name(device)
    tier_specs = resolve_tier_specs(args.tiers)

    if args.baseline_tier not in {label for label, _ in tier_specs}:
        raise SystemExit(f"baseline tier {args.baseline_tier!r} not in --tiers")

    print(f"[gpu] {gpu_name}")
    print(f"[asset] {asset_root}")
    print(f"[seed] {args.seed}")
    print(f"[presets] {', '.join(args.presets)}")
    print(f"[tiers] {', '.join(label for label, _ in tier_specs)}")
    print(f"[baseline] {args.baseline_tier}")

    all_results: list[tuple[str, tuple[tuple[str, str, float], ...], list[TierAccuracy]]] = []

    for preset in args.presets:
        print(f"\n[preset] loading IC: {preset}...", flush=True)
        batch, config = load_preset_batch(preset, asset_root)
        ckpt = checkpoint_path(config, asset_root)
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint missing for {preset}: {ckpt}")
        var_specs = output_var_tolerances(config)
        h, w = batch.spatial_shape
        print(
            f"  grid={h}x{w}  vars={len(var_specs)}  ckpt={ckpt.name}",
            flush=True,
        )

        preds: dict[str, dict[str, torch.Tensor]] = {}
        for tier_label, precision in tier_specs:
            print(f"  [run] {tier_label}...", flush=True)
            model = build_model(config, ckpt, precision=precision, device=device)
            try:
                preds[tier_label] = run_forward_tensors(model, batch, device=device)
            finally:
                purge_gpu(model)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        baseline = preds[args.baseline_tier]
        tier_results: list[TierAccuracy] = []
        for tier_label, _precision in tier_specs:
            rows = accuracy_rows(baseline, preds[tier_label], var_specs)
            tier_results.append(TierAccuracy(tier=tier_label, vars=rows))

        print_preset_report(
            preset, tier_results, baseline_tier=args.baseline_tier, seed=args.seed
        )
        all_results.append((preset, var_specs, tier_results))

    report_path = args.report_out
    if report_path is None:
        report_path = Path("benchmark") / f"precision_all_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}.md"
    write_markdown_report(
        report_path,
        asset_root=asset_root,
        gpu_name=gpu_name,
        torch_version=torch.__version__,
        seed=args.seed,
        baseline_tier=args.baseline_tier,
        all_results=all_results,
    )


if __name__ == "__main__":
    main()
