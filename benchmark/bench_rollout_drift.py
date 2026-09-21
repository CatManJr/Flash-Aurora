#!/usr/bin/env python3
"""Autoregressive multi-step drift versus the unfused PyTorch FP32 reference.

Each tier rolls out independently from the same initial condition. Step k of a
candidate is compared to step k of the reference trajectory (not teacher-forced).
This is implementation fidelity under AR feedback, not WeatherBench skill.
Perceiver encoder/decoder stays FP32; only the backbone may change
(``bf16_mixed@fp32``, ``tf32@fp32``, ``tf32x3@fp32``). Named ``tf32`` / ``tf32x3``
/ ``bf16_mixed`` presets are rejected because they turn on Perceiver TF32.

Example::

    export AURORA_ASSET_ROOT=/path/to/data/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_rollout_drift.py \\
        --presets era5_pretrained cams --horizon-hours 240
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
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
MEDIUM_RANGE_LEAD_HOURS = 240
_DEFAULT_TIERS: tuple[str, ...] = (
    _PYTORCH_BASELINE_KEY,
    "bf16_mixed@fp32",
    "tf32@fp32",
    "tf32x3@fp32",
    "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
)
_NAMED_PRESETS_WITH_PERCEIVER_TF32 = frozenset({"tf32", "tf32x3", "bf16_mixed"})
CLOSEDLOOP_WORKER = Path(_BENCH_DIR) / "_closedloop_tier_worker.py"
_PRESET_PLOT_ORDER: tuple[str, ...] = (
    "era5_pretrained",
    "small_pretrained",
    "aurora_v1p5",
    "aurora_v1p5_ensemble",
    "hres_t0_finetuned",
    "tc_tracking",
    "hres_0.1",
    "cams",
)
_PRESET_TITLES: dict[str, str] = {
    "era5_pretrained": "0.25 Pretrained",
    "small_pretrained": "0.25 Small",
    "aurora_v1p5": "Aurora 1.5",
    "aurora_v1p5_ensemble": "Aurora 1.5 Ensemble",
    "hres_t0_finetuned": "0.25 Fine-Tuned",
    "tc_tracking": "0.25 Fine-Tuned (TC)",
    "hres_0.1": "0.1 Fine-Tuned",
    "cams": "0.4 Air Pollution",
}


def set_benchmark_seed(seed: int = _BENCHMARK_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_perceiver_fp32(precision: str) -> None:
    """Closed-loop may change the backbone; Perceiver encoder/decoder stay FP32."""
    raw = precision.strip().lower().replace("-", "_")
    if raw in _NAMED_PRESETS_WITH_PERCEIVER_TF32:
        raise ValueError(
            f"{precision!r} turns on Perceiver TF32; use {raw}@fp32 so encoder/decoder stay FP32."
        )
    from flash_aurora.models.inference_precision import (
        EncoderDecoderMatmulLevel,
        resolve_inference_config,
    )

    if raw in {label for label, _p, _d in pytorch_reference_tiers()}:
        return
    cfg = resolve_inference_config(precision)
    if cfg is None:
        raise ValueError(f"Could not resolve inference tier {precision!r}.")
    if cfg.encoder_decoder_matmul_level != EncoderDecoderMatmulLevel.FP32:
        raise ValueError(
            f"{precision!r} sets Perceiver encoder/decoder to "
            f"{cfg.encoder_decoder_matmul_level.value}; closed-loop requires FP32."
        )


def steps_for_horizon(horizon_hours: int, timestep_hours: float) -> int:
    if timestep_hours <= 0:
        raise ValueError(f"timestep_hours must be positive, got {timestep_hours}")
    steps = int(round(horizon_hours / timestep_hours))
    if steps < 1:
        raise ValueError(f"horizon {horizon_hours} h is shorter than one {timestep_hours:g} h step")
    reconstructed = steps * timestep_hours
    if abs(reconstructed - horizon_hours) > 1e-6:
        raise ValueError(
            f"horizon {horizon_hours} h is not an integer number of {timestep_hours:g} h steps"
        )
    return steps


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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


def build_model(config, ckpt: Path, *, precision: str, device: torch.device):
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


def mean_rel(ref: torch.Tensor, cand: torch.Tensor) -> float:
    err = (cand - ref).abs()
    return float(err.mean().item() / ref.abs().mean().clamp_min(1e-8).item())


def rollout_tensors(
    model: Any,
    batch: Any,
    *,
    steps: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], float, float]:
    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout
    from flash_aurora.models.aurora.rollout import rollout as legacy_rollout
    from flash_aurora.models.aurora_v1p5.rollout import rollout as v1p5_rollout

    set_benchmark_seed()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stream = (
        v1p5_rollout(model, batch, steps)
        if model_uses_v1p5_rollout(model)
        else legacy_rollout(model, batch, steps)
    )
    preds: list[dict[str, torch.Tensor]] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for pred in stream:
            preds.append(prediction_tensors(pred))
            del pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _synchronize(device)
    forecast_s = time.perf_counter() - t0
    peak_gib = 0.0
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    return preds, peak_gib, forecast_s


def compare_step(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    var_specs: tuple[tuple[str, str, float], ...],
) -> dict[str, Any]:
    rows = []
    worst_name = ""
    worst_rel = -1.0
    n_fail = 0
    for group, name, tol in var_specs:
        key = f"{group}.{name}"
        rel = mean_rel(reference[key], candidate[key])
        ok = rel <= tol
        if not ok:
            n_fail += 1
        if rel > worst_rel:
            worst_rel = rel
            worst_name = name
        rows.append({"name": name, "mean_rel": rel, "tol": tol, "ok": ok})
    return {
        "vars": rows,
        "n_fail": n_fail,
        "n_vars": len(var_specs),
        "worst_name": worst_name,
        "worst_rel": worst_rel,
    }


def format_forecast_timing_table(
    timing: dict[str, dict[str, Any]],
    *,
    baseline: str,
) -> list[str]:
    lines = [
        "| tier | load (s) | forecast (s) | per step (s) | peak GiB | vs FP32 forecast |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ref_forecast = None
    ref = timing.get(baseline)
    if ref and ref.get("ok"):
        ref_forecast = float(ref["forecast_s"])
    for tier, row in timing.items():
        if not row.get("ok"):
            err = row.get("error", "failed")
            lines.append(f"| {tier} | — | — | — | — | {err} |")
            continue
        vs = "base"
        if tier != baseline and ref_forecast and row["forecast_s"] > 0:
            vs = f"{ref_forecast / float(row['forecast_s']):.2f}x"
        lines.append(
            f"| {tier} | {row['load_s']:.1f} | {row['forecast_s']:.1f} | "
            f"{row['per_step_s']:.2f} | {row['peak_gib']:.1f} | {vs} |"
        )
    return lines


def write_markdown(
    path: Path,
    *,
    payload: dict[str, Any],
) -> None:
    lines = [
        "# Autoregressive rollout drift vs unfused PyTorch FP32",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Asset root: `{payload['asset_root']}`",
        f"- Seed: **{payload['seed']}**",
        f"- Horizon: **{payload.get('horizon_hours', MEDIUM_RANGE_LEAD_HOURS)} h** (medium range)",
        "- Perceiver encoder/decoder: **FP32** (backbone may be mixed / TF32 / 3xTF32)",
        "- Metric: $\\bar{e}_v=\\mathrm{mean}(|y-\\hat{y}|)/\\mathrm{mean}(|\\hat{y}|)$ at each AR step",
        "- Each tier rolls out on its own predictions from the same IC (not teacher-forced)",
        "",
    ]
    for preset, block in payload["presets"].items():
        hours = block["timestep_hours"]
        lines.append(
            f"## `{preset}` ({block['steps']} steps, {hours:g} h/step, "
            f"{block['steps'] * hours:g} h lead)"
        )
        lines.append("")
        timing = block.get("timing") or {}
        if timing:
            lines.extend(format_forecast_timing_table(timing, baseline=payload["baseline"]))
            lines.append("")
        else:
            lines.append(
                "Peak allocated VRAM (GiB): "
                + ", ".join(f"{k}={v:.1f}" for k, v in block["peak_gib"].items())
            )
            lines.append("")
        for tier, series in block["tiers"].items():
            if tier == payload["baseline"] or not series:
                continue
            lines.append(f"### {tier}")
            lines.append("")
            lines.append("| step | lead (h) | fail | worst | worst $\\bar{e}_v$ | 2t | msl | 10u | pm10 |")
            lines.append("| ---: | -------: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
            for row in series:
                by_name = {v["name"]: v["mean_rel"] for v in row["vars"]}

                def _fmt(name: str) -> str:
                    val = by_name.get(name)
                    return "—" if val is None else f"{val:.3e}"

                lines.append(
                    f"| {row['step']} | {row['lead_hours']:g} | "
                    f"{row['n_fail']}/{row['n_vars']} | {row['worst_name']} | "
                    f"{row['worst_rel']:.3e} | {_fmt('2t')} | {_fmt('msl')} | "
                    f"{_fmt('10u')} | {_fmt('pm10')} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_drift(payload: dict[str, Any], dest: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    presets = [p for p in _PRESET_PLOT_ORDER if p in payload["presets"]]
    presets.extend(p for p in payload["presets"] if p not in presets)
    n = len(presets)
    ncols = min(3, max(n, 1))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    colors = {
        "bf16_mixed@fp32": "#0D7377",
        "tf32@fp32": "#C45C26",
        "tf32x3@fp32": "#6A1B9A",
        "pytorch_backbone_autocast_bf16_encoder_decoder_fp32": "#90A4AE",
    }
    labels = {
        "bf16_mixed@fp32": "bf16_mixed@fp32",
        "tf32@fp32": "tf32@fp32",
        "tf32x3@fp32": "tf32x3@fp32",
        "pytorch_backbone_autocast_bf16_encoder_decoder_fp32": "PyTorch autocast",
    }
    for idx, preset in enumerate(presets):
        ax = axes[idx // ncols][idx % ncols]
        block = payload["presets"][preset]
        for tier, series in block["tiers"].items():
            if tier == payload["baseline"] or not series:
                continue
            xs = [row["lead_hours"] for row in series]
            ys = [row["worst_rel"] for row in series]
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=3.5,
                color=colors.get(tier, "#333333"),
                label=labels.get(tier, tier),
            )
        ax.set_yscale("log")
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(r"worst-variable $\bar{e}_v$ vs FP32 AR ref")
        ax.set_title(_PRESET_TITLES.get(preset, preset))
        ax.legend(frameon=False, fontsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(
        "Autoregressive drift vs unfused PyTorch FP32 (same IC, seed 42)",
        fontsize=11,
        y=1.02,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)


def run_tier_isolated(
    *,
    preset: str,
    precision: str,
    steps: int,
    asset_root: Path,
    step_dir: Path,
) -> dict[str, Any]:
    """Spawn a fresh process so CuTe/Triton cannot leak VRAM across tiers."""
    require_perceiver_fp32(precision)
    if step_dir.exists():
        shutil.rmtree(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(CLOSEDLOOP_WORKER),
        "--preset",
        preset,
        "--precision",
        precision,
        "--steps",
        str(steps),
        "--asset-root",
        str(asset_root),
        "--step-dir",
        str(step_dir),
    ]
    err_path = step_dir / "worker.stderr"
    preds: list[dict[str, torch.Tensor]] = []
    meta: dict[str, Any] = {}
    with err_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=_REPO,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=err_f,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("event")
            if kind == "step":
                path = Path(msg["path"])
                tensors = torch.load(path, map_location="cpu", weights_only=False)
                preds.append(tensors)
                path.unlink(missing_ok=True)
            elif kind == "done":
                meta = msg
        rc = proc.wait()
    stderr_text = err_path.read_text(encoding="utf-8") if err_path.is_file() else ""
    shutil.rmtree(step_dir, ignore_errors=True)
    if rc != 0:
        tail = stderr_text[-4000:] if stderr_text else ""
        raise RuntimeError(
            f"isolated closed-loop failed preset={preset!r} precision={precision!r} rc={rc}\n{tail}"
        )
    if len(preds) != steps:
        raise RuntimeError(
            f"isolated closed-loop returned {len(preds)} steps, expected {steps} "
            f"(preset={preset!r} precision={precision!r})"
        )
    return {
        "preds": preds,
        "peak_gib": float(meta.get("peak_gib", 0.0)),
        "load_s": float(meta.get("load_s", 0.0)),
        "forecast_s": float(meta.get("forecast_s", 0.0)),
        "per_step_s": float(meta.get("per_step_s", 0.0)),
    }


def run_preset(
    *,
    preset: str,
    asset_root: Path,
    steps: int,
    tier_specs: list[tuple[str, str]],
    device: torch.device,
    baseline: str,
    isolate_tiers: bool,
    scratch_root: Path,
) -> dict[str, Any]:
    batch, config = load_preset_batch(preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    var_specs = output_var_tolerances(config)
    hours = float(config.variant.timestep_hours)
    print(f"\n=== {preset}  steps={steps}  dt={hours:g}h  ckpt={ckpt.name} ===", flush=True)

    trajectories: dict[str, list[dict[str, torch.Tensor]]] = {}
    peak_gib: dict[str, float] = {}
    timing: dict[str, dict[str, Any]] = {}
    for label, precision in tier_specs:
        require_perceiver_fp32(precision)
        print(f"  [load] {label} ({precision})", flush=True)
        try:
            if isolate_tiers:
                result = run_tier_isolated(
                    preset=preset,
                    precision=precision,
                    steps=steps,
                    asset_root=asset_root,
                    step_dir=scratch_root / f"{preset}_{label}",
                )
                trajectories[label] = result["preds"]
                peak_gib[label] = result["peak_gib"]
                timing[label] = {
                    "ok": True,
                    "load_s": result["load_s"],
                    "forecast_s": result["forecast_s"],
                    "per_step_s": result["per_step_s"],
                    "peak_gib": result["peak_gib"],
                }
            else:
                purge_gpu()
                t_load = time.perf_counter()
                model = build_model(config, ckpt, precision=precision, device=device)
                _synchronize(device)
                load_s = time.perf_counter() - t_load
                preds, peak, forecast_s = rollout_tensors(
                    model, batch, steps=steps, device=device
                )
                trajectories[label] = preds
                peak_gib[label] = peak
                per_step_s = forecast_s / steps if steps else 0.0
                timing[label] = {
                    "ok": True,
                    "load_s": load_s,
                    "forecast_s": forecast_s,
                    "per_step_s": per_step_s,
                    "peak_gib": peak,
                }
                del model
                purge_gpu()
                gc.collect()
            print(
                f"  [done] {label}  forecast={timing[label]['forecast_s']:.1f}s  "
                f"peak={peak_gib[label]:.1f} GiB",
                flush=True,
            )
        except Exception as exc:
            print(f"  [fail] {label}: {type(exc).__name__}: {exc}", flush=True)
            timing[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            trajectories[label] = []
            peak_gib[label] = 0.0
            purge_gpu()
            gc.collect()

    ref_traj = trajectories.get(baseline) or []
    tiers_out: dict[str, Any] = {baseline: []}
    if not ref_traj:
        print(f"  [skip-compare] baseline {baseline} produced no trajectory", flush=True)
    for label, _precision in tier_specs:
        if label == baseline:
            continue
        cand = trajectories.get(label) or []
        if not cand or not ref_traj:
            tiers_out[label] = []
            continue
        rows = []
        for step, (ref, pred) in enumerate(zip(ref_traj, cand), start=1):
            stats = compare_step(ref, pred, var_specs)
            stats["step"] = step
            stats["lead_hours"] = step * hours
            rows.append(stats)
            print(
                f"  [{label}] step {step:02d}  fail {stats['n_fail']}/{stats['n_vars']}  "
                f"worst {stats['worst_name']}={stats['worst_rel']:.3e}",
                flush=True,
            )
        tiers_out[label] = rows
        del cand
        trajectories[label] = []
        gc.collect()
    return {
        "steps": steps,
        "timestep_hours": hours,
        "peak_gib": peak_gib,
        "timing": timing,
        "tiers": tiers_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(PRECISION_PRESETS),
        choices=list(PRECISION_PRESETS),
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=MEDIUM_RANGE_LEAD_HOURS,
        help="Medium-range lead time. Step count is horizon / timestep (6 h -> 40).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override step count for every preset. Default is --horizon-hours / dt.",
    )
    parser.add_argument(
        "--era5-steps",
        type=int,
        default=None,
        help="Override step count for era5_pretrained only.",
    )
    parser.add_argument(
        "--cams-steps",
        type=int,
        default=None,
        help="Override step count for cams only.",
    )
    parser.add_argument(
        "--ensemble-steps",
        type=int,
        default=None,
        help="Override step count for aurora_v1p5_ensemble only.",
    )
    parser.add_argument("--tiers", nargs="+", default=list(_DEFAULT_TIERS))
    parser.add_argument(
        "--isolate-tiers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each precision tier in a fresh subprocess (default: on).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Directory for isolated-worker step files (default: /dev/shm).",
    )
    parser.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Keep already-measured presets from this JSON and only run the rest.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path(_REPO) / "benchmark" / "rollout_drift_latest.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(_REPO) / "benchmark" / "rollout_drift_latest.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path(_REPO) / "docs" / "image" / "rollout_ar_drift.png",
    )
    args = parser.parse_args()

    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for rollout drift")
    gpu = torch.cuda.get_device_name(device)
    for _label, precision in resolve_tier_specs(args.tiers):
        require_perceiver_fp32(precision)
    tier_specs = resolve_tier_specs(args.tiers)

    scratch_root = args.scratch_dir
    if scratch_root is None:
        shm = Path("/dev/shm")
        scratch_root = shm if shm.is_dir() else Path(tempfile.gettempdir())
    scratch_root = scratch_root / f"flash_aurora_cl_{os.getpid()}"
    scratch_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "torch": torch.__version__,
        "asset_root": str(asset_root),
        "seed": _BENCHMARK_SEED,
        "horizon_hours": args.horizon_hours,
        "baseline": _PYTORCH_BASELINE_KEY,
        "presets": {},
    }
    if args.merge_json is not None and args.merge_json.is_file():
        prior = json.loads(args.merge_json.read_text(encoding="utf-8"))
        payload["presets"].update(prior.get("presets", {}))
        print(f"[merge] loaded {len(payload['presets'])} presets from {args.merge_json}", flush=True)

    def _n_steps(preset: str, timestep_hours: float) -> int:
        if preset == "era5_pretrained" and args.era5_steps is not None:
            return args.era5_steps
        if preset == "cams" and args.cams_steps is not None:
            return args.cams_steps
        if preset == "aurora_v1p5_ensemble" and args.ensemble_steps is not None:
            return args.ensemble_steps
        if args.steps is not None:
            return args.steps
        return steps_for_horizon(args.horizon_hours, timestep_hours)

    try:
        for preset in args.presets:
            if preset in payload["presets"]:
                print(f"[skip] {preset} already in merge JSON", flush=True)
                continue
            _, config = load_preset_batch(preset, asset_root)
            n_steps = _n_steps(preset, float(config.variant.timestep_hours))
            payload["presets"][preset] = run_preset(
                preset=preset,
                asset_root=asset_root,
                steps=n_steps,
                tier_specs=tier_specs,
                device=device,
                baseline=_PYTORCH_BASELINE_KEY,
                isolate_tiers=args.isolate_tiers,
                scratch_root=scratch_root,
            )
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            write_markdown(args.report_out, payload=payload)
            print(f"[checkpoint] {preset} -> {args.json_out}", flush=True)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.report_out, payload=payload)
    plot_drift(payload, args.plot_out)
    print(f"\nwrote {args.report_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.plot_out}")


if __name__ == "__main__":
    main()
