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
import subprocess
import sys
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
# One fixed channel per preset so the curve is not the max over variables.
_PRESET_PLOT_VARIABLE: dict[str, str] = {
    "era5_pretrained": "10v",
    "small_pretrained": "10v",
    "hres_t0_finetuned": "10v",
    "tc_tracking": "10v",
    "hres_0.1": "10v",
    "aurora_v1p5": "scaled_sf_1h",
    "aurora_v1p5_ensemble": "scaled_sf_1h",
    "cams": "pm10",
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


def data_disk_tmp(anchor: Path) -> Path:
    """Scratch directory on the data disk that holds ``anchor``, not the system disk."""
    path = anchor.expanduser().resolve().parent / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pin_temp_to_data_disk(anchor: Path) -> Path:
    """Point temp and matplotlib caches at the data disk."""
    path = data_disk_tmp(anchor)
    for key in ("TMPDIR", "TEMP", "TMP", "MPLCONFIGDIR"):
        os.environ[key] = str(path)
    return path


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


def field_of(pred: Any, variable: str) -> torch.Tensor:
    """Copy one predicted variable to CPU and drop the rest of the field."""
    for group in ("surf_vars", "atmos_vars"):
        variables = getattr(pred, group)
        if variable in variables:
            return variables[variable].detach().to(dtype=torch.float32, device="cpu").contiguous()
    raise KeyError(variable)


def compare_variable(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    name: str,
    tolerance: float,
    step: int,
    hours: float,
) -> dict[str, Any]:
    rel = mean_rel(reference, candidate)
    ok = rel <= tolerance
    return {
        "step": step,
        "lead_hours": step * hours,
        "vars": [{"name": name, "mean_rel": rel, "tol": tolerance, "ok": ok}],
        "n_fail": 0 if ok else 1,
        "n_vars": 1,
        "worst_name": name,
        "worst_rel": rel,
    }


def rollout_variable(
    model: Any,
    batch: Any,
    *,
    steps: int,
    device: torch.device,
    on_field: Any,
    variable: str,
) -> tuple[float, float]:
    """Roll out ``steps`` predictions and hand one CPU variable per step to ``on_field``.

    The rollout runs through the engine's ``RolloutSession`` with
    ``cache_in_dram=False``, the same switch as ``EngineConfig.cache_in_dram``,
    so no full-field trajectory is retained. Returns ``(peak_gib, forecast_s)``.
    """
    from flash_aurora.engine.core.rollout_session import RolloutSession

    set_benchmark_seed()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stream = RolloutSession(model, cache_in_dram=False).run(batch, steps)
    t0 = time.perf_counter()
    for step_index, pred in enumerate(stream, start=1):
        on_field(step_index, field_of(pred, variable))
        del pred
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _synchronize(device)
    forecast_s = time.perf_counter() - t0
    peak_gib = 0.0
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    return peak_gib, forecast_s


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


def representative_variable(preset: str) -> str:
    return _PRESET_PLOT_VARIABLE.get(preset, "10v")


def variable_drift_series(
    series: list[dict[str, Any]],
    variable: str,
) -> tuple[list[float], list[float], float | None]:
    """Lead time, mean relative error, and tolerance for one named variable."""
    lead_hours: list[float] = []
    mean_rel: list[float] = []
    tolerance: float | None = None
    for row in series:
        match = next((item for item in row["vars"] if item["name"] == variable), None)
        if match is None:
            continue
        lead_hours.append(float(row["lead_hours"]))
        mean_rel.append(float(match["mean_rel"]))
        tolerance = float(match["tol"])
    return lead_hours, mean_rel, tolerance


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
        variable = representative_variable(preset)
        tolerance: float | None = None
        for tier, series in block["tiers"].items():
            if tier == payload["baseline"] or not series:
                continue
            xs, ys, tier_tol = variable_drift_series(series, variable)
            if tier_tol is not None:
                tolerance = tier_tol
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=3.5,
                color=colors.get(tier, "#333333"),
                label=labels.get(tier, tier),
            )
        if tolerance is not None and tolerance > 0:
            ax.axhline(
                tolerance,
                color="#616161",
                linestyle=(0, (4, 2)),
                linewidth=1.0,
                zorder=0,
                label=rf"$\tau$ ({variable})",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(r"$\bar{e}_v$ vs FP32 AR ref")
        title = _PRESET_TITLES.get(preset, preset)
        ax.set_title(f"{title}, {variable}")
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
    variable: str,
    tolerance: float,
    hours: float,
    reference_fields: list[torch.Tensor] | None,
) -> dict[str, Any]:
    """One model per process. The parent keeps only ``variable`` from the reference tier."""
    import multiprocessing as mp

    require_perceiver_fp32(precision)
    from _closedloop_tier_worker import stream_tier

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=stream_tier,
        args=(queue, preset, precision, steps, str(asset_root), variable),
    )
    process.start()
    fields: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    try:
        while True:
            kind, payload = queue.get()
            if kind == "field":
                tensor = payload
                if reference_fields is None:
                    fields.append(tensor)
                else:
                    rows.append(
                        compare_variable(
                            reference_fields[len(rows)],
                            tensor,
                            name=variable,
                            tolerance=tolerance,
                            step=len(rows) + 1,
                            hours=hours,
                        )
                    )
                    del tensor
            elif kind == "done":
                meta = payload
                break
            elif kind == "error":
                raise RuntimeError(payload)
            else:
                raise RuntimeError(f"unknown worker event {kind!r}")
    finally:
        process.join()
        queue.close()
    if process.exitcode not in (0, None):
        raise RuntimeError(
            f"isolated closed-loop failed preset={preset!r} precision={precision!r} "
            f"rc={process.exitcode}"
        )
    expected = fields if reference_fields is None else rows
    if len(expected) != steps:
        raise RuntimeError(
            f"isolated closed-loop returned {len(expected)} steps, expected {steps} "
            f"(preset={preset!r} precision={precision!r})"
        )
    return {
        "fields": fields,
        "rows": rows,
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
) -> dict[str, Any]:
    batch, config = load_preset_batch(preset, asset_root)
    ckpt = checkpoint_path(config, asset_root)
    var_specs = output_var_tolerances(config)
    hours = float(config.variant.timestep_hours)
    variable = representative_variable(preset)
    tolerance = next((tol for _group, name, tol in var_specs if name == variable), 5e-3)
    print(
        f"\n=== {preset}  steps={steps}  dt={hours:g}h  ckpt={ckpt.name}  "
        f"variable={variable} ===",
        flush=True,
    )

    reference_fields: list[torch.Tensor] = []
    peak_gib: dict[str, float] = {}
    timing: dict[str, dict[str, Any]] = {}
    tiers_out: dict[str, Any] = {baseline: []}
    for label, precision in tier_specs:
        require_perceiver_fp32(precision)
        if label != baseline and len(reference_fields) != steps:
            print(f"  [skip] {label}: no baseline trajectory to compare against", flush=True)
            timing[label] = {"ok": False, "error": "baseline trajectory unavailable"}
            peak_gib[label] = 0.0
            tiers_out[label] = []
            continue
        print(f"  [load] {label} ({precision})", flush=True)
        try:
            if isolate_tiers:
                result = run_tier_isolated(
                    preset=preset,
                    precision=precision,
                    steps=steps,
                    asset_root=asset_root,
                    variable=variable,
                    tolerance=tolerance,
                    hours=hours,
                    reference_fields=None if label == baseline else reference_fields,
                )
                if label == baseline:
                    reference_fields = result["fields"]
                rows = result["rows"]
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
                collected: list[torch.Tensor] = []
                compared: list[dict[str, Any]] = []

                def _keep_reference(_index: int, tensor: torch.Tensor) -> None:
                    collected.append(tensor)

                def _compare_candidate(index: int, tensor: torch.Tensor) -> None:
                    compared.append(
                        compare_variable(
                            reference_fields[index - 1],
                            tensor,
                            name=variable,
                            tolerance=tolerance,
                            step=index,
                            hours=hours,
                        )
                    )
                    del tensor

                t_load = time.perf_counter()
                model = build_model(config, ckpt, precision=precision, device=device)
                _synchronize(device)
                load_s = time.perf_counter() - t_load
                peak, forecast_s = rollout_variable(
                    model,
                    batch,
                    steps=steps,
                    device=device,
                    on_field=_keep_reference if label == baseline else _compare_candidate,
                    variable=variable,
                )
                del model
                purge_gpu()
                gc.collect()
                if label == baseline:
                    reference_fields = collected
                rows = compared
                per_step_s = forecast_s / steps if steps else 0.0
                peak_gib[label] = peak
                timing[label] = {
                    "ok": True,
                    "load_s": load_s,
                    "forecast_s": forecast_s,
                    "per_step_s": per_step_s,
                    "peak_gib": peak,
                }
            if label != baseline:
                tiers_out[label] = rows
                for stats in rows:
                    print(
                        f"  [{label}] step {stats['step']:02d}  "
                        f"fail {stats['n_fail']}/{stats['n_vars']}  "
                        f"worst {stats['worst_name']}={stats['worst_rel']:.3e}",
                        flush=True,
                    )
            print(
                f"  [done] {label}  forecast={timing[label]['forecast_s']:.1f}s  "
                f"peak={peak_gib[label]:.1f} GiB",
                flush=True,
            )
        except Exception as exc:
            print(f"  [fail] {label}: {type(exc).__name__}: {exc}", flush=True)
            timing[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            peak_gib[label] = 0.0
            if label != baseline:
                tiers_out[label] = []
            purge_gpu()
            gc.collect()
    if not timing.get(baseline, {}).get("ok"):
        print(f"  [skip-compare] baseline {baseline} produced no trajectory", flush=True)
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
    pin_temp_to_data_disk(asset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for rollout drift")
    gpu = torch.cuda.get_device_name(device)
    for _label, precision in resolve_tier_specs(args.tiers):
        require_perceiver_fp32(precision)
    tier_specs = resolve_tier_specs(args.tiers)

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
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_markdown(args.report_out, payload=payload)
        print(f"[checkpoint] {preset} -> {args.json_out}", flush=True)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.report_out, payload=payload)
    plot_drift(payload, args.plot_out)
    print(f"\nwrote {args.report_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.plot_out}")


if __name__ == "__main__":
    main()
