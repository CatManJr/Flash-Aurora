#!/usr/bin/env python3
"""In-depth benchmarks: std, kernel ablation, compiler/graph,
stage split, VRAM, mean relative error, and a second ERA5 initial condition.

    export AURORA_ASSET_ROOT=/path/to/data/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_indepth_eval.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
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
from _preset_ic import checkpoint_path, load_preset_batch, output_var_tolerances  # noqa: E402
from _pretrained_era5 import prediction_tensors, purge_gpu  # noqa: E402
from _stage_timing import time_forward_stages  # noqa: E402

import torch

SEED = 42
BASELINE = "fp32"
PHYS_VARS = (("surf_vars", "2t", "K"), ("surf_vars", "msl", "Pa"), ("surf_vars", "10u", "m/s"), ("surf_vars", "10v", "m/s"))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def build_model(
    config,
    ckpt: Path,
    *,
    precision: str,
    device: torch.device,
    compile_backbone: bool = False,
    disable_cute: bool = False,
):
    from flash_aurora.engine.core.model_registry import ModelFactory

    set_seed()
    variant = config.variant
    kwargs: dict[str, Any] = {"inference_precision": precision}
    if compile_backbone:
        kwargs["compile_backbone"] = True
        kwargs["compile_backbone_dynamic"] = False
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
    model = model.to(device)
    if disable_cute:
        for mod in model.modules():
            if hasattr(mod, "use_cute_window_attn"):
                mod.use_cute_window_attn = False
    return model


def time_samples(model, batch, *, warmup: int, repeat: int, device: torch.device) -> tuple[list[float], float]:
    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout

    def _fwd():
        if model_uses_v1p5_rollout(model):
            example = next(iter(batch.surf_vars.values()))
            lead_hours = model.timestep.total_seconds() / 3600.0
            lead = torch.full((example.shape[0],), lead_hours, device=device, dtype=example.dtype)
            return model.forward(batch, lead_times=lead)
        return model.forward(batch)

    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _fwd()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _fwd()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(float(start.elapsed_time(end)))
    peak = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    return samples, peak


def mean_rel(ref: torch.Tensor, cand: torch.Tensor) -> float:
    err = (cand - ref).abs()
    return float(err.mean().item() / ref.abs().mean().clamp_min(1e-8).item())


def mae(ref: torch.Tensor, cand: torch.Tensor) -> float:
    return float((cand - ref).abs().mean().item())


def rmse(ref: torch.Tensor, cand: torch.Tensor) -> float:
    return float(torch.sqrt(((cand - ref) ** 2).mean()).item())


def run_latency(config, ckpt, batch, device, warmup: int, repeat: int) -> dict[str, Any]:
    cases = [
        ("unfused_fp32", "fp32", False, False, False),
        ("fast_fp32_triton_sdpa", "fast_fp32", False, False, False),
        ("fp32@fp32", "fp32@fp32", False, False, False),
        ("tf32@fp32", "tf32@fp32", False, False, False),
        ("bf16_mixed@fp32", "bf16_mixed@fp32", False, False, False),
        ("autocast", "pytorch_autocast", False, False, False),
        ("bf16_mixed_sdpa", "bf16_mixed@fp32", False, True, False),
        ("bf16_mixed_compile", "bf16_mixed@fp32", True, False, False),
        ("bf16_mixed_cudagraph", "bf16_mixed@fp32", False, False, True),
    ]
    out: dict[str, Any] = {}
    for name, precision, compile_bb, no_cute, use_graph in cases:
        print(f"  [latency] {name}", flush=True)
        purge_gpu()
        model = None
        try:
            model = build_model(
                config,
                ckpt,
                precision=precision,
                device=device,
                compile_backbone=compile_bb,
                disable_cute=no_cute,
            )
            dev_batch = batch.to(device)
            extra_warmup = warmup + (8 if compile_bb or use_graph else 0)
            if use_graph:
                with torch.inference_mode():
                    for _ in range(2):
                        model.forward(dev_batch)
                model.capture_inference_cuda_graph(dev_batch, scope="backbone", warmup_iters=2)
            samples, peak = time_samples(
                model, dev_batch, warmup=extra_warmup, repeat=repeat, device=device
            )
            row = {"ok": True, "precision": precision, **stats(samples), "peak_gib": peak}
            print(
                f"    mean={row['mean']:.1f} std={row['std']:.2f} p95={row['p95']:.1f} peak={peak:.1f} GiB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row = {"ok": False, "precision": precision, "error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAIL {row['error']}", flush=True)
        out[name] = row
        del model
        purge_gpu()
        gc.collect()
    return out


def run_stages(config, ckpt, batch, device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, precision in (("unfused_fp32", "fp32"), ("bf16_mixed@fp32", "bf16_mixed@fp32")):
        print(f"  [stages] {name}", flush=True)
        purge_gpu()
        model = build_model(config, ckpt, precision=precision, device=device)
        timing, _pred = time_forward_stages(
            model, batch.to(device), warmup=2, repeat=5, device=device
        )
        out[name] = {
            "encoder_ms": timing.encoder_ms,
            "backbone_ms": timing.backbone_ms,
            "decoder_ms": timing.decoder_ms,
            "post_ms": timing.post_ms,
            "total_ms": timing.total_ms,
            "backbone_pct": timing.backbone_pct,
        }
        print(
            f"    enc={timing.encoder_ms:.1f} bb={timing.backbone_ms:.1f} "
            f"({timing.backbone_pct:.1f}%) dec={timing.decoder_ms:.1f} tot={timing.total_ms:.1f}",
            flush=True,
        )
        del model
        purge_gpu()
    return out


def run_vram(asset_root: Path, device: torch.device, presets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for preset in presets:
        print(f"  [vram] {preset}", flush=True)
        batch, config = load_preset_batch(preset, asset_root)
        ckpt = checkpoint_path(config, asset_root)
        purge_gpu()
        model = build_model(config, ckpt, precision="bf16_mixed@fp32", device=device)
        _samples, peak = time_samples(model, batch.to(device), warmup=1, repeat=1, device=device)
        out[preset] = {
            "peak_gib": peak,
            "grid": list(config.variant.resolution),
            "lora": bool(config.variant.use_lora),
        }
        print(f"    peak={peak:.1f} GiB", flush=True)
        del model
        purge_gpu()
        gc.collect()
    return out


def compare_phys(ref: dict[str, torch.Tensor], cand: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows = {}
    for group, name, unit in PHYS_VARS:
        key = f"{group}.{name}"
        if key not in ref or key not in cand:
            continue
        rows[name] = {
            "unit": unit,
            "mae": mae(ref[key], cand[key]),
            "rmse": rmse(ref[key], cand[key]),
            "mean_rel": mean_rel(ref[key], cand[key]),
        }
    return rows


def one_step_accuracy(config, ckpt, batch, device, tiers: list[str]) -> dict[str, Any]:
    from flash_aurora.models.aurora.rollout import prepare_rollout_batch

    specs = output_var_tolerances(config)
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for precision in tiers:
        print(f"  [acc] {precision}", flush=True)
        purge_gpu()
        model = build_model(config, ckpt, precision=precision, device=device)
        prepared = prepare_rollout_batch(model, batch)
        with torch.inference_mode():
            pred = model.forward(prepared)
        tensors[precision] = prediction_tensors(pred)
        del model, pred
        purge_gpu()
    ref = tensors[BASELINE]
    out: dict[str, Any] = {}
    for precision, cand in tensors.items():
        if precision == BASELINE:
            continue
        fails = []
        for group, name, tol in specs:
            key = f"{group}.{name}"
            rel = mean_rel(ref[key], cand[key])
            if rel > tol:
                fails.append({"name": name, "mean_rel": rel, "tol": tol})
        out[precision] = {
            "n_fail": len(fails),
            "n_vars": len(specs),
            "fails": fails,
            "phys": compare_phys(ref, cand),
        }
        print(f"    fail {len(fails)}/{len(specs)}  2t={out[precision]['phys'].get('2t', {}).get('mean_rel')}", flush=True)
    return out


def ar_phys(config, ckpt, batch, device, *, steps: int, tiers: list[str]) -> dict[str, Any]:
    from flash_aurora.models.aurora.rollout import rollout

    traj: dict[str, list[dict[str, torch.Tensor]]] = {}
    hours = float(config.variant.timestep_hours)
    for precision in tiers:
        print(f"  [ar] {precision} steps={steps}", flush=True)
        purge_gpu()
        model = build_model(config, ckpt, precision=precision, device=device)
        preds: list[dict[str, torch.Tensor]] = []
        with torch.inference_mode():
            for pred in rollout(model, batch, steps):
                preds.append(prediction_tensors(pred))
                del pred
                torch.cuda.empty_cache()
        traj[precision] = preds
        del model
        purge_gpu()
        gc.collect()
    ref_traj = traj[BASELINE]
    out: dict[str, Any] = {}
    for precision, series in traj.items():
        if precision == BASELINE:
            continue
        rows = []
        for step, (ref, cand) in enumerate(zip(ref_traj, series), start=1):
            rows.append(
                {
                    "step": step,
                    "lead_hours": step * hours,
                    "phys": compare_phys(ref, cand),
                }
            )
        out[precision] = rows
        last = rows[-1]["phys"]
        print(
            f"    last 2t={last['2t']['mean_rel']:.3e}  "
            f"10u={last['10u']['mean_rel']:.3e}  "
            f"msl={last['msl']['mean_rel']:.3e}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--ar-steps", type=int, default=20)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(_REPO) / "benchmark" / "indepth_eval_latest.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(_REPO) / "benchmark" / "indepth_eval_latest.md",
    )
    args = parser.parse_args()

    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(device)
    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "torch": torch.__version__,
        "seed": SEED,
        "warmup": args.warmup,
        "repeat": args.repeat,
    }

    print("=== era5_pretrained primary IC 2023-01-01 ===", flush=True)
    batch, config = load_preset_batch("era5_pretrained", asset_root)
    ckpt = checkpoint_path(config, asset_root)
    payload["latency_era5"] = run_latency(config, ckpt, batch, device, args.warmup, args.repeat)
    payload["stages_era5"] = run_stages(config, ckpt, batch, device)
    payload["one_step_2023"] = one_step_accuracy(
        config, ckpt, batch, device,         ["fp32", "bf16_mixed@fp32", "tf32@fp32", "pytorch_autocast"]
    )
    payload["ar_phys_2023"] = ar_phys(
        config,
        ckpt,
        batch,
        device,
        steps=args.ar_steps,
        tiers=["fp32", "bf16_mixed@fp32", "tf32@fp32", "pytorch_autocast"],
    )

    print("=== era5_pretrained second IC 2026-07-01 ===", flush=True)
    batch2, config2 = load_preset_batch(
        "era5_pretrained", asset_root, valid_time=datetime(2026, 7, 1, 6)
    )
    payload["one_step_2026"] = one_step_accuracy(
        config2, ckpt, batch2, device, ["fp32", "bf16_mixed@fp32", "tf32@fp32"]
    )
    payload["ar_phys_2026"] = ar_phys(
        config2,
        ckpt,
        batch2,
        device,
        steps=args.ar_steps,
        tiers=["fp32", "bf16_mixed@fp32", "tf32@fp32"],
    )

    print("=== VRAM bf16_mixed@fp32 ===", flush=True)
    payload["vram"] = run_vram(
        asset_root,
        device,
        ["era5_pretrained", "aurora_v1p5", "hres_t0_finetuned", "hres_0.1", "cams"],
    )

    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


def _fmt_lat(row: dict[str, Any]) -> str:
    if not row.get("ok"):
        return row.get("error", "FAIL")
    return f"{row['mean']:.1f} ± {row['std']:.1f} (p95 {row['p95']:.1f}, n={row['n']})"


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# In-depth benchmarks",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Seed: {payload['seed']}; warmup {payload['warmup']}, per-iter CUDA events n={payload['repeat']}",
        "",
        "## One-step latency with variance (era5_pretrained)",
        "",
        "| config | mean±std (ms) | peak GiB |",
        "| --- | --- | ---: |",
    ]
    for name, row in payload["latency_era5"].items():
        peak = f"{row['peak_gib']:.1f}" if row.get("ok") else "—"
        lines.append(f"| `{name}` | {_fmt_lat(row)} | {peak} |")
    lines += ["", "## Encoder / backbone / decoder (era5_pretrained)", ""]
    lines.append("| config | encoder | backbone | decoder | total | backbone % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in payload["stages_era5"].items():
        lines.append(
            f"| `{name}` | {row['encoder_ms']:.1f} | {row['backbone_ms']:.1f} | "
            f"{row['decoder_ms']:.1f} | {row['total_ms']:.1f} | {row['backbone_pct']:.1f} |"
        )
    lines += ["", "## One-step mean relative error vs unfused FP32", ""]
    for ic, key in (("2023-01-01", "one_step_2023"), ("2026-07-01", "one_step_2026")):
        lines.append(f"### {ic}")
        lines.append("")
        lines.append("| tier | fail | 2t | msl | 10u |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for tier, row in payload[key].items():
            phys = row["phys"]
            lines.append(
                f"| `{tier}` | {row['n_fail']}/{row['n_vars']} | "
                f"{phys['2t']['mean_rel']:.3e} | {phys['msl']['mean_rel']:.3e} | "
                f"{phys['10u']['mean_rel']:.3e} |"
            )
        lines.append("")
    lines += ["", "## AR mean relative error at last step", ""]
    for ic, key in (("2023-01-01", "ar_phys_2023"), ("2026-07-01", "ar_phys_2026")):
        lines.append(f"### {ic}")
        lines.append("")
        for tier, series in payload[key].items():
            last = series[-1]
            p = last["phys"]
            lines.append(
                f"- `{tier}` lead {last['lead_hours']:g} h: "
                f"2t {p['2t']['mean_rel']:.3e}, msl {p['msl']['mean_rel']:.3e}, "
                f"10u {p['10u']['mean_rel']:.3e}, 10v {p['10v']['mean_rel']:.3e}"
            )
        lines.append("")
    lines += ["", "## Peak allocated VRAM (`bf16_mixed@fp32`, one forward)", ""]
    lines.append("| preset | grid | peak GiB |")
    lines.append("| --- | --- | ---: |")
    for preset, row in payload["vram"].items():
        g = "x".join(str(x) for x in row["grid"])
        lines.append(f"| `{preset}` | {g} | {row['peak_gib']:.1f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
