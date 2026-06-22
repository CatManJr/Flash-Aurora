#!/usr/bin/env python3
"""Simple precision benchmark for AuroraSmallPretrained on HF test batch.

Loads ``aurora-0.25-small-pretrained-test-input.pickle`` + ``aurora-0.25-static.pickle``,
runs each inference tier, times end-to-end ``forward``, and compares outputs against the
PyTorch FP32 baseline. Prints per-variable **official tolerances** from ``tests/test_model.py``.

**Default suite (10 tiers)** - no preset shorthand required in output:

1. PyTorch: ``backbone=fp32, encoder/decoder=fp32`` (baseline for accuracy tables)
2. PyTorch: ``backbone=autocast_bf16, encoder/decoder=fp32``
3. Eight explicit custom combos ``{fp32,tf32,bf16_mixed,bf16}@{fp32,tf32}`` (Triton/CuTe Swin)

Examples::

    uv run python benchmark/bench_small_pretrained.py
    uv run python benchmark/bench_small_pretrained.py --compare-hf
    uv run python benchmark/bench_small_pretrained.py --suite legacy
    uv run python benchmark/bench_small_pretrained.py --combo-matrix \\
        --backbone-levels fp32 tf32 bf16_mixed bf16 --encoder-decoder-levels fp32 tf32
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402
from _asset_root import default_asset_root


import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]

from _asset_root import default_asset_root

_DEFAULT_DATA_DIR = str(default_asset_root())
_CHECKPOINT_NAME = "aurora-0.25-small-pretrained.ckpt"
_INPUT_NAME = "aurora-0.25-small-pretrained-test-input.pickle"
_STATIC_NAME = "aurora-0.25-static.pickle"
_OUTPUT_NAME = "aurora-0.25-small-pretrained-test-output.pickle"

_PYTORCH_BASELINE_KEY = "pytorch_backbone_fp32_encoder_decoder_fp32"

_LEGACY_NAMED_TIERS: tuple[tuple[str, str, str], ...] = (
    ("fp32", "fp32", "PyTorch FP32"),
    ("fast_fp32", "fast_fp32", "Triton layout + AdaLN + PyTorch GELU"),
    ("tf32", "tf32", "fast_fp32 + TF32 matmul + CuTe TF32 attn"),
    ("bf16_mixed", "bf16_mixed", "hybrid: BF16 attention QKV/proj + BF16 MLP"),
    ("bf16", "bf16", "full backbone BF16 linears + CuTe BF16 attn"),
    ("pytorch_autocast", "pytorch_autocast", "PyTorch backbone BF16 autocast"),
)


def _pytorch_reference_tiers() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            _PYTORCH_BASELINE_KEY,
            "fp32",
            "PyTorch: backbone matmul FP32, encoder/decoder matmul FP32, "
            "no torch.autocast, no Triton/CuTe",
        ),
        (
            "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
            "pytorch_autocast",
            "PyTorch: backbone torch.autocast BF16, encoder/decoder matmul FP32, "
            "no Triton/CuTe",
        ),
    )


def _custom_matmul_combo_tiers(
    *,
    backbone_levels: tuple[str, ...],
    encoder_decoder_levels: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    from flash_aurora.aurora.model.inference_precision import (
        DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS,
        DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS,
        describe_inference_config,
        expand_precision_combos,
    )

    bb = backbone_levels or DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS
    ed = encoder_decoder_levels or DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS
    tiers: list[tuple[str, str, str]] = []
    for label, cfg in expand_precision_combos(bb, ed):
        spec = f"{cfg.backbone_matmul_level.value}@{cfg.encoder_decoder_matmul_level.value}"
        tiers.append((label, spec, describe_inference_config(cfg)))
    return tuple(tiers)


def _full_default_suite(
    *,
    backbone_levels: tuple[str, ...] = (),
    encoder_decoder_levels: tuple[str, ...] = (),
) -> tuple[tuple[str, str, str], ...]:
    from flash_aurora.aurora.model.inference_precision import (
        DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS,
        DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS,
    )

    bb = backbone_levels or DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS
    ed = encoder_decoder_levels or DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS
    return _pytorch_reference_tiers() + _custom_matmul_combo_tiers(
        backbone_levels=bb,
        encoder_decoder_levels=ed,
    )


def _tier_entry(name: str) -> tuple[str, str, str]:
    """Resolve a legacy preset name or ``backbone@encoder_decoder`` combo string."""
    for key, precision, label in _LEGACY_NAMED_TIERS:
        if name == key:
            return key, precision, label
    from flash_aurora.aurora.model.inference_precision import describe_inference_config, resolve_inference_config

    cfg = resolve_inference_config(name)
    if cfg is None:
        raise ValueError(f"Could not resolve inference tier {name!r}.")
    spec = f"{cfg.backbone_matmul_level.value}@{cfg.encoder_decoder_matmul_level.value}"
    if name == spec or "@" in name:
        return cfg.config_label, spec, describe_inference_config(cfg)
    return cfg.config_label, name, describe_inference_config(cfg)


def _tiers_from_args(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    if args.tiers is not None:
        return tuple(_tier_entry(t) for t in args.tiers)

    bb = tuple(args.backbone_levels)
    ed = tuple(args.encoder_decoder_levels)

    if args.suite == "legacy":
        return _LEGACY_NAMED_TIERS
    if args.suite == "combos":
        return _custom_matmul_combo_tiers(backbone_levels=bb, encoder_decoder_levels=ed)
    if args.combo_matrix:
        if args.combos_only:
            return _custom_matmul_combo_tiers(backbone_levels=bb, encoder_decoder_levels=ed)
        return _full_default_suite(backbone_levels=bb, encoder_decoder_levels=ed)
    return _full_default_suite()


# Same relative mean error gates as aurora/tests/test_model.py::test_aurora_small
_OFFICIAL_TOLERANCES: dict[str, float] = {
    "2t": 1e-4,
    "10u": 5e-3,
    "10v": 5e-3,
    "msl": 1e-4,
    "u": 5e-3,
    "v": 5e-3,
    "t": 1e-4,
    "q": 5e-3,
}

_OFFICIAL_VAR_ORDER: tuple[tuple[str, str], ...] = (
    ("surf_vars", "2t"),
    ("surf_vars", "10u"),
    ("surf_vars", "10v"),
    ("surf_vars", "msl"),
    ("atmos_vars", "u"),
    ("atmos_vars", "v"),
    ("atmos_vars", "t"),
    ("atmos_vars", "q"),
)


def _purge_gpu(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _load_batch(data_dir: Path) -> Any:
    from flash_aurora.aurora import Batch, Metadata
    from flash_aurora.aurora.batch import interpolate_numpy

    input_path = data_dir / _INPUT_NAME
    static_path = data_dir / _STATIC_NAME
    if not input_path.is_file():
        raise FileNotFoundError(f"missing test input: {input_path}")
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static vars: {static_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(input_path, "rb") as f:
            test_input = pickle.load(f)
        with open(static_path, "rb") as f:
            static_vars = pickle.load(f)

    if os.name == "nt":

        class PatchedDateTime(datetime):
            def timestamp(self) -> float:
                return -631134000.0

        test_input["metadata"]["time"] = [PatchedDateTime(1950, 1, 1, 6, 0)]

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    return Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )


def _load_hf_output_tensors(data_dir: Path) -> dict[str, torch.Tensor] | None:
    output_path = data_dir / _OUTPUT_NAME
    if not output_path.is_file():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(output_path, "rb") as f:
            test_output = pickle.load(f)
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, arr in test_output[group].items():
            out[f"{group}.{name}"] = torch.from_numpy(np.asarray(arr)).float()
    return out


def _prediction_tensors(pred: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, tensor in getattr(pred, group).items():
            out[f"{group}.{name}"] = tensor.detach().float().cpu()
    return out


def _diff_vs_reference(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, float, float, float]:
    max_diff = 0.0
    max_rel = 0.0
    total = 0.0
    cos_total = 0.0
    count = 0
    for key, ref in reference.items():
        cand = candidate[key]
        diff = (cand - ref).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        denom = ref.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        total += float(diff.mean().item())

        ref_flat = ref.flatten().double()
        cand_flat = cand.flatten().double()
        ref_norm = ref_flat.norm()
        cand_norm = cand_flat.norm()
        if ref_norm.item() == 0.0 and cand_norm.item() == 0.0:
            cos = 1.0
        elif ref_norm.item() == 0.0 or cand_norm.item() == 0.0:
            cos = 0.0
        else:
            cos = float(torch.dot(ref_flat, cand_flat).item() / (ref_norm.item() * cand_norm.item()))
        cos_total += cos
        count += 1
    return max_diff, total / max(count, 1), max_rel, cos_total / max(count, 1)


def _official_tol_rows(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> list[tuple[str, float, float, float, bool]]:
    """Per-variable mean(|err|)/mean(|ref|) vs official tol (test_model.py)."""
    rows: list[tuple[str, float, float, float, bool]] = []
    for group, name in _OFFICIAL_VAR_ORDER:
        key = f"{group}.{name}"
        ref = reference[key]
        cand = candidate[key]
        err = (cand - ref).abs()
        mean_rel = float(err.mean().item() / ref.abs().mean().clamp_min(1e-8).item())
        tol = _OFFICIAL_TOLERANCES[name]
        max_abs = float(err.max().item())
        rows.append((name, mean_rel, tol, max_abs, mean_rel <= tol))
    return rows


def _print_official_tol_table(title: str, rows: list[tuple[str, float, float, float, bool]]) -> None:
    print(f"\n{title}")
    print("  metric: mean(|out-ref|) / mean(|ref|)  (aurora/tests/test_model.py)")
    print(f"  {'var':<6} {'mean_rel':>10} {'tol':>10} {'max_abs':>10} {'ok':>4}")
    print("  " + "-" * 44)
    for name, mean_rel, tol, max_abs, ok in rows:
        mark = "yes" if ok else "NO"
        print(f"  {name:<6} {mean_rel:10.4e} {tol:10.4e} {max_abs:10.4g} {mark:>4}")
    passed = sum(1 for r in rows if r[4])
    print(f"  summary: {passed}/{len(rows)} variables within official tolerance")


def _set_cute_window_attn(model: Any, enabled: bool) -> None:
    """Toggle CuTe window attention without changing matmul / Triton preset hooks."""
    backbone = model.backbone
    backbone.use_cute_window_attn = enabled
    for module in backbone.modules():
        if hasattr(module, "use_cute_window_attn"):
            module.use_cute_window_attn = enabled


def _build_model(
    precision: str,
    checkpoint: Path,
    device: torch.device,
    *,
    use_cute_window_attn: bool | None = None,
) -> Any:
    from flash_aurora.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision=precision)
    if use_cute_window_attn is not None:
        _set_cute_window_attn(model, use_cute_window_attn)
    model.load_checkpoint_local(str(checkpoint), strict=True)
    model.eval()
    return model.to(device)


def _time_forward(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[Any, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
        else:
            import time

            t0 = time.perf_counter()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            ms_total = (time.perf_counter() - t0) * 1e3

    return pred, ms_total / repeat


def _run_tier(
    *,
    precision: str,
    checkpoint: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
    use_cute_window_attn: bool | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    model = _build_model(
        precision,
        checkpoint,
        device,
        use_cute_window_attn=use_cute_window_attn,
    )
    try:
        pred, ms_per = _time_forward(model, batch, warmup=warmup, repeat=repeat, device=device)
        return _prediction_tensors(pred), ms_per
    finally:
        _purge_gpu(model)


def _run_ablate_cute(
    *,
    checkpoint: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> None:
    """Isolate pooled / per-var max_abs: CuTe DSL vs PyTorch SDPA (same Triton + matmul preset)."""
    cases: list[tuple[str, str, bool | None]] = [
        ("fp32", "fp32 baseline", None),
        ("pytorch_autocast", "PyTorch autocast + SDPA", None),
        ("tf32", "TF32 + Triton + CuTe attn", None),
        ("tf32", "TF32 + Triton + SDPA (no CuTe)", False),
        ("bf16_mixed", "BF16 MLP + Triton + CuTe attn", None),
        ("bf16_mixed", "BF16 MLP + Triton + SDPA (no CuTe)", False),
    ]
    baseline_key = "fp32"
    baseline_ms: float | None = None
    baseline: dict[str, torch.Tensor] | None = None
    all_preds: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[tuple[str, str, float, float, float, float, float, float | None]] = []

    print()
    print("=" * 60)
    print("CuTe ablation: same preset, toggle window attention only")
    print("=" * 60)

    for precision, label, use_cute in cases:
        tier_key = precision if use_cute is None else f"{precision}{'_cute' if use_cute else '_sdpa'}"
        display = label
        print(f"[run] {display}...", flush=True)
        pred, ms_per = _run_tier(
            precision=precision,
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=warmup,
            repeat=repeat,
            use_cute_window_attn=use_cute,
        )
        all_preds[tier_key] = pred
        print(f"[run] {display} e2e forward={ms_per:.1f} ms", flush=True)
        if tier_key == baseline_key:
            baseline = pred
            baseline_ms = ms_per
            rows.append((tier_key, display, ms_per, None, 0.0, 0.0, 0.0, 1.0))
            continue
        assert baseline is not None and baseline_ms is not None
        max_abs, mean_abs, max_rel, cos = _diff_vs_reference(baseline, pred)
        rows.append(
            (tier_key, display, ms_per, baseline_ms / ms_per, max_abs, mean_abs, max_rel, cos)
        )

    _print_table(rows)

    assert baseline is not None
    print()
    print("Per-variable max_abs vs fp32")
    print("-" * 60)
    print(f"  {'tier':<22} {'msl':>10} {'2t':>10} {'10u':>10} {'10v':>10} {'max_all':>10}")
    for tier_key in (
        "pytorch_autocast",
        "tf32",
        "tf32_sdpa",
        "bf16_mixed",
        "bf16_mixed_sdpa",
    ):
        if tier_key not in all_preds:
            continue
        tol_rows = _official_tol_rows(baseline, all_preds[tier_key])
        by_name = {name: max_abs for name, _mr, _tol, max_abs, _ok in tol_rows}
        pooled_max = max(by_name.values()) if by_name else float("nan")
        print(
            f"  {tier_key:<22} "
            f"{by_name.get('msl', float('nan')):10.4g} "
            f"{by_name.get('2t', float('nan')):10.4g} "
            f"{by_name.get('10u', float('nan')):10.4g} "
            f"{by_name.get('10v', float('nan')):10.4g} "
            f"{pooled_max:10.4g}"
        )

    print()
    print("Interpretation: if *_sdpa max_abs ~ autocast and *_cute ~ 93, outlier is CuTe DSL.")


def _print_table(
    rows: list[tuple[str, str, float, float, float, float, float, float | None]],
) -> None:
    """Print summary rows sorted by ``ms`` ascending (fastest tier first)."""
    rows_sorted = sorted(rows, key=lambda r: r[2])
    tier_w = max(36, max((len(r[0]) for r in rows_sorted), default=18))
    print(
        f"\n{'tier':<{tier_w}} {'ms':>8} {'speedup':>8} {'max_abs':>10} {'mean_abs':>10} "
        f"{'max_rel':>10} {'cos_sim':>8}"
    )
    print("-" * (tier_w + 70))
    for key, _label, ms, speedup, max_abs, mean_abs, max_rel, cos in rows_sorted:
        speedup_s = f"{speedup:.2f}x" if speedup is not None else "  base"
        print(
            f"{key:<{tier_w}} {ms:8.1f} {speedup_s:>8} {max_abs:10.4g} {mean_abs:10.4g} "
            f"{max_rel:10.4g} {cos:8.6f}"
        )
    print("\nTier details (full precision description):")
    for key, label, *_rest in rows_sorted:
        print(f"  {key}: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(_DEFAULT_DATA_DIR),
        help=f"Directory with HF pickles (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint path (default: <data-dir>/{_CHECKPOINT_NAME})",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        help=(
            "Explicit tiers: legacy preset names and/or combo strings (bf16_mixed@fp32). "
            "Overrides --suite."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("full", "legacy", "combos"),
        default="full",
        help=(
            "full (default): 2 PyTorch reference tiers + 8 custom backbone@encoder_decoder combos; "
            "legacy: named presets (fp32, tf32, bf16, …); "
            "combos: only the 4×2 custom matmul grid."
        ),
    )
    parser.add_argument(
        "--combo-matrix",
        action="store_true",
        help=(
            "Like --suite full but override backbone/encoder-decoder level lists for the "
            "8 custom combos (PyTorch tiers still included unless --combos-only)."
        ),
    )
    parser.add_argument(
        "--combos-only",
        action="store_true",
        help="With --combo-matrix or --suite combos: skip the two PyTorch reference tiers.",
    )
    parser.add_argument(
        "--backbone-levels",
        nargs="+",
        default=["fp32", "tf32", "bf16_mixed", "bf16"],
        help=(
            "Backbone matmul levels for custom combos "
            "(default: fp32 tf32 bf16_mixed bf16 → 4×2=8 combos)."
        ),
    )
    parser.add_argument(
        "--encoder-decoder-levels",
        nargs="+",
        default=["fp32", "tf32"],
        help="Encoder/decoder matmul levels for custom combos (default: fp32 tf32).",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help=(
            "Also print official tol for fp32 tier vs HF test-output.pickle. "
            "Informational only: HF gold was generated with Microsoft's reference "
            "settings (see aurora/tests/test_model.py: model.double(), use_lora=True, "
            "batch×2), not bench fp32 (use_lora=False, float32 CUDA). "
            "5/8 on 2t/10u/10v is typical FP32-vs-FP64 drift; use tier-vs-fp32 for preset QA."
        ),
    )
    parser.add_argument(
        "--no-official-tol",
        action="store_true",
        help="Skip per-variable official tolerance tables (default: print for each tier vs fp32)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before timing (default: 1)")
    parser.add_argument("--repeat", type=int, default=3, help="Timed forwards per tier (default: 3)")
    parser.add_argument(
        "--ablate-cute",
        action="store_true",
        help=(
            "Compare tf32 / bf16_mixed with CuTe window attn on vs off (SDPA); "
            "print per-variable max_abs vs fp32"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or data_dir / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    batch = _load_batch(data_dir).to(device)
    lat = batch.metadata.lat.numel()
    lon = batch.metadata.lon.numel()
    print(f"[config] AuroraSmallPretrained @ {lat}x{lon}, batch={next(iter(batch.surf_vars.values())).shape[0]}")
    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] data_dir={data_dir}")
    print(f"[config] timing warmup={args.warmup} repeat={args.repeat}")

    if args.ablate_cute:
        _run_ablate_cute(
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        return

    if args.tiers is None and not args.combo_matrix:
        pass  # use --suite (default full)

    try:
        tier_list = _tiers_from_args(args)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\nUse --suite full (default), combo strings like bf16_mixed@fp32, "
            "or --suite legacy for named presets."
        ) from exc

    print(f"[config] suite={args.suite} tiers={len(tier_list)}")
    for key, spec, label in tier_list:
        print(f"  [{key}] inference_precision={spec!r}")
        print(f"           {label}")

    baseline_key = _PYTORCH_BASELINE_KEY
    if baseline_key not in {t[0] for t in tier_list}:
        baseline_key = tier_list[0][0]
        print(
            f"[warn] PyTorch FP32 baseline {_PYTORCH_BASELINE_KEY!r} not in run; "
            f"using {baseline_key!r} for relative diffs"
        )
    baseline: dict[str, torch.Tensor] | None = None
    baseline_ms: float | None = None
    all_preds: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[tuple[str, str, float, float, float, float, float, float | None]] = []

    for key, precision, label in tier_list:
        print(f"[run] {key} ({label})...", flush=True)
        pred, ms_per = _run_tier(
            precision=precision,
            checkpoint=checkpoint,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        all_preds[key] = pred
        print(f"[run] {key} e2e forward={ms_per:.1f} ms ({1000.0 / ms_per:.2f} fwd/s)", flush=True)
        if key == baseline_key:
            baseline = pred
            baseline_ms = ms_per
            rows.append((key, label, ms_per, None, 0.0, 0.0, 0.0, 1.0))
            continue
        assert baseline is not None and baseline_ms is not None
        max_abs, mean_abs, max_rel, cos = _diff_vs_reference(baseline, pred)
        speedup = baseline_ms / ms_per
        rows.append((key, label, ms_per, speedup, max_abs, mean_abs, max_rel, cos))

    _print_table(rows)

    if not args.no_official_tol and baseline is not None:
        for tier_key, pred in all_preds.items():
            if tier_key == baseline_key:
                continue
            tol_rows = _official_tol_rows(baseline, pred)
            _print_official_tol_table(f"[official tol] {tier_key} vs {baseline_key}", tol_rows)

    if args.compare_hf:
        hf_ref = _load_hf_output_tensors(data_dir)
        if hf_ref is None:
            print(f"\n[compare-hf] skipped: {_OUTPUT_NAME} not found under {data_dir}")
        elif baseline is None:
            print("\n[compare-hf] skipped: no fp32 baseline in this run")
        else:
            max_abs, mean_abs, max_rel, cos = _diff_vs_reference(hf_ref, baseline)
            print(
                f"\n[compare-hf] pooled fp32 vs HF test-output: "
                f"max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} max_rel={max_rel:.4g} cos_sim={cos:.6f}"
            )
            tol_rows = _official_tol_rows(hf_ref, baseline)
            _print_official_tol_table("[official tol] fp32 vs HF test-output", tol_rows)
            passed = sum(1 for r in tol_rows if r[4])
            if passed < len(tol_rows):
                print(
                    "  [note] Accuracy tables above are vs "
                    f"{_PYTORCH_BASELINE_KEY} (PyTorch backbone FP32, E/D FP32). "
                    "HF pickle is float64 gold from Microsoft's test harness "
                    "(model.double(), use_lora=True, batch×2). "
                    "Compare custom combos against the PyTorch baseline, not HF gold alone."
                )


if __name__ == "__main__":
    main()
