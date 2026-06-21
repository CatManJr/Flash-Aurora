"""Shared helpers for AuroraPretrained benchmarks on real ERA5 ingress data."""

from __future__ import annotations

import dataclasses
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.bootstrap import ensure_repo_paths  # noqa: E402

ensure_repo_paths()

_DEFAULT_ASSET_ROOT = Path("/root/autodl-tmp/aurora")
_CHECKPOINT_NAME = "aurora-0.25-pretrained.ckpt"
_PYTORCH_BASELINE_KEY = "pytorch_backbone_fp32_encoder_decoder_fp32"

_LEGACY_NAMED_TIERS: tuple[tuple[str, str, str], ...] = (
    ("fp32", "fp32", "PyTorch FP32"),
    ("fast_fp32", "fast_fp32", "Triton layout + AdaLN + PyTorch GELU"),
    ("tf32", "tf32", "fast_fp32 + TF32 matmul + CuTe TF32 attn"),
    ("bf16_mixed", "bf16_mixed", "hybrid: TF32 QKV/proj + BF16 MLP + CuTe BF16 attn"),
    ("bf16", "bf16", "full backbone BF16 linears + CuTe BF16 attn"),
    ("pytorch_autocast", "pytorch_autocast", "PyTorch backbone BF16 autocast"),
)

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


def pytorch_reference_tiers() -> tuple[tuple[str, str, str], ...]:
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


def custom_matmul_combo_tiers(
    *,
    backbone_levels: tuple[str, ...],
    encoder_decoder_levels: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    from aurora.model.inference_precision import (
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


def full_default_suite(
    *,
    backbone_levels: tuple[str, ...] = (),
    encoder_decoder_levels: tuple[str, ...] = (),
) -> tuple[tuple[str, str, str], ...]:
    from aurora.model.inference_precision import (
        DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS,
        DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS,
    )

    bb = backbone_levels or DEFAULT_CUSTOM_COMBO_BACKBONE_LEVELS
    ed = encoder_decoder_levels or DEFAULT_CUSTOM_COMBO_ENCODER_DECODER_LEVELS
    return pytorch_reference_tiers() + custom_matmul_combo_tiers(
        backbone_levels=bb,
        encoder_decoder_levels=ed,
    )


def tier_entry(name: str) -> tuple[str, str, str]:
    for key, precision, label in _LEGACY_NAMED_TIERS:
        if name == key:
            return key, precision, label
    from aurora.model.inference_precision import describe_inference_config, resolve_inference_config

    cfg = resolve_inference_config(name)
    if cfg is None:
        raise ValueError(f"Could not resolve inference tier {name!r}.")
    spec = f"{cfg.backbone_matmul_level.value}@{cfg.encoder_decoder_matmul_level.value}"
    if name == spec or "@" in name:
        return cfg.config_label, spec, describe_inference_config(cfg)
    return cfg.config_label, name, describe_inference_config(cfg)


def tiers_from_args(args: Any) -> tuple[tuple[str, str, str], ...]:
    if args.tiers is not None:
        return tuple(tier_entry(t) for t in args.tiers)

    bb = tuple(args.backbone_levels)
    ed = tuple(args.encoder_decoder_levels)

    if args.suite == "legacy":
        return _LEGACY_NAMED_TIERS
    if args.suite == "combos":
        return custom_matmul_combo_tiers(backbone_levels=bb, encoder_decoder_levels=ed)
    if args.combo_matrix:
        if args.combos_only:
            return custom_matmul_combo_tiers(backbone_levels=bb, encoder_decoder_levels=ed)
        return full_default_suite(backbone_levels=bb, encoder_decoder_levels=ed)
    return full_default_suite()


def purge_gpu(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def load_era5_batch(
    asset_root: Path,
    *,
    era5_cache: Path | None = None,
    valid_time: datetime | None = None,
    time_index: int = 1,
) -> Any:
    """Build a validated IC ``Batch`` from cached CDS ERA5 NetCDF files."""
    from dataclasses import replace

    from engine.core.presets import DEFAULT_PRESETS
    from engine.ingress.build_ic import InitialConditionBuilder
    from engine.ingress.download import DataDownloader

    asset_root = asset_root.expanduser().resolve()
    cache = (era5_cache or asset_root / "era5").expanduser().resolve()
    vt = valid_time or datetime(2023, 1, 1, 6)

    config = replace(
        DEFAULT_PRESETS.get("era5_pretrained"),
        asset_root=asset_root,
        allow_hub_download=False,
    )
    downloader = DataDownloader(config)
    request = downloader.ingest_request(vt, cache_dir=cache, time_index=time_index, download=False)
    return InitialConditionBuilder(config).from_source(request)


def repeat_batch(batch: Any, n: int) -> Any:
    from aurora import Batch

    if n == 1:
        return batch
    assert isinstance(batch, Batch)
    return dataclasses.replace(
        batch,
        surf_vars={k: v.repeat(n, 1, 1, 1) for k, v in batch.surf_vars.items()},
        atmos_vars={k: v.repeat(n, 1, 1, 1, 1) for k, v in batch.atmos_vars.items()},
    )


def prediction_tensors(pred: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, tensor in getattr(pred, group).items():
            out[f"{group}.{name}"] = tensor.detach().float().cpu()
    return out


def diff_vs_reference(
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


def official_tol_rows(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> list[tuple[str, float, float, float, bool]]:
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


def print_official_tol_table(title: str, rows: list[tuple[str, float, float, float, bool]]) -> None:
    print(f"\n{title}")
    print("  metric: mean(|out-ref|) / mean(|ref|)  (aurora/tests/test_model.py)")
    print(f"  {'var':<6} {'mean_rel':>10} {'tol':>10} {'max_abs':>10} {'ok':>4}")
    print("  " + "-" * 44)
    for name, mean_rel, tol, max_abs, ok in rows:
        mark = "yes" if ok else "NO"
        print(f"  {name:<6} {mean_rel:10.4e} {tol:10.4e} {max_abs:10.4g} {mark:>4}")
    passed = sum(1 for r in rows if r[4])
    print(f"  summary: {passed}/{len(rows)} variables within official tolerance")


def print_summary_table(
    rows: list[tuple[str, str, float, float, float, float, float, float | None]],
) -> None:
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


def build_model(precision: str, checkpoint: Path, device: torch.device) -> Any:
    from aurora import AuroraPretrained

    model = AuroraPretrained(use_lora=False, inference_precision=precision)
    model.load_checkpoint_local(str(checkpoint), strict=True)
    model.eval()
    return model.to(device)


def time_forward(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[Any, float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
        else:
            import time

            t0 = time.perf_counter()
            pred = None
            for _ in range(repeat):
                pred = model.forward(batch)
            ms_total = (time.perf_counter() - t0) * 1e3
            peak_alloc = float("nan")
            peak_reserved = float("nan")

    return pred, ms_total / repeat, peak_alloc, peak_reserved


def run_tier(
    *,
    precision: str,
    checkpoint: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, torch.Tensor], float, float, float]:
    model = build_model(precision, checkpoint, device)
    try:
        pred, ms_per, peak_alloc, peak_reserved = time_forward(
            model, batch, warmup=warmup, repeat=repeat, device=device
        )
        return prediction_tensors(pred), ms_per, peak_alloc, peak_reserved
    finally:
        purge_gpu(model)


def cuda_oom_like(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "outofmemory" in msg.replace(" ", "") or "out of memory" in msg:
        return True
    if "cudaerrormemoryallocation" in msg.replace(" ", ""):
        return True
    if type(exc).__name__ in {"OutOfMemoryError", "AcceleratorError"} and "memory" in msg:
        return True
    return False


def recover_cuda_after_oom() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def probe_max_batch(
    *,
    model: Any,
    batch_b1: Any,
    cap: int,
    device: torch.device,
    forward_only: bool = True,
    rollout_steps: int = 2,
) -> int:
    from aurora import rollout

    use_cuda = device.type == "cuda"

    def attempt(n: int) -> bool:
        try:
            batch = repeat_batch(batch_b1, n).to(device)
            with torch.inference_mode():
                if forward_only:
                    _ = model.forward(batch)
                else:
                    for _ in rollout(model, batch, rollout_steps):
                        pass
            if use_cuda:
                torch.cuda.synchronize(device)
            return True
        except Exception as exc:
            if not cuda_oom_like(exc):
                raise
            recover_cuda_after_oom()
            return False

    if cap < 1 or not attempt(1):
        return 0 if cap >= 1 else 0
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if attempt(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def peak_mb_forward(model: Any, batch: Any, device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    purge_gpu()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model.forward(batch)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e6
