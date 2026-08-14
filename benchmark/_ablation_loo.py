"""Leave-one-out rows for the LLM-style GFM inference stack.

Each LOO row starts from ``bf16_mixed@fp32`` (Triton layout, Triton AdaLN,
CuTe window attention, hybrid backbone matmul) and turns one mechanism off.
Framework autocast and compile share the same warmup/repeat budget. Layout and
AdaLN flags are applied after checkpoint load because ``inference_precision``
overrides constructor kwargs. These rows attribute the forward path, not the
job-level serving runtime.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _pretrained_era5 import prediction_tensors, purge_gpu, set_cute_window_attn
from _preset_ic import output_var_tolerances
from bench_indepth_eval import (  # noqa: E402
    PHYS_VARS,
    build_model,
    mean_rel,
    time_samples,
)

PRODUCTION_PRECISION = "bf16_mixed@fp32"
FP32_PRECISION = "fp32"
TF32_FUSED_PRECISION = "tf32@fp32"
AUTOCAST_PRECISION = "pytorch_autocast"
COMPILE_EXTRA_WARMUP = 8
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 12
DEFAULT_PRESET = "era5_pretrained"
SEED = 42
KIND_LOO = "loo"
KIND_BASELINE = "baseline"


@dataclass(frozen=True)
class LooRow:
    row_id: str
    precision: str
    mechanism: str
    kind: str = KIND_LOO
    disable_layout: bool = False
    disable_adaln: bool = False
    disable_cute: bool = False
    compile_after_load: bool = False


LOO_ROWS: tuple[LooRow, ...] = (
    LooRow(
        row_id="full",
        precision=PRODUCTION_PRECISION,
        mechanism="stacked production mixed fused path",
        kind=KIND_LOO,
    ),
    LooRow(
        row_id="no_layout",
        precision=PRODUCTION_PRECISION,
        mechanism="fused window layout",
        disable_layout=True,
    ),
    LooRow(
        row_id="no_adaln",
        precision=PRODUCTION_PRECISION,
        mechanism="fused AdaLN and residual",
        disable_adaln=True,
    ),
    LooRow(
        row_id="no_cute",
        precision=PRODUCTION_PRECISION,
        mechanism="short-window CuTe attention",
        disable_cute=True,
    ),
    LooRow(
        row_id="no_bf16_routing",
        precision=TF32_FUSED_PRECISION,
        mechanism="BF16 mixed-precision routing",
    ),
    LooRow(
        row_id="unfused_fp32",
        precision=FP32_PRECISION,
        mechanism="do-nothing PyTorch FP32",
        kind=KIND_BASELINE,
    ),
    LooRow(
        row_id="pytorch_autocast",
        precision=AUTOCAST_PRECISION,
        mechanism="framework mixed precision",
        kind=KIND_BASELINE,
    ),
    LooRow(
        row_id="compile_after_load",
        precision=PRODUCTION_PRECISION,
        mechanism="torch.compile after checkpoint load",
        kind=KIND_BASELINE,
        compile_after_load=True,
    ),
)

ROW_BY_ID: dict[str, LooRow] = {row.row_id: row for row in LOO_ROWS}


def row_ids() -> tuple[str, ...]:
    return tuple(row.row_id for row in LOO_ROWS)


def get_row(row_id: str) -> LooRow:
    try:
        return ROW_BY_ID[row_id]
    except KeyError as exc:
        known = ", ".join(row_ids())
        raise ValueError(f"unknown ablation row {row_id!r}; expected one of: {known}") from exc


def extra_warmup(row: LooRow, warmup: int) -> int:
    if row.compile_after_load:
        return warmup + COMPILE_EXTRA_WARMUP
    return warmup


def set_triton_layout(model: Any, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "use_triton_layout"):
            module.use_triton_layout = enabled


def set_triton_adaln(model: Any, enabled: bool) -> None:
    for module in model.modules():
        for attr in ("norm1", "norm2"):
            norm = getattr(module, attr, None)
            if norm is not None and hasattr(norm, "use_triton"):
                norm.use_triton = enabled


def compile_backbone_after_load(model: Any) -> None:
    model.backbone = torch.compile(model.backbone, dynamic=False)


def apply_loo_flags(model: Any, row: LooRow) -> None:
    """Mutate a loaded model so exactly the named mechanism is off."""
    if row.disable_layout:
        set_triton_layout(model, False)
    if row.disable_adaln:
        set_triton_adaln(model, False)
    if row.disable_cute:
        set_cute_window_attn(model, False)
    if row.compile_after_load:
        compile_backbone_after_load(model)


def latency_stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


def _forward_tensors(model: Any, batch: Any) -> dict[str, torch.Tensor]:
    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout
    from flash_aurora.models.aurora.rollout import prepare_rollout_batch

    prepared = prepare_rollout_batch(model, batch)
    with torch.inference_mode():
        if model_uses_v1p5_rollout(model):
            example = next(iter(prepared.surf_vars.values()))
            lead_hours = model.timestep.total_seconds() / 3600.0
            lead = torch.full(
                (example.shape[0],),
                lead_hours,
                device=example.device,
                dtype=example.dtype,
            )
            pred = model.forward(prepared, lead_times=lead)
        else:
            pred = model.forward(prepared)
    tensors = prediction_tensors(pred)
    del pred
    return tensors


def _quality_vs_fp32(
    *,
    config: Any,
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    specs = output_var_tolerances(config)
    fails: list[dict[str, Any]] = []
    for group, name, tol in specs:
        key = f"{group}.{name}"
        rel = mean_rel(reference[key], candidate[key])
        if rel > tol:
            fails.append({"name": name, "mean_rel": rel, "tol": tol})
    phys = {}
    for _group, name, unit in PHYS_VARS:
        key = f"surf_vars.{name}"
        if key not in reference or key not in candidate:
            continue
        phys[name] = {
            "unit": unit,
            "mean_rel": mean_rel(reference[key], candidate[key]),
        }
    return {
        "n_fail": len(fails),
        "n_vars": len(specs),
        "fails": fails,
        "phys": phys,
    }


def run_loo_row(
    *,
    row: LooRow,
    config: Any,
    ckpt: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    """Time the row, then score one-step mean-rel versus an unfused FP32 twin."""
    payload: dict[str, Any] = {
        "ok": True,
        "row_id": row.row_id,
        "precision": row.precision,
        "mechanism": row.mechanism,
        "kind": row.kind,
        "warmup": extra_warmup(row, warmup),
        "repeat": repeat,
    }
    model = None
    ref_model = None
    try:
        model = build_model(config, ckpt, precision=row.precision, device=device)
        apply_loo_flags(model, row)
        dev_batch = batch.to(device)
        samples, peak = time_samples(
            model,
            dev_batch,
            warmup=extra_warmup(row, warmup),
            repeat=repeat,
            device=device,
        )
        payload.update(latency_stats(samples))
        payload["peak_gib"] = peak
        payload["samples_ms"] = [float(x) for x in samples]

        candidate = _forward_tensors(model, dev_batch)
        del model
        model = None
        purge_gpu()
        gc.collect()

        if row.precision == FP32_PRECISION and not (
            row.disable_layout or row.disable_adaln or row.disable_cute or row.compile_after_load
        ):
            zeros = {name: {"unit": unit, "mean_rel": 0.0} for _g, name, unit in PHYS_VARS}
            payload["quality"] = {
                "n_fail": 0,
                "n_vars": len(output_var_tolerances(config)),
                "fails": [],
                "phys": zeros,
            }
        else:
            ref_model = build_model(config, ckpt, precision=FP32_PRECISION, device=device)
            reference = _forward_tensors(ref_model, batch.to(device))
            del ref_model
            ref_model = None
            payload["quality"] = _quality_vs_fp32(
                config=config, candidate=candidate, reference=reference
            )
        return payload
    except Exception as exc:  # noqa: BLE001
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload
    finally:
        del model
        del ref_model
        purge_gpu()
        gc.collect()


def intervals_overlap(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    lo_a, hi_a = mean_a - std_a, mean_a + std_a
    lo_b, hi_b = mean_b - std_b, mean_b + std_b
    return lo_a <= hi_b and lo_b <= hi_a
