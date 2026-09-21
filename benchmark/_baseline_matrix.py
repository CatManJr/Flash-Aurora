"""End-to-end PyTorch baseline matrix for Group A isolate-tiers.

Rows give the fused mixed path a same-budget PyTorch class: eager FP32 with
TF32 off, eager TF32 with no fused kernels, torch.compile on both eager
PyTorch rows, the fused ladder ``fast_fp32`` / ``tf32@fp32`` / ``tf32x3@fp32`` /
``bf16_mixed@fp32``, and each SDPA backend in both FP32 and mixed BF16. An
unavailable row stores the exception instead of asserting incompatibility.
"""

from __future__ import annotations

import gc
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention import sdpa_kernel

from _preset_ic import output_var_tolerances
from _pretrained_era5 import purge_gpu, set_cute_window_attn
from _window_attn_libs import available_sdpa_backends
from bench_indepth_eval import PHYS_VARS, build_model, mean_rel, time_samples
from _ablation_loo import COMPILE_EXTRA_WARMUP, latency_stats, _forward_tensors, _quality_vs_fp32

PRODUCTION_PRECISION = "bf16_mixed@fp32"
FP32_PRECISION = "fp32"
FAST_FP32_PRECISION = "fast_fp32"
TF32_FUSED_PRECISION = "tf32@fp32"
TF32X3_FUSED_PRECISION = "tf32x3@fp32"
TF32_EAGER_PRECISION = "pytorch_tf32"
AUTOCAST_PRECISION = "pytorch_autocast"
DEFAULT_PRESET = "era5_pretrained"
DEFAULT_WARMUP = 2
DEFAULT_REPEAT = 5
SEED = 42
SDPA_PRECISIONS: tuple[tuple[str, str, str], ...] = (
    ("fp32", FP32_PRECISION, "eager FP32 with CuTe off"),
    ("bf16", PRODUCTION_PRECISION, "production mixed with CuTe off"),
)


@dataclass(frozen=True)
class BaselineRow:
    row_id: str
    precision: str
    mechanism: str
    compile_after_load: bool = False
    disable_cute: bool = False
    sdpa_backend: str | None = None


def _sdpa_rows() -> tuple[BaselineRow, ...]:
    rows: list[BaselineRow] = []
    for tag, precision, desc in SDPA_PRECISIONS:
        rows.append(
            BaselineRow(
                row_id=f"sdpa_auto_{tag}",
                precision=precision,
                mechanism=f"{desc}, default SDPA dispatch",
                disable_cute=True,
            )
        )
        for short, _backend in available_sdpa_backends():
            rows.append(
                BaselineRow(
                    row_id=f"sdpa_{short}_{tag}",
                    precision=precision,
                    mechanism=f"{desc}, SDPA {short}",
                    disable_cute=True,
                    sdpa_backend=short,
                )
            )
    return tuple(rows)


STATIC_ROWS: tuple[BaselineRow, ...] = (
    BaselineRow(
        row_id="mixed",
        precision=PRODUCTION_PRECISION,
        mechanism="production mixed fused path",
    ),
    BaselineRow(
        row_id="eager_fp32",
        precision=FP32_PRECISION,
        mechanism="eager PyTorch FP32, TF32 off, no Triton/CuTe",
    ),
    BaselineRow(
        row_id="eager_tf32",
        precision=TF32_EAGER_PRECISION,
        mechanism="eager PyTorch TF32 on, no Triton/CuTe",
    ),
    BaselineRow(
        row_id="compile_fp32",
        precision=FP32_PRECISION,
        mechanism="torch.compile after load on eager FP32",
        compile_after_load=True,
    ),
    BaselineRow(
        row_id="autocast",
        precision=AUTOCAST_PRECISION,
        mechanism="framework autocast BF16, no Triton/CuTe",
    ),
    BaselineRow(
        row_id="compile_autocast",
        precision=AUTOCAST_PRECISION,
        mechanism="torch.compile after load on framework autocast",
        compile_after_load=True,
    ),
    BaselineRow(
        row_id="fast_fp32",
        precision=FAST_FP32_PRECISION,
        mechanism="fused fast_fp32 (Triton layout, AdaLN, SDPA)",
    ),
    BaselineRow(
        row_id="tf32_fused",
        precision=TF32_FUSED_PRECISION,
        mechanism="fused tf32@fp32 (CuTe TF32 attention)",
    ),
    BaselineRow(
        row_id="tf32x3_fused",
        precision=TF32X3_FUSED_PRECISION,
        mechanism="fused tf32x3@fp32 (CuTe TF32x3 attention)",
    ),
)


def baseline_rows() -> tuple[BaselineRow, ...]:
    return STATIC_ROWS + _sdpa_rows()


def row_ids() -> tuple[str, ...]:
    return tuple(row.row_id for row in baseline_rows())


def get_row(row_id: str) -> BaselineRow:
    for row in baseline_rows():
        if row.row_id == row_id:
            return row
    known = ", ".join(row_ids())
    raise ValueError(f"unknown baseline row {row_id!r}; expected one of: {known}")


def extra_warmup(row: BaselineRow, warmup: int) -> int:
    if row.compile_after_load:
        return warmup + COMPILE_EXTRA_WARMUP
    return warmup


def compile_backbone_after_load(model: Any) -> None:
    model.backbone = torch.compile(model.backbone, dynamic=False)


def apply_row_flags(model: Any, row: BaselineRow) -> None:
    if row.disable_cute:
        set_cute_window_attn(model, False)
    if row.compile_after_load:
        compile_backbone_after_load(model)


def _sdpa_context(row: BaselineRow):
    if row.sdpa_backend is None:
        return nullcontext()
    backends = {name: enum for name, enum in available_sdpa_backends()}
    backend = backends.get(row.sdpa_backend)
    if backend is None:
        raise RuntimeError(f"SDPBackend for {row.sdpa_backend!r} is not in this PyTorch build")
    return sdpa_kernel(backends=[backend])


HEADLINE_IDS: tuple[str, ...] = (
    "eager_fp32",
    "eager_tf32",
    "compile_fp32",
    "autocast",
    "compile_autocast",
)
SDPA_MATH_REF_ID = "sdpa_math_fp32"


def strongest_passing_baseline(rows: dict[str, dict[str, Any]]) -> str | None:
    passing: list[tuple[float, str]] = []
    for row_id in HEADLINE_IDS:
        row = rows.get(row_id) or {}
        if not row.get("ok"):
            continue
        quality = row.get("quality") or {}
        if int(quality.get("n_fail", 1)) != 0:
            continue
        passing.append((float(row["mean"]), row_id))
    if not passing:
        return None
    passing.sort()
    return passing[0][1]


def annotate_speedups(rows: dict[str, dict[str, Any]]) -> None:
    """Speedup is SDPA MATH FP32 mean divided by the row mean."""
    ref = rows.get(SDPA_MATH_REF_ID) or {}
    ref_mean = ref.get("mean") if ref.get("ok") else None
    for row in rows.values():
        mean = row.get("mean") if row.get("ok") else None
        row["vs_sdpa_math"] = (
            None if ref_mean is None or mean is None or mean == 0 else ref_mean / mean
        )
        row["speedup_ref"] = SDPA_MATH_REF_ID


def run_baseline_row(
    *,
    row: BaselineRow,
    config: Any,
    ckpt: Path,
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "row_id": row.row_id,
        "precision": row.precision,
        "mechanism": row.mechanism,
        "compile_after_load": row.compile_after_load,
        "disable_cute": row.disable_cute,
        "sdpa_backend": row.sdpa_backend,
        "warmup": extra_warmup(row, warmup),
        "repeat": repeat,
        "compile_warnings": [],
    }
    model = None
    ref_model = None
    try:
        model = build_model(config, ckpt, precision=row.precision, device=device)
        apply_row_flags(model, row)
        dev_batch = batch.to(device)
        caught: list[warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _sdpa_context(row):
                samples, peak = time_samples(
                    model,
                    dev_batch,
                    warmup=extra_warmup(row, warmup),
                    repeat=repeat,
                    device=device,
                )
                candidate = _forward_tensors(model, dev_batch)
        payload["compile_warnings"] = [
            str(item.message)
            for item in caught
            if "recompile" in str(item.message).lower()
        ]
        payload.update(latency_stats(samples))
        payload["peak_gib"] = peak
        payload["samples_ms"] = [float(x) for x in samples]
        del model
        model = None
        purge_gpu()
        gc.collect()

        if row.row_id == "eager_fp32":
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
