#!/usr/bin/env python3
"""CuTe versus PyTorch SDPA and FlashAttention-4 on Aurora QKV shapes.

This is a kernel microbenchmark, not an end-to-end quality test. The fused
ladder is ``fast_fp32`` (SDPA), CuTe TF32, CuTe TF32x3, and CuTe BF16_MIXED.
Every SDPA backend is timed in FP32 and in BF16. FA-4 is BF16 only. SDPA
``FLASH_ATTENTION`` already covers FlashAttention-2, so the standalone
``flash-attn`` package is skipped when that backend exists. FA-4 does not
take a Swin additive bias; masked rows record the exception type.

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_window_attn_sota.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

from _aurora_attn_shapes import SHAPES_ERA5_025  # noqa: E402
from _window_attn_libs import (  # noqa: E402
    AURORA_WINDOW_DHW,
    MICROBENCH_DTYPES,
    library_specs,
    probe_library,
    specs_for_dtype,
)
from bench_window_attn import (  # noqa: E402
    MEASURED,
    WARMUP,
    _scale_for_dh,
    bench,
    make_qkv,
    make_swin_bias,
)

_REPO = Path(_BENCH_DIR).parent


def _time_library(
    spec: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> dict[str, Any]:
    _out, err = probe_library(spec, q, k, v, scale=scale, bias=bias)
    if err is not None:
        return {"ok": False, "error": err, "mean_ms": None}
    try:
        if spec.make_timed is not None:
            timed = spec.make_timed(q, k, v, scale=scale, bias=bias)
            torch.cuda.synchronize()
            stats = bench(timed)
        else:
            stats = bench(lambda: spec.run(q, k, v, scale=scale, bias=bias))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "mean_ms": None}
    return {"ok": True, "error": None, "mean_ms": stats.mean, "ci95_ms": stats.ci95}


def run_shape(
    *,
    bwin: int,
    heads: int,
    n_tokens: int,
    head_dim: int,
    label: str,
    dtype: torch.dtype,
    masked: bool,
) -> dict[str, Any]:
    scale = _scale_for_dh(head_dim)
    q, k, v = make_qkv(bwin, heads, n_tokens, head_dim, dtype)
    bias = make_swin_bias(1, n_tokens) if masked else None
    rows: dict[str, Any] = {}
    for spec in specs_for_dtype(dtype):
        rows[spec.name] = _time_library(spec, q, k, v, scale=scale, bias=bias)
    return {
        "label": label,
        "bwin": bwin,
        "heads": heads,
        "n_tokens": n_tokens,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "masked": masked,
        "window_dhw": list(AURORA_WINDOW_DHW),
        "libraries": rows,
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    names = payload["library_order"]
    lines = [
        "# Window-attention libraries on Aurora tensor shapes",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Window DHW: `{tuple(payload['window_dhw'])}`",
        f"- Harness: trimmed mean of {payload['measured']} CUDA events "
        f"(warmup {payload['warmup']})",
        "",
        "FA-4 is BF16 only and does not take a Swin additive bias. "
        "SDPA FLASH_ATTENTION is the in-tree FA-2 path; a separate "
        "flash-attn package is not installed when that backend exists. "
        "FA-4 consumes BSHD rather than Aurora BHSD. On era5 enc1 "
        "1800×8 BF16 the full adapter is 2.343 ms and the kernel "
        "alone is 0.787 ms; with those layout copies, FA-4 in actual "
        "use is much slower than SDPA FLASH_ATTENTION. "
        "The table times the FA-4 kernel on pre-converted BSHD tensors. "
        "An unavailable row names the exception type.",
        "",
    ]
    for block in payload["shapes"]:
        mask = "masked -100" if block["masked"] else "unmasked"
        lines.append(f"## {block['label']} ({mask}, {block['dtype']})")
        lines.append("")
        lines.append("| library | ms | status |")
        lines.append("| --- | ---: | --- |")
        for name in names:
            if name not in block["libraries"]:
                continue
            row = block["libraries"][name]
            if row["ok"]:
                lines.append(f"| `{name}` | {row['mean_ms']:.3f} | ok |")
            else:
                lines.append(f"| `{name}` | — | `{row['error']}` |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=_REPO / "benchmark" / "window_attn_sota_latest.json")
    parser.add_argument("--md-out", type=Path, default=_REPO / "benchmark" / "window_attn_sota_latest.md")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    specs = library_specs()
    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "warmup": WARMUP,
        "measured": MEASURED,
        "window_dhw": list(AURORA_WINDOW_DHW),
        "library_order": [spec.name for spec in specs],
        "dtypes": [str(dtype).replace("torch.", "") for dtype in MICROBENCH_DTYPES],
        "shapes": [],
    }
    print(
        f"[gpu] {payload['gpu']}  libraries={', '.join(payload['library_order'])}",
        flush=True,
    )
    for dtype in MICROBENCH_DTYPES:
        dtype_name = str(dtype).replace("torch.", "")
        for masked in (False, True):
            for bwin, heads, n_tokens, head_dim, label in SHAPES_ERA5_025:
                print(
                    f"[shape] {label} dtype={dtype_name} masked={masked}",
                    flush=True,
                )
                block = run_shape(
                    bwin=bwin,
                    heads=heads,
                    n_tokens=n_tokens,
                    head_dim=head_dim,
                    label=label,
                    dtype=dtype,
                    masked=masked,
                )
                payload["shapes"].append(block)
                for name, row in block["libraries"].items():
                    if row["ok"]:
                        print(f"  {name}: {row['mean_ms']:.3f} ms", flush=True)
                    else:
                        print(f"  {name}: {row['error']}", flush=True)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
