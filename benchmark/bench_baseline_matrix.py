#!/usr/bin/env python3
"""Isolate-tiers PyTorch baseline matrix (Group A).

Each row is a fresh subprocess. Quality is one-step mean-rel versus an
in-process unfused FP32 twin. The paper dump remains the named twin once
frozen; this harness matches that protocol.

    export AURORA_ASSET_ROOT=/path/to/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_baseline_matrix.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent
_WORKER = _BENCH_DIR / "_baseline_matrix_worker.py"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import _bootstrap  # noqa: F401, E402

from _ablation_loo import COMPILE_EXTRA_WARMUP  # noqa: E402
from _asset_root import default_asset_root  # noqa: E402
from _baseline_matrix import (  # noqa: E402
    DEFAULT_PRESET,
    DEFAULT_REPEAT,
    DEFAULT_WARMUP,
    SDPA_MATH_REF_ID,
    SEED,
    annotate_speedups,
    baseline_rows,
    strongest_passing_baseline,
)
from bench_ablation_loo import _parse_worker_payload  # noqa: E402

import torch


def _run_row_isolated(
    *,
    row_id: str,
    preset: str,
    asset_root: Path,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(_WORKER),
        "--row",
        row_id,
        "--preset",
        preset,
        "--asset-root",
        str(asset_root),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_worker_payload(proc.stdout)
    if payload is not None:
        if proc.returncode != 0:
            payload["ok"] = False
            payload.setdefault("error", payload.get("error") or f"worker exit {proc.returncode}")
        return payload
    err = (proc.stderr or proc.stdout or f"worker exit {proc.returncode}").strip()
    return {"ok": False, "row_id": row_id, "error": err[-800:]}


def _fmt_lat(row: dict[str, Any]) -> str:
    if not row.get("ok"):
        return row.get("error", "fail")[:40]
    return f"{row['mean']:.1f}±{row['std']:.2f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}x"


def _phys_rel(row: dict[str, Any], name: str) -> str:
    phys = (row.get("quality") or {}).get("phys") or {}
    if name not in phys:
        return "—"
    return f"{phys[name]['mean_rel']:.2e}"


def _annotate(rows: dict[str, dict[str, Any]]) -> None:
    annotate_speedups(rows)


def _to_markdown(payload: dict[str, Any]) -> str:
    ref_id = payload.get("speedup_ref") or SDPA_MATH_REF_ID
    lines = [
        "# Group A PyTorch baseline matrix",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Preset: `{payload['preset']}`",
        f"- Isolate-tiers, seed {payload['seed']}, warmup {payload['warmup']}, "
        f"n={payload['repeat']}",
        f"- Compile extra warmup: {payload['compile_extra_warmup']}",
        f"- Speedup reference: `{ref_id}` (SDPA MATH, FP32, CuTe off)",
        "",
        "| row | mechanism | mean±std (ms) | p50 | p99 | peak GiB | vs SDPA MATH | n_fail |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row_def in baseline_rows():
        row = payload["rows"][row_def.row_id]
        if row.get("ok"):
            quality = row.get("quality") or {}
            fail_s = f"{quality.get('n_fail', '—')}/{quality.get('n_vars', '')}"
            peak = f"{row['peak_gib']:.1f}"
            p50 = f"{row['p50']:.1f}"
            p99 = f"{row['p99']:.1f}"
        else:
            fail_s = row.get("error", "fail")[:48]
            peak = "—"
            p50 = "—"
            p99 = "—"
        lines.append(
            f"| `{row_def.row_id}` | {row.get('mechanism', '')} | {_fmt_lat(row)} | "
            f"{p50} | {p99} | {peak} | {_fmt_ratio(row.get('vs_sdpa_math'))} | "
            f"{fail_s} |"
        )
    lines += [
        "",
        "`vs SDPA MATH` is `sdpa_math_fp32` mean / row mean. MATH is the portable "
        "SDPA backend (CuTe off, FP32). FLASH and CUDNN rows that do not dispatch "
        "are recorded by exception type. Fused `fast_fp32` / `tf32_fused` / "
        "`tf32x3_fused` / `mixed` are treatments.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export AURORA_ASSET_ROOT=/path/to/aurora",
        "export CUTE_DSL_ARCH=sm_120a",
        "uv run python benchmark/bench_baseline_matrix.py",
        "uv run python benchmark/bench_window_attn_sota.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--rows", nargs="+", default=None, help="Subset of row ids")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_BENCH_DIR / "baseline_matrix_latest.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=_BENCH_DIR / "baseline_matrix_latest.md",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    selected = list(baseline_rows())
    if args.rows:
        wanted = set(args.rows)
        selected = [row for row in selected if row.row_id in wanted]
        missing = wanted.difference(row.row_id for row in selected)
        if missing:
            raise SystemExit(f"unknown rows: {sorted(missing)}")

    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "preset": args.preset,
        "seed": SEED,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "compile_extra_warmup": COMPILE_EXTRA_WARMUP,
        "rows": {},
    }
    for row in selected:
        print(f"=== {row.row_id} ({row.mechanism}) ===", flush=True)
        result = _run_row_isolated(
            row_id=row.row_id,
            preset=args.preset,
            asset_root=asset_root,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        payload["rows"][row.row_id] = result
        if result.get("ok"):
            print(
                f"    mean={result['mean']:.1f} std={result['std']:.2f} "
                f"p99={result['p99']:.1f} peak={result['peak_gib']:.1f} GiB",
                flush=True,
            )
        else:
            print(f"    FAIL {result.get('error', 'unknown')[:400]}", flush=True)

    _annotate(payload["rows"])
    payload["speedup_ref"] = SDPA_MATH_REF_ID
    payload["strongest_baseline"] = strongest_passing_baseline(payload["rows"])
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")
    print(f"speedup reference: {payload['speedup_ref']}")


if __name__ == "__main__":
    main()
