#!/usr/bin/env python3
"""Leave-one-out ablation of the LLM-style inference stack for GFMs.

Each row runs in a fresh subprocess so cuDNN autotune cannot leak across
mechanisms. Layout, AdaLN, CuTe attention, and BF16 routing are turned off
one at a time; framework autocast and torch.compile share the same timing
budget as LLM-style baselines. This table attributes the forward path, not
the job-level serving runtime.

    export AURORA_ASSET_ROOT=/path/to/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_ablation_loo.py
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
_WORKER = _BENCH_DIR / "_ablation_loo_worker.py"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import _bootstrap  # noqa: F401, E402

from _ablation_loo import (  # noqa: E402
    COMPILE_EXTRA_WARMUP,
    DEFAULT_PRESET,
    DEFAULT_REPEAT,
    DEFAULT_WARMUP,
    LOO_ROWS,
    SEED,
    intervals_overlap,
    row_ids,
)
from _asset_root import default_asset_root  # noqa: E402

import torch


def _parse_worker_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last = text.splitlines()[-1]
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return None


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
            payload.setdefault(
                "error",
                payload.get("error") or f"worker exit {proc.returncode}",
            )
        return payload
    err = (proc.stderr or proc.stdout or f"worker exit {proc.returncode}").strip()
    return {
        "ok": False,
        "row_id": row_id,
        "error": err[-800:],
    }


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return float(num) / float(den)


def _annotate(rows: dict[str, dict[str, Any]]) -> None:
    full = rows.get("full") or {}
    fp32 = rows.get("unfused_fp32") or {}
    full_mean = full.get("mean") if full.get("ok") else None
    full_std = full.get("std") if full.get("ok") else None
    fp32_mean = fp32.get("mean") if fp32.get("ok") else None
    fp32_std = fp32.get("std") if fp32.get("ok") else None
    for row in rows.values():
        if not row.get("ok"):
            row["vs_full"] = None
            row["vs_unfused_fp32"] = None
            row["vs_full_tie"] = False
            row["vs_unfused_fp32_tie"] = False
            continue
        mean = row["mean"]
        std = row["std"]
        row["vs_full"] = _ratio(mean, full_mean)
        row["vs_unfused_fp32"] = _ratio(fp32_mean, mean)
        row["vs_full_tie"] = (
            full_mean is not None
            and full_std is not None
            and intervals_overlap(mean, std, full_mean, full_std)
        )
        row["vs_unfused_fp32_tie"] = (
            fp32_mean is not None
            and fp32_std is not None
            and intervals_overlap(mean, std, fp32_mean, fp32_std)
        )


def _fmt_ratio(value: float | None, *, tie: bool) -> str:
    if value is None:
        return "—"
    suffix = " (tie)" if tie else ""
    return f"{value:.2f}x{suffix}"


def _fmt_lat(row: dict[str, Any]) -> str:
    if not row.get("ok"):
        return row.get("error", "FAIL")[:80]
    return f"{row['mean']:.1f} ± {row['std']:.2f}"


def _phys_rel(row: dict[str, Any], name: str) -> str:
    if not row.get("ok"):
        return "—"
    phys = (row.get("quality") or {}).get("phys") or {}
    if name not in phys:
        return "—"
    return f"{phys[name]['mean_rel']:.3e}"


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Mechanism leave-one-out ablation",
        "",
        "Paradigm: GFM inference keeps spatially structured tensors and high-intensity I/O.",
        "This table attributes the fused forward path (layout, AdaLN, short-window attention, precision).",
        "Each leave-one-out row turns off one mechanism in a fresh subprocess.",
        "Scheduler, pipeline, and ROI egress are reported elsewhere. Do not mix with isolate-tiers ratios.",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Preset: `{payload['preset']}`",
        f"- Seed: {payload['seed']}; warmup {payload['warmup']}, per-iter CUDA events n={payload['repeat']}",
        f"- Compile-after-load extra warmup: {payload['compile_extra_warmup']}",
        "",
        "| row | mechanism | mean±std (ms) | p50 | p99 | peak GiB | vs full | vs FP32 | 2t | 10u | n_fail |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row_id in row_ids():
        row = payload["rows"][row_id]
        if row.get("ok"):
            n_fail = (row.get("quality") or {}).get("n_fail", "—")
            n_vars = (row.get("quality") or {}).get("n_vars", "")
            fail_s = f"{n_fail}/{n_vars}" if n_vars != "" else str(n_fail)
            peak = f"{row['peak_gib']:.1f}"
            p50 = f"{row['p50']:.1f}"
            p99 = f"{row['p99']:.1f}"
        else:
            fail_s = "—"
            peak = "—"
            p50 = "—"
            p99 = "—"
        lines.append(
            f"| `{row_id}` | {row.get('mechanism', '')} | {_fmt_lat(row)} | "
            f"{p50} | {p99} | {peak} | "
            f"{_fmt_ratio(row.get('vs_full'), tie=bool(row.get('vs_full_tie')))} | "
            f"{_fmt_ratio(row.get('vs_unfused_fp32'), tie=bool(row.get('vs_unfused_fp32_tie')))} | "
            f"{_phys_rel(row, '2t')} | {_phys_rel(row, '10u')} | {fail_s} |"
        )
    lines += [
        "",
        "`vs full` is candidate mean / production mean (slowdown when a mechanism is removed).",
        "`vs FP32` is unfused FP32 mean / candidate mean. Overlapping mean±std intervals are marked `(tie)`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export AURORA_ASSET_ROOT=/path/to/aurora",
        "export CUTE_DSL_ARCH=sm_120a",
        "uv run --python 3.12 python benchmark/bench_ablation_loo.py",
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
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_BENCH_DIR / "ablation_loo_latest.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=_BENCH_DIR / "ablation_loo_latest.md",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

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
    for row in LOO_ROWS:
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
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
