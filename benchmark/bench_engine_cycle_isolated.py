#!/usr/bin/env python3
"""Run engine-cycle configs in isolated subprocesses (fair GPU/JIT state).

Each configuration is a fresh Python process with --model-warmup so CuTe JIT
does not skew the timed preset. Rollout uses --forward-warmup (default 2).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent


@dataclass(frozen=True)
class Config:
    name: str
    flags: tuple[str, ...]


CONFIGS: tuple[Config, ...] = (
    Config("baseline", ()),
    Config("overlap_ic", ("--overlap-ic-load",)),
    Config("async_export", ("--async-export",)),
    Config("overlap_async", ("--overlap-ic-load", "--async-export")),
)

_LINE_RE = re.compile(
    r"^(?P<preset>\S+): total=(?P<total>[\d.]+)s "
    r"bottleneck=(?P<bottleneck>\S+) "
    r"build_ic=(?P<build_ic>[\d.]+)ms "
    r"load=(?P<load>[\d.]+)ms "
    r"rollout=(?P<rollout>[\d.]+)ms"
)


@dataclass
class Row:
    config: str
    preset: str
    total_s: float
    bottleneck: str
    build_ic_ms: float
    load_ms: float
    rollout_ms: float
    forward_per_step_ms: float | None = None
    export_total_s: float | None = None


def _parse_forward_step(report_text: str, preset: str) -> float | None:
    section = report_text.split(f"### {preset}", 1)
    if len(section) < 2:
        return None
    block = section[1].split("###", 1)[0]
    match = re.search(r"forward/step:\s*([\d.]+)\s*ms", block)
    return float(match.group(1)) if match else None


def _parse_export_total(report_text: str, preset: str) -> float | None:
    section = report_text.split(f"### {preset}", 1)
    if len(section) < 2:
        return None
    block = section[1].split("###", 1)[0]
    for line in block.splitlines():
        if line.startswith("| export |"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                return float(parts[1]) / 1000.0
    return None


def _run_config(
    config: Config,
    *,
    asset_root: Path,
    presets: list[str],
    steps: int,
    forward_warmup: int,
    inference_precision: str,
    report_dir: Path,
    export_parent: Path,
) -> tuple[list[Row], str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"engine_cycle_{config.name}_{stamp}.md"
    export_dir = export_parent / config.name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_BENCH_DIR / "bench_engine_cycle.py"),
        "--asset-root",
        str(asset_root),
        "--presets",
        *presets,
        "--steps",
        str(steps),
        "--forward-warmup",
        str(forward_warmup),
        "--model-warmup",
        "--inference-precision",
        inference_precision,
        "--export-dir",
        str(export_dir),
        "--report-out",
        str(report_path),
        *config.flags,
    ]
    env = os.environ.copy()
    env.setdefault("FLASH_AURORA_GPU_GUARD", "0")
    print(f"\n=== [{config.name}] {' '.join(config.flags) or '(serial sync)'} ===", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"config {config.name} failed (exit {proc.returncode}):\n{output}")

    rows: list[Row] = []
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else output
    for line in output.splitlines():
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        preset = match.group("preset")
        rows.append(
            Row(
                config=config.name,
                preset=preset,
                total_s=float(match.group("total")),
                bottleneck=match.group("bottleneck"),
                build_ic_ms=float(match.group("build_ic")),
                load_ms=float(match.group("load")),
                rollout_ms=float(match.group("rollout")),
                forward_per_step_ms=_parse_forward_step(report_text, preset),
                export_total_s=_parse_export_total(report_text, preset),
            )
        )
    if not rows:
        raise RuntimeError(f"config {config.name}: no result lines parsed\n{output}")
    shutil.rmtree(export_dir, ignore_errors=True)
    return rows, report_text


def _format_matrix(all_rows: list[Row], *, export_parent: Path, baseline_name: str = "baseline") -> str:
    presets = sorted({row.preset for row in all_rows})
    configs = [c.name for c in CONFIGS]
    base = {(r.preset): r for r in all_rows if r.config == baseline_name}

    lines = [
        "# Engine cycle isolated comparison",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "- Isolation: **one subprocess per configuration**",
        "- Each run: `--model-warmup` + `--forward-warmup 2`",
        "- Includes NetCDF export",
        f"- Export parent: `{export_parent}` (cleaned per config)",
        "",
        "## Summary (total wall time, seconds)",
        "",
        "| preset | " + " | ".join(configs) + " | best | vs baseline |",
        "|--------|" + "|".join(["----------:" for _ in configs]) + "|------|------------:|",
    ]
    for preset in presets:
        cells: list[str] = []
        best = float("inf")
        best_name = ""
        for config in configs:
            row = next((r for r in all_rows if r.config == config and r.preset == preset), None)
            if row is None:
                cells.append("—")
                continue
            cells.append(f"{row.total_s:.1f}")
            if row.total_s < best:
                best = row.total_s
                best_name = config
        base_total = base[preset].total_s if preset in base else None
        vs = f"{(best / base_total - 1.0) * 100:+.1f}%" if base_total else "—"
        lines.append(f"| {preset} | " + " | ".join(cells) + f" | {best_name} | {vs} |")

    lines.extend(["", "## Rollout forward/step (ms, warmed)", ""])
    lines.append("| preset | " + " | ".join(configs) + " |")
    lines.append("|--------|" + "|".join(["-------:" for _ in configs]) + "|")
    for preset in presets:
        cells = []
        for config in configs:
            row = next((r for r in all_rows if r.config == config and r.preset == preset), None)
            cells.append(f"{row.forward_per_step_ms:.1f}" if row and row.forward_per_step_ms else "—")
        lines.append(f"| {preset} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Export total (seconds, timed steps)", ""])
    lines.append("| preset | " + " | ".join(configs) + " |")
    lines.append("|--------|" + "|".join(["-------:" for _ in configs]) + "|")
    for preset in presets:
        cells = []
        for config in configs:
            row = next((r for r in all_rows if r.config == config and r.preset == preset), None)
            cells.append(f"{row.export_total_s:.1f}" if row and row.export_total_s else "—")
        lines.append(f"| {preset} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--presets", nargs="+", default=["era5_pretrained", "hres_0.1"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--forward-warmup", type=int, default=2)
    parser.add_argument("--inference-precision", default="bf16_mixed@fp32")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--export-parent",
        type=Path,
        default=Path("/tmp/flash-aurora-engine-cycle"),
        help="Parent dir for per-config export trees (cleaned after each run)",
    )
    args = parser.parse_args()

    report_dir = _BENCH_DIR
    export_parent = args.export_parent.expanduser().resolve()
    export_parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[Row] = []
    for config in CONFIGS:
        rows, _report = _run_config(
            config,
            asset_root=args.asset_root.expanduser().resolve(),
            presets=args.presets,
            steps=args.steps,
            forward_warmup=args.forward_warmup,
            inference_precision=args.inference_precision,
            report_dir=report_dir,
            export_parent=export_parent,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  {row.preset}: total={row.total_s:.2f}s "
                f"rollout={row.rollout_ms:.0f}ms "
                f"fwd/step={row.forward_per_step_ms or 0:.1f}ms "
                f"export={row.export_total_s or 0:.1f}s",
                flush=True,
            )

    matrix = _format_matrix(all_rows, export_parent=export_parent)
    print()
    print(matrix)
    out = args.report_out or (
        report_dir / f"engine_cycle_isolated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out.write_text(matrix, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
