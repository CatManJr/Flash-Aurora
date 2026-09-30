#!/usr/bin/env python3
"""Four-step rollout with async NetCDF export on two GPUs.

``hres_0.1`` and ``era5_pretrained`` each occupy one GPU and write only into
their own directory, so the two exports never share a file. Each directory
keeps the two newest step files; older ones are removed after the write so a
0.1 degree grid does not fill the disk. The trace is the overlap of one step's
NetCDF write with the next step's forward.

    export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_export_overlap.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _preset_ic import load_preset_ingest_request  # noqa: E402

PRECISION = "bf16_mixed@fp32"
STEPS = 4
RETAIN_STEPS = 2
RUNS = (
    ("hres_0.1", "cuda:0"),
    ("era5_pretrained", "cuda:1"),
)
EXPORT_ROOT = Path("/root/autodl-tmp/export-overlap")
REPORT_DIR = Path(_REPO).parent / "groupB" / "reports"
_OFFLINE = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUTE_DSL_ARCH": "sm_120a",
}


def run_one(preset: str, device: str, export_dir: Path, report_path: Path, asset_root: Path) -> None:
    """Load one preset and record four forward/export spans on ``device``."""
    from flash_aurora.engine.core.engine import AuroraEngine

    export_dir.mkdir(parents=True, exist_ok=True)
    request, _config = load_preset_ingest_request(preset, asset_root)
    engine = AuroraEngine.from_preset(
        preset,
        asset_root=asset_root,
        allow_hub_download=False,
        inference_precision=PRECISION,
        async_export=True,
        overlap_ic_load=True,
        export_dir=export_dir,
    )
    engine.config.device = device
    origin_s = time.perf_counter()
    batch = engine.prepare(request, rollout_steps=STEPS, overlap=True)
    prepared_s = time.perf_counter()
    trace: list[dict[str, float | int | str]] = []
    paths = list(
        engine.rollout_and_export(
            batch,
            STEPS,
            async_export=True,
            trace=trace,
            retain_steps=RETAIN_STEPS,
        )
    )
    spans = [
        {
            "kind": "prepare",
            "step": 0,
            "start_s": 0.0,
            "end_s": prepared_s - origin_s,
        }
    ]
    spans.extend(
        {
            "kind": span["kind"],
            "step": int(span["step"]),
            "start_s": float(span["start_s"]) - origin_s,
            "end_s": float(span["end_s"]) - origin_s,
        }
        for span in trace
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "preset": preset,
                "device": device,
                "steps": STEPS,
                "export_dir": str(export_dir),
                "files": [str(path) for path in paths],
                "spans": spans,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report_path} ({len(spans)} spans)", flush=True)


def _launch(preset: str, device: str, asset_root: Path, out_dir: Path) -> subprocess.Popen[str]:
    report_path = out_dir / f"export_overlap_{preset}.json"
    log_path = out_dir / f"export_overlap_{preset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(_OFFLINE)
    env["AURORA_ASSET_ROOT"] = str(asset_root)
    # Each process sees only its own GPU, so Triton and the default device stay aligned.
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--preset",
            preset,
            "--device",
            "cuda:0",
            "--export-dir",
            str(EXPORT_ROOT / preset),
            "--report",
            str(report_path),
            "--asset-root",
            str(asset_root),
        ],
        cwd=_REPO,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pair", help="Run both presets one after the other")
    one = commands.add_parser("run", help="One preset on one GPU")
    one.add_argument("--preset", required=True)
    one.add_argument("--device", required=True)
    one.add_argument("--export-dir", type=Path, required=True)
    one.add_argument("--report", type=Path, required=True)
    one.add_argument("--asset-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
        run_one(args.preset, args.device, args.export_dir, args.report, asset_root)
        return 0

    asset_root = default_asset_root().expanduser().resolve()

    out_dir = REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for preset, device in RUNS:
        process = _launch(preset, device, asset_root, out_dir)
        code = process.wait()
        log_path = out_dir / f"export_overlap_{preset}.log"
        if code != 0:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
            print(f"{preset} exited {code}", file=sys.stderr)
            return 1
    runs = [
        json.loads((out_dir / f"export_overlap_{preset}.json").read_text(encoding="utf-8"))
        for preset, _device in RUNS
    ]
    merged = out_dir / "export_overlap.json"
    merged.write_text(json.dumps({"steps": STEPS, "runs": runs}, indent=2), encoding="utf-8")
    print(f"wrote {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
