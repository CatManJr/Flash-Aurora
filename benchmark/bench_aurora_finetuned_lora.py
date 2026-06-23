#!/usr/bin/env python3
"""LoRA eager vs merged on finetuned presets (legacy entry point).

Prefer ``bench_aurora_latency_all.py`` for the full per-preset latency matrix
(includes ``tc_tracking``, ``era5_pretrained``, ``small_pretrained``).

This script runs finetuned presets only (default includes ``tc_tracking``).

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_finetuned_lora.py \\
        --asset-root /root/autodl-tmp/aurora
"""

from __future__ import annotations

import os
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from bench_aurora_latency_all import main as _latency_main  # noqa: E402
from _preset_ic import PRECISION_PRESETS  # noqa: E402

FINETUNED_LORA_PRESETS: tuple[str, ...] = tuple(
    p for p in PRECISION_PRESETS if p not in ("era5_pretrained", "small_pretrained")
)


def main() -> None:
    import argparse
    from pathlib import Path

    from _asset_root import default_asset_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(FINETUNED_LORA_PRESETS),
        choices=FINETUNED_LORA_PRESETS,
    )
    parser.add_argument("--tiers", nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report-out", type=Path, default=None)
    args, _unknown = parser.parse_known_args()

    argv = [
        "bench_aurora_latency_all.py",
        "--asset-root",
        str(args.asset_root),
        "--presets",
        *args.presets,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    if args.tiers is not None:
        argv.extend(["--tiers", *args.tiers])
    if args.report_out is not None:
        argv.extend(["--report-out", str(args.report_out)])
    sys.argv = argv
    _latency_main()


if __name__ == "__main__":
    main()
