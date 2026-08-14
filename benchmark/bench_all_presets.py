#!/usr/bin/env python3
"""Per-preset mixed-precision latency, stage split, and VRAM (all presets except wave).

    export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_all_presets.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _preset_ic import PRECISION_PRESETS, checkpoint_path, load_preset_batch  # noqa: E402
from _pretrained_era5 import purge_gpu  # noqa: E402
from _stage_timing import time_forward_stages  # noqa: E402
from bench_indepth_eval import build_model, stats, time_samples  # noqa: E402

import torch

OUT_JSON = Path(_REPO) / "benchmark" / "all_presets_latest.json"
OUT_MD = Path(_REPO) / "benchmark" / "all_presets_latest.md"


def main() -> None:
    asset = (default_asset_root()).expanduser().resolve()
    device = torch.device("cuda")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "warmup": 3,
        "repeat": 8,
        "presets": {},
    }
    for preset in PRECISION_PRESETS:
        print(f"=== {preset} ===", flush=True)
        try:
            batch, config = load_preset_batch(preset, asset)
        except Exception as exc:  # noqa: BLE001
            payload["presets"][preset] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAIL load IC: {exc}", flush=True)
            continue
        ckpt = checkpoint_path(config, asset)
        purge_gpu()
        model = build_model(config, ckpt, precision="bf16_mixed@fp32", device=device)
        gpu_batch = batch.to(device)
        samples, peak = time_samples(model, gpu_batch, warmup=3, repeat=8, device=device)
        try:
            timing, _pred = time_forward_stages(
                model, gpu_batch, warmup=2, repeat=5, device=device
            )
            stages = {
                "encoder_ms": timing.encoder_ms,
                "backbone_ms": timing.backbone_ms,
                "decoder_ms": timing.decoder_ms,
                "total_ms": timing.total_ms,
                "backbone_pct": timing.backbone_pct,
            }
        except Exception as exc:  # noqa: BLE001
            stages = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  stages skipped: {exc}", flush=True)
        row = {
            "ok": True,
            "grid": list(config.variant.resolution),
            "lora": bool(config.variant.use_lora),
            "latency": {**stats(samples), "peak_gib": peak},
            "stages": stages,
        }
        bb = stages.get("backbone_pct")
        bb_s = f"{bb:.1f}%" if isinstance(bb, float) else "n/a"
        print(
            f"  mixed {row['latency']['mean']:.1f} +/- {row['latency']['std']:.2f} ms  "
            f"peak={peak:.1f} GiB  bb={bb_s}",
            flush=True,
        )
        payload["presets"][preset] = row
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        del model
        purge_gpu()

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# All-preset mixed-precision latency, stages, VRAM",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- `bf16_mixed@fp32`, warmup {payload['warmup']}, n={payload['repeat']}",
        "",
        "| preset | grid | mean±std (ms) | p95 | peak GiB | enc | backbone | dec | bb % |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for preset, row in payload["presets"].items():
        if not row.get("ok"):
            lines.append(f"| `{preset}` | — | {row.get('error', 'FAIL')} | — | — | — | — | — | — |")
            continue
        lat = row["latency"]
        st = row["stages"]
        g = "x".join(str(x) for x in row["grid"])
        if "error" in st:
            lines.append(
                f"| `{preset}` | {g} | {lat['mean']:.1f} ± {lat['std']:.2f} | "
                f"{lat['p95']:.1f} | {lat['peak_gib']:.1f} | — | — | — | — |"
            )
            continue
        lines.append(
            f"| `{preset}` | {g} | {lat['mean']:.1f} ± {lat['std']:.2f} | "
            f"{lat['p95']:.1f} | {lat['peak_gib']:.1f} | "
            f"{st['encoder_ms']:.1f} | {st['backbone_ms']:.1f} | {st['decoder_ms']:.1f} | "
            f"{st['backbone_pct']:.1f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
