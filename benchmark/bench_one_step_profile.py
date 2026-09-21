#!/usr/bin/env python3
"""Cold one-step bring-up profile at production mixed precision.

Times local IC read (no download), model construct, checkpoint load, device
move, then one forward. Finetuned presets use ``lora_merged``. Numbers only;
no figures.

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_one_step_profile.py \\
        --preset era5_pretrained --asset-root "$AURORA_ASSET_ROOT"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _latency_bench import time_forward_ms  # noqa: E402
from _preset_ic import PRECISION_PRESETS, checkpoint_path, load_preset_batch  # noqa: E402
from _pretrained_era5 import purge_gpu  # noqa: E402

import torch

PROFILE_PRECISION = "bf16_mixed@fp32"
WARMUP = 2
REPEAT = 5
SEED = 42


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_once(model: Any, batch: Any) -> Any:
    from flash_aurora.engine.core.model_protocol import model_uses_v1p5_rollout

    if model_uses_v1p5_rollout(model):
        batch_size = next(iter(batch.surf_vars.values())).shape[0]
        param = next(model.parameters())
        lead_hours = model.timestep.total_seconds() / 3600.0
        lead_times = torch.full(
            (batch_size,),
            lead_hours,
            device=param.device,
            dtype=param.dtype,
        )
        return model.forward(batch, lead_times=lead_times)
    return model.forward(batch)


def format_profile_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| preset | grid | data read (s) | construct (s) | checkpoint (s) | "
        "model to GPU (s) | IC to GPU (s) | first step (s) | warmed step (ms) | peak GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['preset']} | {row['grid']} | "
            f"{row['data_read_s']:.2f} | {row['construct_s']:.2f} | "
            f"{row['checkpoint_s']:.2f} | {row['model_to_device_s']:.2f} | "
            f"{row['ic_to_device_s']:.2f} | {row['first_step_s']:.2f} | "
            f"{row['warmed_step_ms']:.1f} | {row['peak_gib']:.1f} |"
        )
    return lines


def profile_preset(
    *,
    preset: str,
    asset_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    from flash_aurora.engine.core.model_registry import ModelFactory

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"[preset] {preset}  precision={PROFILE_PRECISION}", flush=True)
    t0 = time.perf_counter()
    batch, config = load_preset_batch(preset, asset_root)
    data_read_s = time.perf_counter() - t0
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"  data_read={data_read_s:.2f}s  ckpt={ckpt.name}", flush=True)

    variant = config.variant
    kwargs: dict[str, Any] = {"inference_precision": PROFILE_PRECISION}
    if variant.use_lora:
        kwargs["use_lora_merged_inference"] = True

    t0 = time.perf_counter()
    model = ModelFactory.create(
        variant.model_class,
        use_lora=variant.use_lora,
        lora_mode=variant.lora_mode,
        **kwargs,
    )
    construct_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.load_checkpoint_local(str(ckpt), strict=variant.strict_checkpoint)
    model.eval()
    checkpoint_s = time.perf_counter() - t0

    _synchronize(device)
    t0 = time.perf_counter()
    model = model.to(device)
    _synchronize(device)
    model_to_device_s = time.perf_counter() - t0

    _synchronize(device)
    t0 = time.perf_counter()
    dev_batch = batch.to(device)
    _synchronize(device)
    ic_to_device_s = time.perf_counter() - t0

    _synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = _forward_once(model, dev_batch)
    _synchronize(device)
    first_step_s = time.perf_counter() - t0
    del pred

    warmed_ms, _std_ms, peak_alloc_mb, _reserved = time_forward_ms(
        model,
        dev_batch,
        warmup=WARMUP,
        repeat=REPEAT,
        device=device,
    )
    peak_gib = peak_alloc_mb / 1024.0
    h, w = batch.spatial_shape
    grid = f"{h}x{w}"
    print(
        f"  construct={construct_s:.2f}s  ckpt={checkpoint_s:.2f}s  "
        f"to_gpu={model_to_device_s:.2f}s  ic_gpu={ic_to_device_s:.2f}s  "
        f"first={first_step_s:.2f}s  warmed={warmed_ms:.1f}ms  peak={peak_gib:.1f} GiB",
        flush=True,
    )
    purge_gpu(model, dev_batch)
    return {
        "preset": preset,
        "precision": PROFILE_PRECISION,
        "use_lora": variant.use_lora,
        "lora_merged": bool(variant.use_lora),
        "grid": grid,
        "checkpoint": ckpt.name,
        "data_read_s": data_read_s,
        "construct_s": construct_s,
        "checkpoint_s": checkpoint_s,
        "model_to_device_s": model_to_device_s,
        "ic_to_device_s": ic_to_device_s,
        "first_step_s": first_step_s,
        "warmed_step_ms": warmed_ms,
        "peak_gib": peak_gib,
    }


def write_markdown(path: Path, *, payload: dict[str, Any]) -> None:
    lines = [
        "# One-step bring-up profile (`bf16_mixed@fp32`)",
        "",
        f"- Generated: {payload['generated']}",
        f"- GPU: {payload['gpu']}",
        f"- PyTorch: `{payload['torch']}`",
        f"- Asset root: `{payload['asset_root']}`",
        "- Ingress: local cache only (no download)",
        "- Finetuned presets: `lora_merged`",
        f"- Warmed step: warmup {WARMUP}, repeat {REPEAT}",
        "",
    ]
    lines.extend(format_profile_table(payload["rows"]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        choices=PRECISION_PRESETS,
    )
    parser.add_argument("--preset", default=None, choices=PRECISION_PRESETS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    presets = list(args.presets or [])
    if args.preset:
        presets = [args.preset]
    if not presets:
        raise SystemExit("pass --preset or --presets")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    rows = [
        profile_preset(preset=name, asset_root=asset_root, device=device)
        for name in presets
    ]
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "asset_root": str(asset_root),
        "precision": PROFILE_PRECISION,
        "warmup": WARMUP,
        "repeat": REPEAT,
        "rows": rows,
    }
    json_out = args.json_out
    md_out = args.md_out
    if json_out is None:
        json_out = Path("benchmark") / "one_step_profile.json"
    if md_out is None:
        md_out = Path("benchmark") / "one_step_profile.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_out, payload=payload)
    print(f"[report] {md_out}")
    print(f"[json] {json_out}")


if __name__ == "__main__":
    main()
