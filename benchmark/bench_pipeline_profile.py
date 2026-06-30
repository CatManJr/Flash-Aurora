#!/usr/bin/env python3
"""Profile pipeline-parallel load: per-stage latency and per-GPU memory.

Measures encoder / backbone / decoder compute time on each device, H2D/D2D
transfer between stages, parameter footprint, and peak VRAM vs planner estimates.

Examples::

    export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora
    export CDSAPI_KEY='<api_key>'

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_pipeline_profile.py \\
        --preset era5_pretrained --inference-precision bf16_mixed@fp32 \\
        --skip-download --force --warmup 1 --repeat 5

    uv run python benchmark/bench_pipeline_profile.py \\
        --preset era5_pretrained --inference-precision fp32 \\
        --skip-download --force --report-json /tmp/pipeline_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from _pipeline_stage_timing import PipelineLoadProfile, build_load_profile  # noqa: E402
from _pretrained_era5 import load_era5_batch, purge_gpu, resolve_bench_asset_root  # noqa: E402
from _preset_ic import load_preset_batch  # noqa: E402
from flash_aurora.engine.core.engine import AuroraEngine  # noqa: E402
from flash_aurora.engine.distributed import DistributedConfig  # noqa: E402


@dataclass
class ProfileReport:
    preset: str
    inference_precision: str
    devices: tuple[str, ...]
    encoder_device: str
    backbone_device: str
    decoder_device: str
    timing_ms: dict[str, float]
    peak_allocated_mib: dict[str, float]
    peak_reserved_mib: dict[str, float]
    after_load_allocated_mib: dict[str, float]
    param_gib: dict[str, float]
    plan_estimated_per_device_gib: tuple[float, ...]
    plan_estimated_peak_gib: float
    decoder_spatial_parallel: bool = False
    decoder_spatial_devices: tuple[str, ...] = ()
    estimated_busy_fraction: dict[str, float] | None = None


def _load_batch(
    preset: str,
    asset_root: Path,
    *,
    skip_download: bool,
    hf_mirror: bool,
    prompt: bool,
):
    if preset == "era5_pretrained":
        return load_era5_batch(
            asset_root,
            download=not skip_download,
            hf_mirror=hf_mirror,
            prompt=prompt,
        )
    batch, _config = load_preset_batch(preset, asset_root)
    return batch


def _report_from_profile(
    preset: str,
    inference_precision: str,
    profile: PipelineLoadProfile,
    *,
    status: dict[str, object],
) -> ProfileReport:
    t = profile.timing
    return ProfileReport(
        preset=preset,
        inference_precision=inference_precision,
        devices=tuple(status["devices"]),  # type: ignore[arg-type]
        encoder_device=str(status["encoder_device"]),
        backbone_device=str(status["backbone_device"]),
        decoder_device=str(status["decoder_device"]),
        timing_ms={
            "prepare": t.prepare_ms,
            "encoder": t.encoder_ms,
            "enc_to_bb": t.enc_to_bb_ms,
            "backbone": t.backbone_ms,
            "bb_to_dec": t.bb_to_dec_ms,
            "decoder": t.decoder_ms,
            "post": t.post_ms,
            "total": t.total_ms,
            "transfer_total": t.transfer_ms,
            "compute_total": t.compute_ms,
        },
        peak_allocated_mib=profile.peak_allocated_mib,
        peak_reserved_mib=profile.peak_reserved_mib,
        after_load_allocated_mib=profile.after_load_allocated_mib,
        param_gib={dev: nbytes / (1024**3) for dev, nbytes in profile.param_bytes.items()},
        plan_estimated_per_device_gib=profile.plan_estimated_per_device_gib,
        plan_estimated_peak_gib=profile.plan_estimated_peak_gib,
        decoder_spatial_parallel=bool(status.get("decoder_spatial_parallel")),
        decoder_spatial_devices=tuple(status.get("decoder_spatial_devices", ())),  # type: ignore[arg-type]
        estimated_busy_fraction=status.get("estimated_busy_fraction"),  # type: ignore[arg-type]
    )


def _print_report(report: ProfileReport) -> None:
    t = report.timing_ms
    total = t["total"]
    print()
    print("=" * 72)
    print(f"Pipeline profile  preset={report.preset}  precision={report.inference_precision!r}")
    print("=" * 72)
    print(f"Placement: encoder={report.encoder_device}  backbone={report.backbone_device}  "
          f"decoder={report.decoder_device}")
    if report.decoder_spatial_parallel:
        west, east = report.decoder_spatial_devices
        print(f"Decoder spatial split: west={west}  east={east}")
    print()
    print("Stage latency (ms, CUDA events per device + wall for transfers)")
    print("-" * 72)
    rows = [
        ("prepare (CPU+crop)", t["prepare"], None),
        (f"encoder ({report.encoder_device})", t["encoder"], report.encoder_device),
        ("transfer enc->bb", t["enc_to_bb"], None),
        (f"backbone ({report.backbone_device})", t["backbone"], report.backbone_device),
        ("transfer bb->dec", t["bb_to_dec"], None),
        (f"decoder ({report.decoder_device})", t["decoder"], report.decoder_device),
        ("post (unnormalise)", t["post"], report.decoder_device),
        ("TOTAL", total, None),
    ]
    for label, ms, _dev in rows:
        pct = 100.0 * ms / total if total > 0 and label != "TOTAL" else 100.0 if label == "TOTAL" else 0.0
        pct_str = f"{pct:5.1f}%" if label != "TOTAL" else ""
        print(f"  {label:<28} {ms:8.1f} ms  {pct_str}")
    print()
    print("Measured compute busy fraction (from stage timings)")
    print("-" * 72)
    enc_ms = t["encoder"] + t["post"]
    bb_ms = t["backbone"]
    dec_ms = t["decoder"]
    busy_ms = {dev: 0.0 for dev in report.devices}
    busy_ms[report.encoder_device] += enc_ms
    busy_ms[report.backbone_device] += bb_ms
    busy_ms[report.decoder_device] += dec_ms
    compute_total = enc_ms + bb_ms + dec_ms
    for dev in sorted(busy_ms):
        frac = busy_ms[dev] / compute_total if compute_total > 0 else 0.0
        print(f"  {dev}: {busy_ms[dev]:8.1f} ms  ({100.0 * frac:5.1f}% of compute)")
    print()
    print("Per-device memory (MiB)")
    print("-" * 72)
    print(f"  {'device':<10} {'params':>10} {'after_load':>12} {'peak_alloc':>12} {'peak_rsv':>12} {'plan_est':>10}")
    devices = sorted(set(report.peak_allocated_mib) | set(report.param_gib))
    plan_map = {
        report.devices[i]: report.plan_estimated_per_device_gib[i]
        for i in range(min(len(report.devices), len(report.plan_estimated_per_device_gib)))
    }
    for dev in devices:
        params = report.param_gib.get(dev, 0.0) * 1024
        after = report.after_load_allocated_mib.get(dev, 0.0)
        peak = report.peak_allocated_mib.get(dev, 0.0)
        rsv = report.peak_reserved_mib.get(dev, 0.0)
        est = plan_map.get(dev, 0.0) * 1024
        print(f"  {dev:<10} {params:10.0f} {after:12.0f} {peak:12.0f} {rsv:12.0f} {est:10.0f}")
    print()
    print(f"Planner single-GPU peak estimate: {report.plan_estimated_peak_gib:.1f} GiB")
    busy = getattr(report, "estimated_busy_fraction", None)
    if busy:
        print("Estimated compute busy fraction (sequential pipeline):")
        for dev, frac in sorted(busy.items()):
            print(f"  {dev}: {100.0 * frac:.1f}%")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--preset", default="era5_pretrained")
    parser.add_argument("--inference-precision", default="bf16_mixed@fp32")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-vram-gib", type=float, default=32.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-hf-mirror", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--no-decoder-spatial", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    purge_gpu()

    asset_root = resolve_bench_asset_root(args.asset_root)
    devices = tuple(f"cuda:{i}" for i in range(args.num_gpus))
    dist_config = DistributedConfig(
        devices=devices,
        max_vram_gib_per_device=args.max_vram_gib,
        force=args.force,
        decoder_spatial_parallel=not args.no_decoder_spatial,
    )

    engine = AuroraEngine.from_preset(
        args.preset,
        asset_root=asset_root,
        inference_precision=args.inference_precision,
        hf_mirror=not args.no_hf_mirror,
        distributed=dist_config,
    )

    try:
        batch = _load_batch(
            args.preset,
            asset_root,
            skip_download=args.skip_download,
            hf_mirror=not args.no_hf_mirror,
            prompt=not args.no_prompt,
        )
        engine.load()
        status = engine.distributed_status()
        profile = build_load_profile(
            engine.model,
            batch,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        report = _report_from_profile(
            args.preset,
            args.inference_precision,
            profile,
            status=status,
        )
        _print_report(report)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(asdict(report), indent=2) + "\n")
            print(f"[json] {args.report_json}")
    finally:
        engine.release_gpu()
        purge_gpu()


if __name__ == "__main__":
    main()
