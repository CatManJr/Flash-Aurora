#!/usr/bin/env python3
"""Engine lifecycle bottleneck profile (cached ingress only, no download).

Measures CPU ingress/build, checkpoint load, H2D, rollout, and NetCDF export.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_engine_cycle.py \\
        --asset-root "$AURORA_ASSET_ROOT"

    uv run python benchmark/bench_engine_cycle.py \\
        --presets era5_pretrained hres_t0_finetuned --steps 4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _engine_cycle import EngineCycleTiming, measure_engine_cycle, purge_gpu  # noqa: E402
from _preset_ic import (  # noqa: E402
    PRECISION_PRESETS,
    _DEFAULT_TIME_INDEX,
    _DEFAULT_VALID_TIME,
    load_preset_batch,
    preset_engine_config,
)

import torch


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def _format_report(
    results: list[EngineCycleTiming],
    *,
    asset_root: Path,
    steps: int,
    generated: datetime,
) -> str:
    lines = [
        "# Engine cycle profile (no download)",
        "",
        f"- Generated: {generated.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- GPU: `{_gpu_name()}`",
        f"- Asset root: `{asset_root}`",
        f"- Rollout steps: {steps}",
        f"- Inference precision: `bf16_mixed@fp32` (unless preset default)",
        "- Excludes: CDS/ADS/MARS download and API queue",
        "",
        "## Summary",
        "",
        "| preset | total (s) | bottleneck | build_ic | build_model | load_ckpt | rollout | export |",
        "|--------|----------:|------------|---------:|------------:|----------:|--------:|-------:|",
    ]
    for row in results:
        export_total = (row.export_per_step_ms or 0.0) * row.rollout_steps / 1000.0
        lines.append(
            f"| {row.preset} | {row.engine_total_ms / 1000.0:.2f} | {row.bottleneck} "
            f"| {row.build_ic_ms / 1000.0:.2f} | {row.build_model_ms / 1000.0:.2f} "
            f"| {row.load_ckpt_ms / 1000.0:.2f} "
            f"| {row.rollout_total_ms / 1000.0:.2f} | {export_total:.2f} |"
        )

    lines.extend(["", "## Per-stage breakdown", ""])
    for row in results:
        lines.append(f"### {row.preset}")
        lines.append("")
        lines.append("| stage | ms | % of total |")
        lines.append("|-------|---:|-----------:|")
        for name, ms, pct in row.stage_rows():
            lines.append(f"| {name} | {ms:.1f} | {pct:.1f} |")
        lines.append("")
        lines.append(
            f"- forward/step: {row.forward_per_step_ms:.1f} ms "
            f"(rollout overhead: {row.rollout_overhead_ms:.1f} ms total)"
        )
        lines.append(
            f"- batch H2D prep: {row.batch_prep_ms:.1f} ms; "
            f"model H2D: {row.model_h2d_ms:.1f} ms"
        )
        if row.forward_stages is not None:
            st = row.forward_stages
            lines.append(
                f"- forward stages (CUDA avg): encoder {st.encoder_ms:.1f} ms, "
                f"backbone {st.backbone_ms:.1f} ms, decoder {st.decoder_ms:.1f} ms"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _resolve_presets(names: list[str] | None) -> list[str]:
    if not names:
        return [
            "era5_pretrained",
            "hres_t0_finetuned",
            "hres_0.1",
            "cams",
            "small_pretrained",
        ]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--inference-precision",
        default="bf16_mixed@fp32",
        help="Override preset inference precision",
    )
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument(
        "--model-warmup",
        action="store_true",
        help="Build/load one throwaway model before timing (stabilizes CuTe/Triton JIT)",
    )
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("FLASH_AURORA_GPU_GUARD", "0")

    results: list[EngineCycleTiming] = []
    errors: list[str] = []

    if args.model_warmup and torch.cuda.is_available():
        warmup_name = "era5_pretrained" if "era5_pretrained" in PRECISION_PRESETS else PRECISION_PRESETS[0]
        warmup_config = replace(
            preset_engine_config(warmup_name, asset_root),
            inference_precision=args.inference_precision,
            gpu_guard=False,
        )
        try:
            from flash_aurora.engine.core.checkpoint import CheckpointLoader

            loader = CheckpointLoader(warmup_config)
            model = loader.build_model()
            loader.load(model)
            model.to(device)
            del model
            purge_gpu()
            print(f"model warmup: {warmup_name}")
        except FileNotFoundError as exc:
            print(f"model warmup skipped: {exc}")

    for preset_name in _resolve_presets(args.presets):
        if preset_name not in PRECISION_PRESETS:
            errors.append(f"skip unknown preset {preset_name!r}")
            continue
        purge_gpu()
        config = preset_engine_config(preset_name, asset_root)
        config = replace(
            config,
            inference_precision=args.inference_precision,
            gpu_guard=False,
        )
        try:
            if preset_name == "small_pretrained":
                timing = measure_engine_cycle(
                    preset_name,
                    replace(
                        preset_engine_config(preset_name, asset_root),
                        inference_precision=args.inference_precision,
                        gpu_guard=False,
                    ),
                    valid_time=_DEFAULT_VALID_TIME["era5_pretrained"],
                    time_index=1,
                    rollout_steps=args.steps,
                    device=device,
                    include_export=not args.no_export,
                    ic_loader=lambda: load_preset_batch(preset_name, asset_root)[0],
                )
            else:
                timing = measure_engine_cycle(
                    preset_name,
                    config,
                    valid_time=_DEFAULT_VALID_TIME[preset_name],
                    time_index=_DEFAULT_TIME_INDEX[preset_name],
                    rollout_steps=args.steps,
                    device=device,
                    include_export=not args.no_export,
                )
            results.append(timing)
            print(
                f"{preset_name}: total={timing.engine_total_ms / 1000:.2f}s "
                f"bottleneck={timing.bottleneck} "
                f"build_ic={timing.build_ic_ms:.0f}ms "
                f"load={timing.load_ckpt_ms:.0f}ms "
                f"rollout={timing.rollout_total_ms:.0f}ms"
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))
            print(f"SKIP {preset_name}: {exc}")

    if not results:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1

    report = _format_report(
        results,
        asset_root=asset_root,
        steps=args.steps,
        generated=datetime.now(),
    )
    print()
    print(report)

    if args.report_out is not None:
        out = args.report_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
