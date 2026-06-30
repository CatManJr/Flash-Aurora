#!/usr/bin/env python3
"""Benchmark multi-step ``rollout_and_export`` with ``DistributedConfig`` (2-GPU).

Compares staged vs overlap-export rollout on a pipeline-parallel engine. Each mode
runs in a **fresh subprocess** so JIT/cuDNN/GPU state from one path does not warm
the other. Defaults benchmark ``era5_pretrained`` and ``hres_0.1`` (largest grid);
both need at least two 32 GiB GPUs.

Examples::

    export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_distributed_rollout.py \\
        --preset era5_pretrained --inference-precision bf16_mixed@fp32 \\
        --steps 4 --skip-download --force --warmup 1 --repeat 3

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_distributed_rollout.py \\
        --modes 2gpu_overlap --report-json /tmp/distributed_rollout.json

    # CPU / DRAM / GPU utilization curves (HPC-style figure)
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_distributed_rollout.py \\
        --preset era5_pretrained --steps 4 --skip-download --force --no-prompt \\
        --plot-utilization docs/image/distributed_rollout_utilization_5090.png
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import _bootstrap  # noqa: F401, E402

import torch
from _distributed_rollout_ipc import (  # noqa: E402
    RESULT_PREFIX as _RESULT_PREFIX,
    RolloutCaseResult,
    case_from_payload as _case_from_payload,
    case_to_payload as _case_to_payload,
)
from _distributed_rollout_assets import ensure_preset_assets, preset_assets_ready  # noqa: E402
from _pretrained_era5 import load_era5_batch, purge_gpu, resolve_bench_asset_root  # noqa: E402
from _preset_ic import load_preset_batch  # noqa: E402
from flash_aurora.engine.core.engine import AuroraEngine  # noqa: E402
from flash_aurora.engine.distributed import DistributedConfig  # noqa: E402
from flash_aurora.engine.ingress.download.options import default_download_workers  # noqa: E402
from flash_aurora.engine.runtime.resource_monitor import (  # noqa: E402
    ResourceMonitor,
    ResourceSample,
    device_index_from_name,
    plot_distributed_rollout_utilization,
)

DEFAULT_PRESETS = ("era5_pretrained", "hres_0.1")
DEFAULT_MODES = ("2gpu_staged", "2gpu_overlap")
ALL_MODES = DEFAULT_MODES


@dataclass(frozen=True)
class RolloutModeSpec:
    name: str
    overlap_rollout: bool
    decoder_spatial_parallel: bool
    async_export: bool


MODE_SPECS: dict[str, RolloutModeSpec] = {
    "2gpu_staged": RolloutModeSpec(
        name="2gpu_staged",
        overlap_rollout=False,
        decoder_spatial_parallel=True,
        async_export=True,
    ),
    "2gpu_overlap": RolloutModeSpec(
        name="2gpu_overlap",
        overlap_rollout=True,
        decoder_spatial_parallel=True,
        async_export=True,
    ),
}


@dataclass
class RolloutBenchmarkReport:
    preset: str
    inference_precision: str
    steps: int
    warmup: int
    repeat: int
    devices: tuple[str, ...]
    gpu_names: tuple[str, ...]
    cases: tuple[RolloutCaseResult, ...]


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


def _gpu_names() -> tuple[str, ...]:
    if not torch.cuda.is_available():
        return ("cpu",)
    return tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))


def _rollout_export_dir() -> Path:
    root = os.environ.get("AURORA_ROLLOUT_TMP")
    if root is None:
        autodl_tmp = Path("/root/autodl-tmp/rollout_tmp")
        if autodl_tmp.parent.is_dir():
            root = str(autodl_tmp)
    if root is not None:
        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="flash_aurora_rollout_", dir=base))
    return Path(tempfile.mkdtemp(prefix="flash_aurora_rollout_"))


def _peak_allocated_gib(device_names: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in device_names:
        idx = torch.device(name).index or 0
        out[name] = torch.cuda.max_memory_allocated(idx) / 1e9
    return out


def _run_mode(
    spec: RolloutModeSpec,
    *,
    preset: str,
    asset_root: Path,
    inference_precision: str,
    batch,
    steps: int,
    devices: tuple[str, ...],
    max_vram_gib: float,
    force: bool,
    warmup: int,
    repeat: int,
    monitor_interval_s: float | None = None,
) -> RolloutCaseResult:
    timings: list[float] = []
    peaks: list[dict[str, float]] = []
    last_status: dict[str, object] = {"enabled": False}
    profile_samples: list[ResourceSample] | None = None
    device_indices = [device_index_from_name(device) for device in devices]

    for iteration in range(warmup + repeat):
        purge_gpu()
        # CuTe JIT persists in-process; drop the engine and skip repeat warmup forwards
        # so 24 GiB cards can survive warmup=1 repeat=3 on hres_0.1.
        forward_warmup_iters = 2 if iteration == 0 else 0
        engine = AuroraEngine.from_preset(
            preset,
            asset_root=asset_root,
            inference_precision=inference_precision,
            forward_warmup_iters=forward_warmup_iters,
            distributed=DistributedConfig(
                devices=devices,
                max_vram_gib_per_device=max_vram_gib,
                force=force,
                rollout_steps=steps,
                decoder_spatial_parallel=spec.decoder_spatial_parallel,
                overlap_rollout=spec.overlap_rollout,
            ),
        )
        engine.load()
        for idx in range(len(devices)):
            torch.cuda.reset_peak_memory_stats(idx)

        export_dir = _rollout_export_dir()
        monitor: ResourceMonitor | None = None
        profile_this_iter = (
            monitor_interval_s is not None and iteration == warmup + repeat - 1
        )
        if profile_this_iter:
            monitor = ResourceMonitor(device_indices, interval_s=monitor_interval_s)
            monitor.start()

        started = time.perf_counter()
        paths = list(
            engine.rollout_and_export(
                batch,
                steps,
                export_dir=export_dir,
                async_export=spec.async_export,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if monitor is not None:
            profile_samples = monitor.stop()

        last_status = engine.distributed_status()
        peak = _peak_allocated_gib(devices)
        engine.release_gpu()
        del engine
        purge_gpu()

        if len(paths) != steps:
            raise RuntimeError(f"{spec.name}: expected {steps} exports, got {len(paths)}")

        if iteration >= warmup:
            timings.append(elapsed_ms)
            peaks.append(peak)

        shutil.rmtree(export_dir, ignore_errors=True)

    avg_total = sum(timings) / len(timings)
    avg_peak = {
        device: sum(row[device] for row in peaks) / len(peaks) for device in peaks[0]
    }

    return RolloutCaseResult(
        mode=spec.name,
        total_ms=avg_total,
        per_step_ms=avg_total / steps,
        peak_allocated_gib=avg_peak,
        distributed_status=last_status,
        resource_samples=profile_samples,
    )


def _emit_case_result(case: RolloutCaseResult, *, preset: str) -> None:
    print(_RESULT_PREFIX + json.dumps(_case_to_payload(case, preset=preset)), flush=True)


def _parse_case_result(stdout: str) -> RolloutCaseResult:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return _case_from_payload(json.loads(line[len(_RESULT_PREFIX) :]))
    raise RuntimeError(f"worker stdout missing {_RESULT_PREFIX.strip()!r} line")


def _build_worker_cmd(args: argparse.Namespace, mode: str, preset: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--preset",
        preset,
        "--inference-precision",
        args.inference_precision,
        "--steps",
        str(args.steps),
        "--num-gpus",
        str(args.num_gpus),
        "--max-vram-gib",
        str(args.max_vram_gib),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--monitor-interval",
        str(args.monitor_interval),
    ]
    if args.asset_root is not None:
        cmd.extend(["--asset-root", str(args.asset_root)])
    if args.force:
        cmd.append("--force")
    if args.skip_download:
        cmd.append("--skip-download")
    if args.no_hf_mirror:
        cmd.append("--no-hf-mirror")
    if args.no_prompt:
        cmd.append("--no-prompt")
    if args.plot_utilization is not None:
        plot_path = args.plot_utilization
        if plot_path.suffix:
            plot_path = plot_path.parent / f"{plot_path.stem}_{preset}{plot_path.suffix}"
        cmd.extend(["--plot-utilization", str(plot_path)])
    return cmd


def _run_mode_subprocess(args: argparse.Namespace, mode: str, preset: str) -> RolloutCaseResult:
    cmd = _build_worker_cmd(args, mode, preset)
    print(f"[bench] preset={preset} mode={mode} (subprocess) ...", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"preset={preset} mode={mode} failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return _parse_case_result(proc.stdout)


def _run_worker(args: argparse.Namespace) -> None:
    preset = args.preset[0]
    if len(args.preset) != 1:
        raise SystemExit("--worker requires exactly one --preset value")
    if args.num_gpus < 2:
        raise SystemExit("--num-gpus must be >= 2 (large presets require pipeline parallel)")
    if torch.cuda.device_count() < args.num_gpus:
        raise SystemExit(
            f"need {args.num_gpus} CUDA devices, found {torch.cuda.device_count()}"
        )
    if args.monitor_interval <= 0:
        raise SystemExit("--monitor-interval must be > 0")

    monitor_interval = args.monitor_interval if args.plot_utilization is not None else None
    asset_root = resolve_bench_asset_root(args.asset_root)
    devices = tuple(f"cuda:{i}" for i in range(args.num_gpus))
    batch = _load_batch(
        preset,
        asset_root,
        skip_download=args.skip_download,
        hf_mirror=not args.no_hf_mirror,
        prompt=not args.no_prompt,
    )
    case = _run_mode(
        MODE_SPECS[args.worker],
        preset=preset,
        asset_root=asset_root,
        inference_precision=args.inference_precision,
        batch=batch,
        steps=args.steps,
        devices=devices,
        max_vram_gib=args.max_vram_gib,
        force=args.force,
        warmup=args.warmup,
        repeat=args.repeat,
        monitor_interval_s=monitor_interval,
    )
    _emit_case_result(case, preset=preset)


def _print_report(report: RolloutBenchmarkReport) -> None:
    print()
    print("=" * 78)
    print(
        f"Distributed rollout  preset={report.preset}  precision={report.inference_precision!r}  "
        f"steps={report.steps}"
    )
    print("=" * 78)
    if report.gpu_names:
        print(f"GPUs: {', '.join(report.gpu_names)}")
    print()
    print(f"{'mode':<16} {'total_ms':>10} {'per_step_ms':>12}  peak_alloc (GiB)")
    print("-" * 78)
    for case in report.cases:
        peak_parts = "  ".join(
            f"{dev}={case.peak_allocated_gib[dev]:.1f}"
            for dev in sorted(case.peak_allocated_gib)
        )
        print(f"{case.mode:<16} {case.total_ms:10.1f} {case.per_step_ms:12.1f}  {peak_parts}")
    if len(report.cases) == 2:
        staged, overlap = report.cases
        speedup = staged.total_ms / overlap.total_ms if overlap.total_ms > 0 else 0.0
        print("-" * 78)
        print(f"overlap speedup vs staged: {speedup:.2f}x  ({staged.total_ms - overlap.total_ms:.0f} ms saved)")
    print("=" * 78)
    print("(each mode timed in a fresh subprocess)")
    print()
    print("Modes (requires >= 2 GPUs; large presets do not fit one 32 GiB card):")
    print("  2gpu_staged   — spatial decoder + async export, sequential rollout stages")
    print("  2gpu_overlap  — export step k-1 on cuda:0 while backbone step k on cuda:1")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument(
        "--preset",
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Preset(s) to benchmark (default: era5_pretrained hres_0.1)",
    )
    parser.add_argument("--inference-precision", default="bf16_mixed@fp32")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-vram-gib", type=float, default=32.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--download-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel DataDownloader threads for ingress fetch "
            f"(default: FLASH_AURORA_DOWNLOAD_WORKERS or {default_download_workers()})"
        ),
    )
    parser.add_argument("--no-hf-mirror", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(DEFAULT_MODES),
        help="2-GPU distributed rollout configurations to benchmark",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--plot-utilization",
        type=Path,
        default=None,
        help=(
            "Write notebook-style 2x2 CPU/DRAM/GPU/VRAM figures. "
            "Use a GPU tag in the stem, e.g. docs/image/distributed_rollout_utilization_4090.png "
            "writes docs/image/distributed_rollout_utilization_4090_{preset}_{mode}.png"
        ),
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.1,
        help="Resource sampling interval in seconds when --plot-utilization is set",
    )
    parser.add_argument(
        "--worker",
        choices=ALL_MODES,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker is not None:
        _run_worker(args)
        return

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.num_gpus < 2:
        raise SystemExit("--num-gpus must be >= 2 (large presets require pipeline parallel)")
    if torch.cuda.device_count() < args.num_gpus:
        raise SystemExit(
            f"need {args.num_gpus} CUDA devices, found {torch.cuda.device_count()}"
        )

    if args.monitor_interval <= 0:
        raise SystemExit("--monitor-interval must be > 0")

    asset_root = resolve_bench_asset_root(args.asset_root)
    devices = tuple(f"cuda:{i}" for i in range(args.num_gpus))
    reports: list[RolloutBenchmarkReport] = []

    for preset in args.preset:
        if args.skip_download:
            ready, reason = preset_assets_ready(preset, asset_root)
            if not ready:
                print(f"[skip] preset={preset}: {reason}", flush=True)
                continue
        else:
            print(f"[ensure] preset={preset} assets under {asset_root} ...", flush=True)
            ensure_preset_assets(
                preset,
                asset_root,
                hf_mirror=not args.no_hf_mirror,
                prompt=not args.no_prompt,
                download_workers=args.download_workers,
            )

        worker_args = argparse.Namespace(**{**vars(args), "skip_download": True})
        cases = [_run_mode_subprocess(worker_args, mode, preset) for mode in args.modes]
        report = RolloutBenchmarkReport(
            preset=preset,
            inference_precision=args.inference_precision,
            steps=args.steps,
            warmup=args.warmup,
            repeat=args.repeat,
            devices=devices,
            gpu_names=_gpu_names(),
            cases=tuple(cases),
        )
        reports.append(report)
        _print_report(report)

        if args.plot_utilization is not None:
            traces = {
                case.mode: case.resource_samples
                for case in cases
                if case.resource_samples is not None
            }
            if not traces:
                print(f"[warn] preset={preset}: no resource samples for plot", flush=True)
            else:
                plot_base = args.plot_utilization
                if plot_base.suffix:
                    plot_path = plot_base.parent / f"{plot_base.stem}_{preset}{plot_base.suffix}"
                else:
                    plot_path = plot_base / preset
                device_indices = [device_index_from_name(device) for device in devices]
                plot_paths = plot_distributed_rollout_utilization(
                    traces,
                    device_indices=device_indices,
                    output_path=plot_path,
                    title=(
                        f"Resource utilization during {args.steps}-step distributed rollout "
                        f"({preset})"
                    ),
                    device_labels={
                        device_index_from_name(device): device for device in devices
                    },
                )
                docs_dir = _BENCH_DIR.parent / "docs" / "image"
                docs_dir.mkdir(parents=True, exist_ok=True)
                for plot_path_written in plot_paths:
                    docs_copy = docs_dir / plot_path_written.name
                    if plot_path_written.resolve() != docs_copy.resolve():
                        shutil.copy2(plot_path_written, docs_copy)
                    print(f"[plot] {plot_path_written}")
                    if plot_path_written.resolve() != docs_copy.resolve():
                        print(f"[plot] {docs_copy}")

    if not reports:
        raise SystemExit("no presets ready to benchmark (missing checkpoint or ingress cache)")

    if args.report_json is not None:
        payload = {
            "inference_precision": args.inference_precision,
            "steps": args.steps,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "devices": list(devices),
            "gpu_names": list(_gpu_names()),
            "presets": [
                {
                    **asdict(report),
                    "cases": [_case_to_payload(case, preset=report.preset) for case in report.cases],
                }
                for report in reports
            ],
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[json] {args.report_json}")


if __name__ == "__main__":
    main()
