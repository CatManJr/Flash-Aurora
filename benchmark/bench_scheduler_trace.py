#!/usr/bin/env python3
"""Group B: two sticky expensive presets and a two-GPU pool for the other five.

The pool workers may switch presets. A switch releases GPU memory and builds a
new engine. Ingress is read from the local cache. The run does not download
initial conditions and does not write NetCDF.

    export AURORA_ASSET_ROOT=/root/autodl-tmp/aurora
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_scheduler_trace.py calibrate
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_scheduler_trace.py trace \\
        --cycle-interval-s 120
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _job_metrics import summarize_run  # noqa: E402
from _job_timeline import JobRecord, JobStatus  # noqa: E402
from _preset_ic import load_preset_ingest_request  # noqa: E402
from _trace_generator import (  # noqa: E402
    SCHEDULER_TRACE_PRESETS,
    ROTATING_PRESETS,
    TraceJob,
    group_b_trace,
)
from _trace_replayer import CachedIngress, TraceReplayer  # noqa: E402
from flash_aurora.scheduler.addresses import ipc_pair  # noqa: E402
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig  # noqa: E402
from flash_aurora.scheduler.coordinator import (  # noqa: E402
    ForecastCoordinator,
    ForecastCoordinatorConfig,
    WorkerEndpoint,
)

PRECISION = "bf16_mixed@fp32"
# Outside the Flash-Aurora repository. Reports and worker logs stay off the git tree.
REPORT_DIR = Path(_REPO).parent / "groupB" / "reports"
READY_TIMEOUT_S = 20 * 60
BIND_TIMEOUT_S = 120.0
TRACE_TIMEOUT_S = 6 * 60 * 60
POLL_TIMEOUT_MS = 100
IPC_SCHEME = "ipc://"
_MEMORY_LINE = re.compile(
    r"memory_high_water allocated_gib=(?P<allocated>[0-9.]+) reserved_gib=(?P<reserved>[0-9.]+)"
)
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUTE_DSL_ARCH": "sm_120a",
}


@dataclass(frozen=True)
class GpuWorker:
    """One GPU process. ``presets[0]`` is loaded at startup when ``preload`` is set."""

    worker_id: str
    device: str
    presets: tuple[str, ...]
    calibrate_steps: int
    preload: bool

    @property
    def preset(self) -> str:
        return self.presets[0]


# Two GPUs stay on the expensive presets. The other two are a pool: they accept
# every smaller preset and rebuild the engine when the preset changes.
EXPENSIVE_WORKERS: tuple[GpuWorker, ...] = (
    GpuWorker("gpu-0", "cuda:0", ("hres_0.1",), 2, True),
    GpuWorker("gpu-1", "cuda:1", ("aurora_v1p5_ensemble",), 4, True),
)
POOL_WORKERS: tuple[GpuWorker, ...] = (
    GpuWorker("gpu-2", "cuda:2", ROTATING_PRESETS, 4, True),
    GpuWorker(
        "gpu-3",
        "cuda:3",
        ("hres_t0_finetuned", "era5_pretrained", "cams", "tc_tracking", "aurora_v1p5"),
        4,
        True,
    ),
)
SCHEDULER_TRACE_WORKERS: tuple[GpuWorker, ...] = EXPENSIVE_WORKERS + POOL_WORKERS


def cached_ingress(asset_root: Path, presets: Sequence[str]) -> dict[str, CachedIngress]:
    """Resolve each preset to its on-disk analysis. Missing files raise before any download."""
    found: dict[str, CachedIngress] = {}
    for preset in presets:
        request, _config = load_preset_ingest_request(preset, asset_root)
        if request.cache_dir is None:
            raise FileNotFoundError(f"preset {preset!r} resolved no cache directory")
        found[preset] = CachedIngress(
            valid_time=request.valid_time.isoformat(),
            cache_dir=str(request.cache_dir),
            time_index=request.time_index,
        )
    return found


def calibration_trace() -> tuple[TraceJob, ...]:
    """One cached job per production preset, all released at t=0."""
    return tuple(
        TraceJob(
            job_id=f"calibrate-{preset}",
            product="calibrate",
            preset=preset,
            steps=2 if preset == "hres_0.1" else 4,
            arrival_s=0.0,
            valid_time="1970-01-01T00:00:00",
            deadline_s=None,
        )
        for preset in SCHEDULER_TRACE_PRESETS
    )


def _workers_per_preset(workers: Sequence[GpuWorker]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for worker in workers:
        for preset in worker.presets:
            counts[preset] = counts.get(preset, 0) + 1
    return counts


def runtime_environment() -> dict[str, object]:
    """Software and hardware identity for one report.

    GPU name and driver come from nvidia-smi. Torch and CUDA come from the
    interpreter that launched the workers.
    """
    import torch

    gpus, driver = _gpu_inventory()
    cpu_model, logical_cpus = _cpu_identity()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "logical_cpus": logical_cpus,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "driver": driver,
        "gpus": gpus,
    }


def _cpu_identity() -> tuple[str | None, int]:
    """Model string from ``/proc/cpuinfo`` and the count of logical CPUs."""
    model: str | None = None
    logical_cpus = 0
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, os.cpu_count() or 0
    for line in cpuinfo.splitlines():
        if line.startswith("processor"):
            logical_cpus += 1
        elif model is None and line.startswith("model name"):
            model = line.split(":", 1)[1].strip() or None
    return model, logical_cpus or (os.cpu_count() or 0)


def _gpu_inventory() -> tuple[list[dict[str, object]], str | None]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return [], None
    gpus: list[dict[str, object]] = []
    driver: str | None = None
    for line in completed.stdout.splitlines():
        index_text, name, memory_text, driver_text = (part.strip() for part in line.split(",", 3))
        driver = driver or driver_text
        gpus.append(
            {
                "index": int(index_text),
                "name": name,
                "memory_mib": float(memory_text),
            }
        )
    return gpus, driver


def _read_memory_report(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["peak_allocated_gib"]), float(payload["peak_reserved_gib"])


def memory_high_water(log_text: str) -> tuple[float, float] | None:
    """Last allocator peaks printed by a worker, ``(allocated_gib, reserved_gib)``."""
    matches = list(_MEMORY_LINE.finditer(log_text))
    if not matches:
        return None
    last = matches[-1]
    return float(last.group("allocated")), float(last.group("reserved"))


def _wait_for_ipc(addr: str, *, timeout_s: float) -> None:
    if not addr.startswith(IPC_SCHEME):
        return
    socket_path = Path(addr[len(IPC_SCHEME):])
    deadline_s = time.monotonic() + timeout_s
    while not socket_path.exists():
        if time.monotonic() >= deadline_s:
            raise TimeoutError(f"{addr} was not bound within {timeout_s:.0f}s")
        time.sleep(0.05)


def _worker_env(asset_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AURORA_ASSET_ROOT"] = str(asset_root)
    env.update(_OFFLINE_ENV)
    return env


def _spawn_worker(
    worker: GpuWorker,
    *,
    asset_root: Path,
    command_addr: str,
    event_addr: str,
    log_path: Path,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "flash_aurora.scheduler",
        "--preset",
        worker.preset,
        "--asset-root",
        str(asset_root),
        "--worker-id",
        worker.worker_id,
        "--device",
        worker.device,
        "--inference-precision",
        PRECISION,
        "--command-addr",
        command_addr,
        "--event-addr",
        event_addr,
        "--poll-timeout-ms",
        str(POLL_TIMEOUT_MS),
    ]
    if len(worker.presets) > 1:
        command.extend(["--presets", ",".join(worker.presets)])
    if worker.preload:
        command.append("--preload")
    command.extend(["--memory-report", str(log_path.with_suffix(".memory.json"))])
    return subprocess.Popen(
        command,
        cwd=_REPO,
        env=_worker_env(asset_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _endpoint(worker: GpuWorker, command_addr: str, event_addr: str) -> WorkerEndpoint:
    return WorkerEndpoint(
        worker_id=worker.worker_id,
        preset=worker.preset,
        command_addr=command_addr,
        event_addr=event_addr,
        device=worker.device,
        capacity=1,
        accepted_presets=worker.presets if len(worker.presets) > 1 else (),
    )


def _stop_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()


def _tail(path: Path, *, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def run_trace(
    trace: Sequence[TraceJob],
    *,
    asset_root: Path,
    ingress: Mapping[str, CachedIngress],
    out_dir: Path,
    timeout_s: float,
    label: str,
) -> dict[str, object]:
    """Start the four workers, wait until the coordinator reports ready, replay ``trace``."""
    socket_dir = out_dir / "ipc"
    socket_dir.mkdir(parents=True, exist_ok=True)
    front_command, front_event = ipc_pair(socket_dir, prefix="front")
    bindings = {
        worker.worker_id: ipc_pair(socket_dir, prefix=worker.worker_id) for worker in SCHEDULER_TRACE_WORKERS
    }
    processes: list[subprocess.Popen[str]] = []
    log_paths = {worker.worker_id: out_dir / f"{worker.worker_id}.log" for worker in SCHEDULER_TRACE_WORKERS}
    for worker in SCHEDULER_TRACE_WORKERS:
        command_addr, event_addr = bindings[worker.worker_id]
        processes.append(
            _spawn_worker(
                worker,
                asset_root=asset_root,
                command_addr=command_addr,
                event_addr=event_addr,
                log_path=log_paths[worker.worker_id],
            )
        )
    coordinator: ForecastCoordinator | None = None
    client: ForecastClient | None = None
    thread: threading.Thread | None = None
    try:
        for worker in SCHEDULER_TRACE_WORKERS:
            _wait_for_ipc(bindings[worker.worker_id][0], timeout_s=BIND_TIMEOUT_S)
        coordinator = ForecastCoordinator(
            ForecastCoordinatorConfig(
                command_addr=front_command,
                event_addr=front_event,
                workers=tuple(
                    _endpoint(worker, *bindings[worker.worker_id]) for worker in SCHEDULER_TRACE_WORKERS
                ),
                poll_timeout_ms=POLL_TIMEOUT_MS,
            )
        )
        thread = threading.Thread(target=coordinator.serve_forever, name="group-b-coordinator", daemon=True)
        thread.start()
        client = ForecastClient(
            ForecastClientConfig(
                command_addr=front_command,
                event_addr=front_event,
                recv_timeout_ms=int(timeout_s * 1000),
            )
        )
        client.wait_for_ready(timeout_s=READY_TIMEOUT_S)
        timeline = TraceReplayer(client, ingress_by_preset=ingress).replay(trace, timeout_s=timeout_s)
    except Exception:
        for worker in SCHEDULER_TRACE_WORKERS:
            print(_tail(log_paths[worker.worker_id]), file=sys.stderr)
        raise
    finally:
        if client is not None:
            try:
                client.shutdown_worker()
            except Exception:
                pass
            client.close()
        if thread is not None:
            thread.join(timeout=30.0)
        _stop_processes(processes)

    records = timeline.records()
    try:
        summary: dict[str, object] = summarize_run(records, _workers_per_preset(SCHEDULER_TRACE_WORKERS))
    except ValueError as exc:
        summary = {"error": str(exc)}
    summary["label"] = label
    summary["environment"] = runtime_environment()
    summary["precision"] = PRECISION
    summary["ingress"] = {preset: asdict(item) for preset, item in ingress.items()}
    summary["memory_gib"] = {
        worker.worker_id: _memory_report(worker, log_paths[worker.worker_id])
        for worker in SCHEDULER_TRACE_WORKERS
    }
    summary["jobs_detail"] = [_job_row(record) for record in records]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{label}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["latency_s"] if "latency_s" in summary else summary, indent=2))
    print(f"wrote {report_path}")
    return summary


def _memory_report(worker: GpuWorker, log_path: Path) -> dict[str, object]:
    peaks = _read_memory_report(log_path.with_suffix(".memory.json"))
    if peaks is None and log_path.is_file():
        peaks = memory_high_water(log_path.read_text(encoding="utf-8", errors="replace"))
    allocated, reserved = peaks if peaks is not None else (None, None)
    return {
        "device": worker.device,
        "presets": list(worker.presets),
        "peak_allocated_gib": allocated,
        "peak_reserved_gib": reserved,
    }


def _job_row(record: JobRecord) -> dict[str, object]:
    """One point for a bubble chart: time on x, worker on y, service time as size."""
    completed = record.status is JobStatus.COMPLETED
    met_deadline = None
    if completed and record.has_deadline:
        met_deadline = record.met_deadline
    return {
        "request_id": record.request_id,
        "preset": record.preset,
        "status": record.status.value,
        "worker_id": record.worker_id,
        "worker_device": record.worker_device,
        "submitted_s": record.submitted_s,
        "finished_s": record.finished_s,
        "deadline_s": record.deadline_s,
        "met_deadline": met_deadline,
        "wait_s": record.wait_s if completed else None,
        "prepare_s": record.prepare_s if completed else None,
        "rollout_s": record.rollout_s if completed else None,
        "service_s": record.service_s if completed else None,
        "latency_s": record.latency_s if completed else None,
        "error": record.error,
    }


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--timeout-s", type=float, default=TRACE_TIMEOUT_S)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="One cached job for each of the seven presets")
    _shared_arguments(calibrate)
    trace = commands.add_parser("trace", help="Heterogeneous operational-cycle trace")
    _shared_arguments(trace)
    trace.add_argument("--cycle-interval-s", type=float, required=True)
    trace.add_argument("--ensemble-members", type=int, default=8)
    trace.add_argument("--rotation-rounds", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()
    ingress = cached_ingress(asset_root, SCHEDULER_TRACE_PRESETS)
    if args.command == "calibrate":
        trace = calibration_trace()
        label = "calibrate"
    else:
        trace = group_b_trace(
            args.cycle_interval_s,
            ensemble_members=args.ensemble_members,
            rotation_rounds=args.rotation_rounds,
        )
        label = "trace"
    run_trace(
        trace,
        asset_root=asset_root,
        ingress=ingress,
        out_dir=args.out_dir,
        timeout_s=args.timeout_s,
        label=label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
