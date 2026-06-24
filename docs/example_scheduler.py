#!/usr/bin/env python3
"""Forecast scheduler demo for flash-aurora.

The engine examples call AuroraEngine inside a long-lived notebook kernel. The
scheduler instead keeps one worker process on the GPU and lets clients submit
jobs over ZMQ. When this script exits, the worker subprocess is shut down and
GPU memory is released.

Prerequisites:
    Run example_era5 first, or ensure checkpoint and ERA5 cache exist under
    ASSET_ROOT. This demo uses download=False (cached ingress only).

Usage:
    uv run python docs/example_scheduler.py
    uv run python docs/example_scheduler.py --asset-root /path/to/aurora

    # Worker already running (see print_worker_cli_hint):
    uv run python docs/example_scheduler.py --client-only \\
        --command-addr tcp://127.0.0.1:9755 \\
        --event-addr tcp://127.0.0.1:9756
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import zmq

from flash_aurora.engine.core.asset_root import normalize_asset_root
from flash_aurora.engine.core.redaction import safe_path
from flash_aurora.scheduler import (
    ForecastClient,
    ForecastClientConfig,
    ForecastRequest,
)
from flash_aurora.scheduler.protocol import ForecastEvent
from flash_aurora.scheduler.worker import wait_for_bind

DEFAULT_PRESET = "era5_pretrained"
DEFAULT_VALID_TIME = "2023-01-01T06:00:00"
DEFAULT_TIME_INDEX = 1
DEFAULT_ROLLOUT_STEPS = 2
DEFAULT_INFERENCE_PRECISION = "bf16_mixed@fp32"
WORKER_JOIN_TIMEOUT_S = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash-Aurora scheduler demo")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Absolute asset root (default: AURORA_ASSET_ROOT)",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--valid-time", default=DEFAULT_VALID_TIME)
    parser.add_argument("--time-index", type=int, default=DEFAULT_TIME_INDEX)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--inference-precision", default=DEFAULT_INFERENCE_PRECISION)
    parser.add_argument(
        "--command-addr",
        default=None,
        help="ZMQ command socket (default: ipc under ASSET_ROOT/scheduler_demo)",
    )
    parser.add_argument(
        "--event-addr",
        default=None,
        help="ZMQ event socket (default: ipc under ASSET_ROOT/scheduler_demo)",
    )
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Do not spawn a worker; connect to an existing one",
    )
    return parser


def resolve_socket_dir(asset_root: Path) -> Path:
    socket_dir = asset_root / "scheduler_demo"
    socket_dir.mkdir(parents=True, exist_ok=True)
    return socket_dir


def default_socket_addresses(asset_root: Path) -> tuple[str, str]:
    socket_dir = resolve_socket_dir(asset_root)
    command_addr = f"ipc://{socket_dir / 'commands.ipc'}"
    event_addr = f"ipc://{socket_dir / 'events.ipc'}"
    return command_addr, event_addr


def spawn_worker(
    *,
    asset_root: Path,
    preset: str,
    inference_precision: str,
    command_addr: str,
    event_addr: str,
) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "flash_aurora.scheduler",
        "--preset",
        preset,
        "--asset-root",
        str(asset_root),
        "--inference-precision",
        inference_precision,
        "--command-addr",
        command_addr,
        "--event-addr",
        event_addr,
        "--poll-timeout-ms",
        "500",
    ]
    return subprocess.Popen(cmd)


def print_event(event: ForecastEvent) -> None:
    if event.kind == "step":
        if event.export_path is not None:
            print(f"step {event.step}: {safe_path(event.export_path)}")
        elif event.valid_time is not None:
            print(f"step {event.step} valid_time={event.valid_time}")
        else:
            print(f"step {event.step}")
    else:
        print(event.kind)


def run_health(client: ForecastClient, preset: str) -> None:
    print("\n--- health ---")
    health = client.health()
    print("kind:", health.kind)
    print("worker_preset:", health.worker_preset)
    print("message:", health.message)
    if health.worker_preset != preset:
        raise RuntimeError(
            f"worker preset {health.worker_preset!r} does not match {preset!r}"
        )


def run_export_forecast(
    client: ForecastClient,
    *,
    preset: str,
    valid_time: str,
    time_index: int,
    rollout_steps: int,
    export_dir: Path,
) -> None:
    print("\n--- forecast (export_paths) ---")
    request = ForecastRequest(
        request_id="scheduler-demo-export",
        preset=preset,
        steps=rollout_steps,
        valid_time=valid_time,
        time_index=time_index,
        download=False,
        export_dir=str(export_dir),
    )
    for event in client.forecast(request):
        print_event(event)


def run_metadata_forecast(
    client: ForecastClient,
    *,
    preset: str,
    valid_time: str,
    time_index: int,
) -> None:
    print("\n--- forecast (metadata_only) ---")
    request = ForecastRequest(
        request_id="scheduler-demo-metadata",
        preset=preset,
        steps=1,
        valid_time=valid_time,
        time_index=time_index,
        download=False,
        output_mode="metadata_only",
    )
    client.submit(request)
    for event in client.events(request.request_id):
        print_event(event)


def print_worker_cli_hint(asset_root: Path, preset: str, inference_precision: str) -> None:
    print("\n--- persistent worker (terminal) ---")
    print("export AURORA_ASSET_ROOT=" + str(asset_root))
    print(
        "python -m flash_aurora.scheduler \\\n"
        f"  --preset {preset} \\\n"
        '  --asset-root "$AURORA_ASSET_ROOT" \\\n'
        f"  --inference-precision {inference_precision} \\\n"
        "  --command-addr tcp://127.0.0.1:9755 \\\n"
        "  --event-addr tcp://127.0.0.1:9756"
    )


def shutdown_client(client: ForecastClient | None, *, stop_worker: bool) -> None:
    if client is None:
        return
    if stop_worker:
        try:
            client.shutdown_worker()
        except Exception as exc:
            print(f"shutdown_worker warning: {exc}", file=sys.stderr)
    client.close()


def terminate_worker(worker_proc: subprocess.Popen[bytes] | None) -> None:
    if worker_proc is None:
        return
    if worker_proc.poll() is not None:
        return
    worker_proc.terminate()
    try:
        worker_proc.wait(timeout=WORKER_JOIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        worker_proc.kill()
        worker_proc.wait(timeout=10)


def main() -> int:
    args = build_parser().parse_args()
    asset_root = normalize_asset_root(args.asset_root)
    command_addr, event_addr = default_socket_addresses(asset_root)
    if args.command_addr is not None:
        command_addr = args.command_addr
    if args.event_addr is not None:
        event_addr = args.event_addr

    export_dir = asset_root / "output" / "scheduler_demo"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("asset_root:", safe_path(asset_root))
    print("command_addr:", command_addr)
    print("event_addr:", event_addr)

    worker_proc: subprocess.Popen[bytes] | None = None
    client: ForecastClient | None = None
    context = zmq.Context.instance()

    try:
        if not args.client_only:
            worker_proc = spawn_worker(
                asset_root=asset_root,
                preset=args.preset,
                inference_precision=args.inference_precision,
                command_addr=command_addr,
                event_addr=event_addr,
            )
            wait_for_bind(command_addr)
            print("worker started for preset:", args.preset)

        client = ForecastClient(
            ForecastClientConfig(
                command_addr=command_addr,
                event_addr=event_addr,
                recv_timeout_ms=600_000,
            ),
            context=context,
        )
        # check if the worker is healthy
        run_health(client, args.preset)
        
        # task 1: export forecast
        run_export_forecast(
            client,
            preset=args.preset,
            valid_time=args.valid_time,
            time_index=args.time_index,
            rollout_steps=args.rollout_steps,
            export_dir=export_dir,
        )
        
        # task 2: metadata forecast
        run_metadata_forecast(
            client,
            preset=args.preset,
            valid_time=args.valid_time,
            time_index=args.time_index,
        )
        print_worker_cli_hint(asset_root, args.preset, args.inference_precision)
        return 0
    finally:
        shutdown_client(client, stop_worker=not args.client_only)
        terminate_worker(worker_proc)
        if worker_proc is not None:
            code = worker_proc.returncode
            if code not in (0, None, -15):
                print(f"worker exited with code {code}", file=sys.stderr)
        time.sleep(0.1)
        print("done")


if __name__ == "__main__":
    raise SystemExit(main())
