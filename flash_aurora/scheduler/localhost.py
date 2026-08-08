"""Localhost loopback entry for the forecast scheduler (worker bind + ready)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from flash_aurora.engine.core.asset_root import normalize_asset_root
from flash_aurora.scheduler.addresses import ipc_pair, tcp_ephemeral_placeholders
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.worker import (
    ForecastWorker,
    ForecastWorkerConfig,
    install_signal_handlers,
)


@dataclass(frozen=True)
class LocalHostEndpoints:
    command_addr: str
    event_addr: str
    transport: str


def allocate_localhost_addresses(
    *,
    transport: str = "ipc",
    socket_dir: Path | str | None = None,
    host: str = "127.0.0.1",
) -> LocalHostEndpoints:
    """Allocate command/event bind addresses for a local worker."""
    if transport == "ipc":
        if socket_dir is None:
            raise ValueError("socket_dir is required for ipc transport")
        command_addr, event_addr = ipc_pair(socket_dir, prefix="localhost")
    elif transport == "tcp":
        command_addr, event_addr = tcp_ephemeral_placeholders(host=host)
    else:
        raise ValueError(f"unsupported transport {transport!r}; use 'ipc' or 'tcp'")
    return LocalHostEndpoints(
        command_addr=command_addr,
        event_addr=event_addr,
        transport=transport,
    )


def build_localhost_worker_config(
    *,
    preset: str,
    asset_root: Path,
    endpoints: LocalHostEndpoints,
    device: str | None = None,
    inference_precision: str | None = None,
    preload: bool = True,
    preload_rollout_steps: int = 1,
    poll_timeout_ms: int = 1000,
    worker_id: str | None = None,
) -> ForecastWorkerConfig:
    return ForecastWorkerConfig(
        preset=preset,
        asset_root=asset_root,
        command_addr=endpoints.command_addr,
        event_addr=endpoints.event_addr,
        worker_id=worker_id or f"localhost-{preset}",
        device=device,
        inference_precision=inference_precision,
        preload=preload,
        preload_rollout_steps=preload_rollout_steps,
        poll_timeout_ms=poll_timeout_ms,
    )


def connect_localhost_client(
    worker: ForecastWorker,
    *,
    recv_timeout_ms: int = 3_600_000,
    wait_ready: bool = True,
    require_model_loaded: bool = True,
    ready_timeout_s: float = 600.0,
) -> ForecastClient:
    """Connect a client to a bound worker and optionally wait for ready."""
    client = ForecastClient(
        ForecastClientConfig(
            command_addr=worker.command_addr,
            event_addr=worker.event_addr,
            recv_timeout_ms=recv_timeout_ms,
        )
    )
    if wait_ready:
        client.wait_for_ready(
            timeout_s=ready_timeout_s,
            require_model_loaded=require_model_loaded,
        )
    return client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a forecast worker on loopback (ipc or tcp://127.0.0.1), "
            "preload the engine, emit ready, then serve forever"
        )
    )
    parser.add_argument("--preset", required=True, help="Preset name bound to this worker")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Local asset root (default: AURORA_ASSET_ROOT)",
    )
    parser.add_argument(
        "--transport",
        choices=("ipc", "tcp"),
        default="ipc",
        help="Loopback transport (default: ipc under asset-root/scheduler_localhost)",
    )
    parser.add_argument(
        "--socket-dir",
        type=Path,
        default=None,
        help="Directory for ipc sockets (default: <asset-root>/scheduler_localhost)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind host (tcp transport)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-precision", default=None)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call engine.load() before accepting jobs (default: true)",
    )
    parser.add_argument("--preload-rollout-steps", type=int, default=1)
    parser.add_argument("--poll-timeout-ms", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asset_root = normalize_asset_root(args.asset_root)
    socket_dir = args.socket_dir
    if args.transport == "ipc" and socket_dir is None:
        socket_dir = asset_root / "scheduler_localhost"
    endpoints = allocate_localhost_addresses(
        transport=args.transport,
        socket_dir=socket_dir,
        host=args.host,
    )
    config = build_localhost_worker_config(
        preset=args.preset,
        asset_root=asset_root,
        endpoints=endpoints,
        device=args.device,
        inference_precision=args.inference_precision,
        preload=args.preload,
        preload_rollout_steps=args.preload_rollout_steps,
        poll_timeout_ms=args.poll_timeout_ms,
        worker_id=args.worker_id,
    )
    worker = ForecastWorker(config)
    install_signal_handlers(worker)
    print(
        f"[localhost] id={worker.worker_id} preset={config.preset} "
        f"device={worker.device} preload={config.preload}",
        flush=True,
    )
    print(f"[localhost] command_addr={worker.command_addr}", flush=True)
    print(f"[localhost] event_addr={worker.event_addr}", flush=True)
    print(
        "[localhost] connect with ForecastClient / wait_for_ready(), then submit jobs",
        flush=True,
    )
    try:
        worker.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
