"""CLI entry point for the P1 forecast worker."""

from __future__ import annotations

import argparse
from pathlib import Path

from flash_aurora.engine.core.asset_root import normalize_asset_root
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, install_signal_handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash-Aurora single-worker forecast scheduler")
    parser.add_argument("--preset", required=True, help="Preset name bound to this worker")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Local asset root on data disk (default: AURORA_ASSET_ROOT)",
    )
    parser.add_argument(
        "--command-addr",
        default="tcp://127.0.0.1:9755",
        help="ZMQ bind address for incoming commands (worker PULL)",
    )
    parser.add_argument(
        "--event-addr",
        default="tcp://127.0.0.1:9756",
        help="ZMQ bind address for outgoing events (worker PUSH)",
    )
    parser.add_argument("--worker-id", default=None, help="Stable worker id for coordinators")
    parser.add_argument("--device", default=None, help="Inference device, for example cuda:1")
    parser.add_argument("--capacity", type=int, default=1, help="Concurrent job slots advertised")
    parser.add_argument("--inference-precision", default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--ic-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--forward-warmup-iters", type=int, default=None)
    parser.add_argument("--overlap-ic-load", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--async-export", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--poll-timeout-ms", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asset_root = normalize_asset_root(args.asset_root)
    config = ForecastWorkerConfig(
        preset=args.preset,
        asset_root=asset_root,
        command_addr=args.command_addr,
        event_addr=args.event_addr,
        worker_id=args.worker_id,
        device=args.device,
        capacity=args.capacity,
        inference_precision=args.inference_precision,
        export_dir=args.export_dir,
        ic_cache=args.ic_cache,
        forward_warmup_iters=args.forward_warmup_iters,
        overlap_ic_load=args.overlap_ic_load,
        async_export=args.async_export,
        poll_timeout_ms=args.poll_timeout_ms,
    )
    worker = ForecastWorker(config)
    install_signal_handlers(worker)
    print(
        f"[worker] id={worker.worker_id} preset={config.preset} device={worker.device} "
        f"capacity={config.capacity} command={config.command_addr} event={config.event_addr}",
        flush=True,
    )
    try:
        worker.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()


if __name__ == "__main__":
    main()
