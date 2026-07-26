from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import zmq

from flash_aurora.scheduler.addresses import ipc_pair, tcp_ephemeral_placeholders
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.localhost import (
    allocate_localhost_addresses,
    build_localhost_worker_config,
)
from flash_aurora.scheduler.protocol import (
    ForecastEvent,
    forecast_event_from_dict,
    forecast_event_to_dict,
)
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind


def test_allocate_ipc_and_tcp_placeholders(tmp_path: Path) -> None:
    ipc = allocate_localhost_addresses(transport="ipc", socket_dir=tmp_path)
    assert ipc.command_addr.startswith("ipc://")
    assert ipc.event_addr.startswith("ipc://")

    tcp = allocate_localhost_addresses(transport="tcp", host="127.0.0.1")
    assert tcp.command_addr == "tcp://127.0.0.1:0"
    assert tcp.event_addr == "tcp://127.0.0.1:0"

    command, event = ipc_pair(tmp_path / "pair")
    assert command != event
    assert tcp_ephemeral_placeholders()[0].endswith(":0")


def test_ready_event_protocol_round_trip() -> None:
    event = ForecastEvent(
        kind="ready",
        worker_preset="era5_pretrained",
        worker_id="localhost-era5",
        worker_device="cuda:0",
        worker_capacity=1,
        message="ready",
    )
    restored = forecast_event_from_dict(forecast_event_to_dict(event))
    assert restored == event


def test_worker_preload_emits_ready_and_resolves_tcp(tmp_path: Path) -> None:
    context = zmq.Context.instance()
    engine = MagicMock()
    engine.load.return_value = MagicMock()
    engine.config.device = "cuda:0"
    downloader = MagicMock()

    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr="tcp://127.0.0.1:0",
            event_addr="tcp://127.0.0.1:0",
            preload=True,
            poll_timeout_ms=100,
        ),
        engine=engine,
        downloader=downloader,
        context=context,
    )
    assert worker.command_addr.startswith("tcp://127.0.0.1:")
    assert not worker.command_addr.endswith(":0")
    assert worker.event_addr.startswith("tcp://127.0.0.1:")
    assert worker.command_addr != worker.event_addr

    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    wait_for_bind(worker.command_addr)

    client = ForecastClient(
        ForecastClientConfig(
            command_addr=worker.command_addr,
            event_addr=worker.event_addr,
            recv_timeout_ms=5_000,
        ),
        context=context,
    )
    ready = client.wait_for_ready(timeout_s=5.0, require_model_loaded=True)
    assert ready.kind in ("ready", "health")
    assert ready.message == "ready"
    engine.load.assert_called()
    assert worker.model_ready

    health = client.health()
    assert health.message == "ready"

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_connect_localhost_config_helper(tmp_path: Path) -> None:
    context = zmq.Context.instance()
    endpoints = allocate_localhost_addresses(transport="ipc", socket_dir=tmp_path)
    config = build_localhost_worker_config(
        preset="era5_pretrained",
        asset_root=tmp_path,
        endpoints=endpoints,
        preload=True,
        poll_timeout_ms=100,
    )
    engine = MagicMock()
    engine.load.return_value = MagicMock()
    engine.config.device = "cuda:0"
    worker = ForecastWorker(
        config,
        engine=engine,
        downloader=MagicMock(),
        context=context,
    )
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    wait_for_bind(worker.command_addr)

    client = ForecastClient(
        ForecastClientConfig(
            command_addr=worker.command_addr,
            event_addr=worker.event_addr,
            recv_timeout_ms=5_000,
        ),
        context=context,
    )
    ready = client.wait_for_ready(timeout_s=5.0)
    assert ready.message == "ready"
    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_listening_ready_without_preload(tmp_path: Path) -> None:
    context = zmq.Context.instance()
    command_addr, event_addr = ipc_pair(tmp_path)
    engine = MagicMock()
    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            preload=False,
            poll_timeout_ms=100,
        ),
        engine=engine,
        downloader=MagicMock(),
        context=context,
    )
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    wait_for_bind(worker.command_addr)

    client = ForecastClient(
        ForecastClientConfig(
            command_addr=worker.command_addr,
            event_addr=worker.event_addr,
            recv_timeout_ms=5_000,
        ),
        context=context,
    )
    ready = client.wait_for_ready(timeout_s=5.0, require_model_loaded=False)
    assert ready.message == "listening"
    engine.load.assert_not_called()

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()
