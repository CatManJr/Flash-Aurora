from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import zmq

from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.protocol import ForecastRequest, SchedulerError
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind


@pytest.fixture
def zmq_addresses(tmp_path: Path) -> tuple[str, str, zmq.Context]:
    context = zmq.Context.instance()
    command_addr = f"ipc://{tmp_path / 'commands.ipc'}"
    event_addr = f"ipc://{tmp_path / 'events.ipc'}"
    return command_addr, event_addr, context


def _build_mock_engine(tmp_path: Path) -> MagicMock:
    engine = MagicMock()
    batch = MagicMock()
    engine.prepare.return_value = batch
    engine.prepare_from_netcdf.return_value = batch
    export_path = tmp_path / "prediction-000.nc"
    export_path.write_text("nc")
    engine.rollout_and_export.return_value = iter([export_path])
    return engine


def _start_worker_thread(worker: ForecastWorker) -> threading.Thread:
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    return thread


def test_worker_forecast_export_paths(zmq_addresses: tuple[str, str, zmq.Context], tmp_path: Path) -> None:
    command_addr, event_addr, context = zmq_addresses
    engine = _build_mock_engine(tmp_path)
    downloader = MagicMock()
    downloader.ingest_request.return_value = MagicMock()

    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            poll_timeout_ms=100,
        ),
        engine=engine,
        downloader=downloader,
        context=context,
    )
    wait_for_bind(command_addr)
    thread = _start_worker_thread(worker)

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=5000),
        context=context,
    )

    request = ForecastRequest(
        request_id="req-1",
        preset="era5_pretrained",
        steps=1,
        valid_time="2024-06-01T06:00:00",
        export_dir=str(tmp_path / "out"),
    )
    events = client.forecast(request)
    kinds = [event.kind for event in events]
    assert kinds == ["accepted", "preparing", "running", "step", "completed"]
    assert events[-2].export_path == str(tmp_path / "prediction-000.nc")

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_worker_forecast_last_step_array(
    zmq_addresses: tuple[str, str, zmq.Context],
    tmp_path: Path,
) -> None:
    command_addr, event_addr, context = zmq_addresses
    engine = _build_mock_engine(tmp_path)
    prediction = MagicMock()
    prediction.surf_vars = {"2t": torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)}
    prediction.metadata.time = (datetime(2024, 6, 1, 12, 0, 0),)
    engine.rollout_stream.return_value = iter([prediction])
    downloader = MagicMock()
    downloader.ingest_request.return_value = MagicMock()

    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            poll_timeout_ms=100,
        ),
        engine=engine,
        downloader=downloader,
        context=context,
    )
    wait_for_bind(command_addr)
    thread = _start_worker_thread(worker)

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=5000),
        context=context,
    )

    request = ForecastRequest(
        request_id="req-array",
        preset="era5_pretrained",
        steps=1,
        valid_time="2024-06-01T06:00:00",
        output_mode="last_step_array",
        preview_var="2t",
    )
    events = client.forecast(request)

    assert [event.kind for event in events] == ["accepted", "preparing", "running", "step", "completed"]
    assert events[-2].array_name == "2t"
    np.testing.assert_array_equal(events[-2].array(), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert events[-2].export_path is None

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_worker_forecast_failure_releases_engine(
    zmq_addresses: tuple[str, str, zmq.Context],
    tmp_path: Path,
) -> None:
    command_addr, event_addr, context = zmq_addresses
    engine = _build_mock_engine(tmp_path)
    engine.prepare.side_effect = RuntimeError("prepare failed")
    downloader = MagicMock()
    downloader.ingest_request.return_value = MagicMock()

    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            poll_timeout_ms=100,
        ),
        engine=engine,
        downloader=downloader,
        context=context,
    )
    wait_for_bind(command_addr)
    thread = _start_worker_thread(worker)

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=5000),
        context=context,
    )
    request = ForecastRequest(
        request_id="req-fail",
        preset="era5_pretrained",
        steps=1,
        valid_time="2024-06-01T06:00:00",
    )

    with pytest.raises(SchedulerError, match="prepare failed"):
        client.forecast(request)

    engine.release_gpu.assert_called_with(move_model_to_cpu=True)

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_worker_rejects_preset_mismatch(
    zmq_addresses: tuple[str, str, zmq.Context],
    tmp_path: Path,
) -> None:
    command_addr, event_addr, context = zmq_addresses
    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            poll_timeout_ms=100,
        ),
        engine=_build_mock_engine(tmp_path),
        downloader=MagicMock(),
        context=context,
    )
    wait_for_bind(command_addr)
    thread = _start_worker_thread(worker)

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=5000),
        context=context,
    )
    request = ForecastRequest(
        request_id="req-bad",
        preset="hres_0.1",
        steps=1,
        valid_time="2024-06-01T06:00:00",
    )
    with pytest.raises(SchedulerError, match="does not match"):
        client.forecast(request)

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()


def test_client_health(zmq_addresses: tuple[str, str, zmq.Context], tmp_path: Path) -> None:
    command_addr, event_addr, context = zmq_addresses
    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="small_pretrained",
            asset_root=tmp_path,
            command_addr=command_addr,
            event_addr=event_addr,
            poll_timeout_ms=100,
        ),
        engine=_build_mock_engine(tmp_path),
        downloader=MagicMock(),
        context=context,
    )
    wait_for_bind(command_addr)
    thread = _start_worker_thread(worker)

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=5000),
        context=context,
    )
    health = client.health()
    assert health.kind == "health"
    assert health.worker_preset == "small_pretrained"

    client.shutdown_worker()
    thread.join(timeout=5.0)
    client.close()
