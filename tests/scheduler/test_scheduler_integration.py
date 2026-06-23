from __future__ import annotations

import threading
from pathlib import Path

import pytest

from flash_aurora.engine.core.netcdf_codec import write_batch_netcdf
from flash_aurora.scheduler.client import ForecastClient, ForecastClientConfig
from flash_aurora.scheduler.protocol import ForecastRequest
from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig, wait_for_bind


@pytest.mark.integration
@pytest.mark.gpu
def test_worker_rollout_export_over_zmq(
    engine_config_offline,
    asset_root: Path,
    tmp_path: Path,
) -> None:
    from flash_aurora.engine.ingress.build_ic import InitialConditionBuilder

    builder = InitialConditionBuilder(engine_config_offline)
    reference_batch = builder.from_pickle("aurora-0.25-small-pretrained-test-input.pickle")
    netcdf_path = tmp_path / "input.nc"
    write_batch_netcdf(reference_batch, netcdf_path)

    command_addr = f"ipc://{tmp_path / 'commands.ipc'}"
    event_addr = f"ipc://{tmp_path / 'events.ipc'}"
    export_dir = tmp_path / "output"

    config = ForecastWorkerConfig(
        preset="small_pretrained",
        asset_root=asset_root,
        command_addr=command_addr,
        event_addr=event_addr,
        export_dir=export_dir,
        poll_timeout_ms=100,
    )
    worker = ForecastWorker(config)
    wait_for_bind(command_addr)

    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()

    client = ForecastClient(
        ForecastClientConfig(command_addr=command_addr, event_addr=event_addr, recv_timeout_ms=600_000),
    )
    request = ForecastRequest(
        request_id="integration-1",
        preset="small_pretrained",
        steps=1,
        netcdf_path=str(netcdf_path),
        export_dir=str(export_dir),
    )
    events = client.forecast(request)
    assert events[-1].kind == "completed"
    step_events = [event for event in events if event.kind == "step"]
    assert len(step_events) == 1
    assert Path(step_events[0].export_path).is_file()

    client.shutdown_worker()
    thread.join(timeout=30.0)
    client.close()
