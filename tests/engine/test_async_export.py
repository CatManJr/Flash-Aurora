from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flash_aurora.engine.egress.export import AsyncRolloutExporter, PipelineRolloutExporter, RolloutExporter
from flash_aurora.engine.egress.io_backend import AsyncNetCDFStepBackend, NetCDFStepBackend
from flash_aurora.engine.egress.offload import owned_cpu_copy
from tests.engine.test_netcdf_codec import _sample_batch


def test_owned_cpu_copy_is_detached_cpu() -> None:
    batch = _sample_batch()
    owned = owned_cpu_copy(batch)
    for group in (owned.surf_vars, owned.static_vars, owned.atmos_vars):
        for tensor in group.values():
            assert not tensor.is_cuda
            assert tensor.data_ptr() != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pipeline_exporter_writes_with_egress_stream(tmp_path: Path) -> None:
    batch = _sample_batch().to("cuda")
    with PipelineRolloutExporter.async_netcdf(tmp_path, pool_size=2) as exporter:
        path = exporter.write_step(0, batch)
    assert path.is_file()


def test_async_rollout_exporter_writes_all_steps(tmp_path: Path) -> None:
    batch = _sample_batch()
    sync = RolloutExporter(tmp_path / "sync")
    async_exporter = AsyncRolloutExporter(tmp_path / "async", pool_size=2)

    sync_paths = [sync.write_step(i, batch) for i in range(3)]
    async_paths = [async_exporter.write_step(i, batch) for i in range(3)]
    async_exporter.close()

    assert len(sync_paths) == 3
    assert len(async_paths) == 3
    for path in async_paths:
        assert path.is_file()


def test_async_netcdf_backend_throttles_inflight(tmp_path: Path) -> None:
    batch = _sample_batch()
    backend = AsyncNetCDFStepBackend(tmp_path, blocking=False, pool_size=2, max_inflight=1)
    backend.write_step(0, batch)
    backend.write_step(1, batch)
    backend.close()


def test_async_rollout_exporter_propagates_write_errors(tmp_path: Path) -> None:
    batch = _sample_batch()
    backend = AsyncNetCDFStepBackend(tmp_path, blocking=False)
    backend._writer.write_netcdf = lambda _b, _p: (_ for _ in ()).throw(OSError("disk full"))  # type: ignore[method-assign]
    exporter = PipelineRolloutExporter(backend)

    exporter.write_step(0, batch)
    with pytest.raises(OSError, match="disk full"):
        exporter.close()
