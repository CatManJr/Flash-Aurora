"""Step-wise IO backends modelled on Earth2Studio ``IOBackend`` / ``AsyncZarrBackend``.

Earth-2 pattern:
- ``write`` accepts tensors and returns immediately when ``blocking=False``.
- ``close()`` drains in-flight writes before shutdown.
- Callers must hand off **owned** CPU buffers that outlive the async write.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

from flash_aurora.models.aurora import Batch

from flash_aurora.engine.egress.naming import PredictionNaming
from flash_aurora.engine.egress.offload import owned_cpu_copy
from flash_aurora.engine.egress.serialize import BatchExporter


@runtime_checkable
class StepIOBackend(Protocol):
    """Write one rollout step to storage (NetCDF file per step today)."""

    def write_step(self, step_index: int, batch: Batch) -> Path:
        """Queue or perform the write for ``step_index``."""
        ...

    def close(self) -> None:
        """Block until all in-flight writes finish."""
        ...


class NetCDFStepBackend:
    """Blocking NetCDF writer — Earth2 ``blocking=True`` semantics."""

    def __init__(
        self,
        export_dir: Path,
        naming: PredictionNaming | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._naming = naming or PredictionNaming()
        self._writer = BatchExporter(export_dir)

    def write_step(self, step_index: int, batch: Batch) -> Path:
        return self.write_owned_step(step_index, owned_cpu_copy(batch))

    def write_owned_step(self, step_index: int, batch: Batch) -> Path:
        path = self._naming.path(self._export_dir, step_index)
        self._writer.write_netcdf(batch, path)
        return path

    def close(self) -> None:
        return None


class AsyncNetCDFStepBackend:
    """Non-blocking step NetCDF writer — Earth2 ``AsyncZarrBackend`` semantics.

    Parameters mirror Earth2Studio:
    - ``blocking=False``: enqueue disk writes on a thread pool.
    - ``pool_size``: worker threads (Earth2 default 8; we default 2 for NetCDF).
    - ``max_inflight``: throttle queued writes (Earth2 ``_limit_pool_size``).
    """

    def __init__(
        self,
        export_dir: Path,
        naming: PredictionNaming | None = None,
        *,
        blocking: bool = False,
        pool_size: int = 2,
        max_inflight: int | None = None,
        trace: list[dict[str, float | int | str]] | None = None,
        retain_steps: int | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._naming = naming or PredictionNaming()
        self._writer = BatchExporter(export_dir)
        self._blocking = blocking
        self._pool_size = max(1, pool_size)
        self._max_inflight = max_inflight if max_inflight is not None else max(0, self._pool_size - 1)
        self._trace = trace
        self._retain_steps = retain_steps
        self._executor: ThreadPoolExecutor | None = None
        self._pending: list[Future[None]] = []
        if not blocking:
            self._executor = ThreadPoolExecutor(
                max_workers=self._pool_size,
                thread_name_prefix="aurora-io",
            )

    def write_step(self, step_index: int, batch: Batch) -> Path:
        return self.write_owned_step(step_index, owned_cpu_copy(batch))

    def write_owned_step(self, step_index: int, batch: Batch) -> Path:
        path = self._naming.path(self._export_dir, step_index)
        if self._blocking:
            self._write_and_record(step_index, batch, path)
            return path

        assert self._executor is not None
        self._limit_inflight()
        future = self._executor.submit(self._write_and_record, step_index, batch, path)
        self._pending.append(future)
        return path

    def _write_and_record(self, step_index: int, batch: Batch, path: Path) -> None:
        started_s = time.perf_counter()
        self._writer.write_netcdf(batch, path)
        finished_s = time.perf_counter()
        if self._trace is not None:
            self._trace.append(
                {"kind": "export", "step": step_index, "start_s": started_s, "end_s": finished_s}
            )
        self._drop_stale(step_index)

    def _drop_stale(self, step_index: int) -> None:
        """Keep only the newest files so a long rollout does not fill the disk."""
        if self._retain_steps is None:
            return
        stale_index = step_index - self._retain_steps
        if stale_index < 0:
            return
        self._naming.path(self._export_dir, stale_index).unlink(missing_ok=True)

    def _limit_inflight(self) -> None:
        while len(self._pending) > self._max_inflight:
            oldest = self._pending.pop(0)
            oldest.result()

    def close(self) -> None:
        for future in self._pending:
            future.result()
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> AsyncNetCDFStepBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if self._executor is None or not self._pending:
            return
        try:
            self.close()
        except Exception:
            pass
