from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora import Batch

from flash_aurora.engine.egress.io_backend import (
    AsyncNetCDFStepBackend,
    NetCDFStepBackend,
    StepIOBackend,
)
from flash_aurora.engine.egress.naming import PredictionNaming
from flash_aurora.engine.egress.offload import EgressOffloader


class RolloutExporter:
    """Blocking step exporter (legacy API)."""

    def __init__(
        self,
        export_dir: Path,
        naming: PredictionNaming | None = None,
    ) -> None:
        self._backend = NetCDFStepBackend(export_dir, naming=naming)

    def write_step(self, step_index: int, batch: Batch) -> Path:
        return self._backend.write_step(step_index, batch)

    def close(self) -> None:
        self._backend.close()


class AsyncRolloutExporter:
    """Non-blocking step exporter — Earth2 ``AsyncZarrBackend(blocking=False)`` for NetCDF."""

    def __init__(
        self,
        export_dir: Path,
        naming: PredictionNaming | None = None,
        *,
        pool_size: int = 2,
        max_inflight: int | None = None,
    ) -> None:
        self._backend = AsyncNetCDFStepBackend(
            export_dir,
            naming=naming,
            blocking=False,
            pool_size=pool_size,
            max_inflight=max_inflight,
        )

    def write_step(self, step_index: int, batch: Batch) -> Path:
        return self._backend.write_step(step_index, batch)

    def flush(self) -> None:
        self._backend.close()

    def shutdown(self) -> None:
        self._backend.close()

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> AsyncRolloutExporter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PipelineRolloutExporter:
    """Earth-2 style pipeline: egress-stream D2H + async IO backend."""

    def __init__(
        self,
        backend: StepIOBackend,
        *,
        offloader: EgressOffloader | None = None,
    ) -> None:
        self._backend = backend
        self._offloader = offloader or EgressOffloader()

    @classmethod
    def async_netcdf(
        cls,
        export_dir: Path,
        naming: PredictionNaming | None = None,
        *,
        pool_size: int = 2,
        max_inflight: int | None = None,
        use_egress_stream: bool = True,
    ) -> PipelineRolloutExporter:
        backend = AsyncNetCDFStepBackend(
            export_dir,
            naming=naming,
            blocking=False,
            pool_size=pool_size,
            max_inflight=max_inflight,
        )
        offloader = EgressOffloader(use_stream=use_egress_stream)
        return cls(backend, offloader=offloader)

    def write_step(self, step_index: int, batch: Batch) -> Path:
        cpu_batch = self._offloader.to_owned_cpu(batch)
        if hasattr(self._backend, "write_owned_step"):
            return self._backend.write_owned_step(step_index, cpu_batch)
        return self._backend.write_step(step_index, cpu_batch)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> PipelineRolloutExporter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
