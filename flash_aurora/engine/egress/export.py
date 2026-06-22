from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora import Batch

from flash_aurora.engine.egress.naming import PredictionNaming
from flash_aurora.engine.egress.offload import CpuOffloader
from flash_aurora.engine.egress.serialize import BatchExporter


class RolloutExporter:
    def __init__(
        self,
        export_dir: Path,
        naming: PredictionNaming | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._naming = naming or PredictionNaming()
        self._writer = BatchExporter(export_dir)

    def write_step(self, step_index: int, batch: Batch) -> Path:
        path = self._naming.path(self._export_dir, step_index)
        cpu_batch = CpuOffloader.to_cpu(batch)
        self._writer.write_netcdf(cpu_batch, path)
        return path
