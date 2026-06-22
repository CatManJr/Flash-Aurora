from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora import Batch

from flash_aurora.engine.core.netcdf_codec import write_batch_netcdf


class BatchExporter:
    def __init__(self, export_dir: Path) -> None:
        self._export_dir = export_dir
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def write_netcdf(self, batch: Batch, path: Path) -> None:
        write_batch_netcdf(batch, path)
