from __future__ import annotations

from pathlib import Path

from aurora import Batch


class BatchExporter:
    def __init__(self, export_dir: Path) -> None:
        self._export_dir = export_dir
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def write_netcdf(self, batch: Batch, path: Path) -> None:
        batch.to_netcdf(str(path))
