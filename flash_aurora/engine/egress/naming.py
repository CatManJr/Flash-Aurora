from __future__ import annotations

from pathlib import Path


class PredictionNaming:
    def __init__(self, prefix: str = "prediction", suffix: str = ".nc") -> None:
        self._prefix = prefix
        self._suffix = suffix

    def filename(self, step_index: int) -> str:
        return f"{self._prefix}-{step_index:03d}{self._suffix}"

    def path(self, export_dir: Path, step_index: int) -> Path:
        return export_dir / self.filename(step_index)

    def regional_dir(self, export_dir: Path, region: str) -> Path:
        return export_dir if not region else export_dir / region

    def regional_path(self, export_dir: Path, region: str, step_index: int) -> Path:
        return self.regional_dir(export_dir, region) / self.filename(step_index)

    def geotiff_filename(self, step_index: int, variable: str) -> str:
        return f"{self._prefix}-{step_index:03d}-{variable}.tif"

    def geotiff_path(self, export_dir: Path, step_index: int, variable: str) -> Path:
        return export_dir / self.geotiff_filename(step_index, variable)

    def regional_geotiff_path(
        self,
        export_dir: Path,
        region: str,
        step_index: int,
        variable: str,
    ) -> Path:
        return self.regional_dir(export_dir, region) / self.geotiff_filename(step_index, variable)

    def iter_paths(self, export_dir: Path, count: int) -> list[Path]:
        return [self.path(export_dir, index) for index in range(count)]
