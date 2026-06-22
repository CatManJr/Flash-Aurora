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

    def iter_paths(self, export_dir: Path, count: int) -> list[Path]:
        return [self.path(export_dir, index) for index in range(count)]
