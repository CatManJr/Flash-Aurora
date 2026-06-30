from __future__ import annotations

import os
from dataclasses import dataclass


def default_download_workers() -> int:
    """Default thread count from ``FLASH_AURORA_DOWNLOAD_WORKERS`` (fallback: 8)."""
    raw = os.environ.get("FLASH_AURORA_DOWNLOAD_WORKERS", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(1, value)


def resolve_download_workers(workers: int | None) -> int:
    if workers is None:
        return default_download_workers()
    return max(1, workers)


@dataclass(frozen=True)
class DownloadOptions:
    """Controls parallel ingress downloads."""

    workers: int = 8

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("download workers must be >= 1")

    @classmethod
    def resolve(cls, workers: int | None = None) -> DownloadOptions:
        return cls(workers=resolve_download_workers(workers))
