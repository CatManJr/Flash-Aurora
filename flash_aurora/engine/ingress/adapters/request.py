from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class IngestRequest:
    """Minimal ingest request shared by all docs example adapters."""

    valid_time: datetime
    cache_dir: Path | None = None
    raw_paths: dict[str, Path] = field(default_factory=dict)
    time_index: int = 1
