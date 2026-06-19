from __future__ import annotations

from aurora.model.aurora import Aurora


class GraphPool:
    """Placeholder for CUDA graph capture reuse."""

    def __init__(self) -> None:
        self._captured: dict[str, object] = {}

    def warmup(self, model: Aurora) -> None:
        return None
