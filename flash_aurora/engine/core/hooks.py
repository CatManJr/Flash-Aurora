from __future__ import annotations

from typing import Protocol

from flash_aurora.aurora import Batch


class RolloutObserver(Protocol):
    def on_step(self, step: int, prediction: Batch) -> None:
        ...
