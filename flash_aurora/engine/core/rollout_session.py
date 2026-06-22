from __future__ import annotations

from typing import Generator, Iterable

import torch
from flash_aurora.aurora import Batch, rollout
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.hooks import RolloutObserver


class RolloutSession:
    """Runs multi-step rollout with optional observers."""

    def __init__(self, model: Aurora, observers: Iterable[RolloutObserver] | None = None) -> None:
        self._model = model
        self._observers = list(observers or [])

    def run(self, batch: Batch, steps: int) -> Generator[Batch, None, None]:
        with torch.inference_mode():
            for step, pred in enumerate(rollout(self._model, batch, steps)):
                for observer in self._observers:
                    observer.on_step(step, pred)
                yield pred
