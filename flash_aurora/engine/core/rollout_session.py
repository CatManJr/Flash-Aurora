from __future__ import annotations

from typing import Generator, Iterable, Sequence

import torch
from flash_aurora.models.aurora import Batch
from flash_aurora.models.aurora.rollout import rollout as optimized_rollout
from flash_aurora.models.aurora_v1p5.rollout import rollout as v1p5_rollout

from flash_aurora.engine.core.hooks import RolloutObserver
from flash_aurora.engine.core.model_protocol import AuroraModel, model_uses_v1p5_rollout


class RolloutSession:
    """Runs multi-step rollout with optional observers."""

    def __init__(
        self,
        model: AuroraModel,
        observers: Iterable[RolloutObserver] | None = None,
    ) -> None:
        self._model = model
        self._observers = list(observers or [])
        self._uses_v1p5 = model_uses_v1p5_rollout(model)
        self._rollout = v1p5_rollout if self._uses_v1p5 else optimized_rollout

    def run(
        self,
        batch: Batch,
        steps: int,
        *,
        fine_lead_times: Sequence[float] | None = None,
        use_noise_accumulation: bool = True,
        apply_rollout_input_clipping: bool = True,
    ) -> Generator[Batch, None, None]:
        if fine_lead_times is not None and not self._uses_v1p5:
            raise ValueError("`fine_lead_times` requires an Aurora 1.5 model.")
        with torch.inference_mode():
            if self._uses_v1p5:
                stream = self._rollout(
                    self._model,
                    batch,
                    steps,
                    fine_lead_times=fine_lead_times,
                    use_noise_accumulation=use_noise_accumulation,
                    apply_rollout_input_clipping=apply_rollout_input_clipping,
                )
            else:
                stream = self._rollout(self._model, batch, steps)
            for step, pred in enumerate(stream):
                for observer in self._observers:
                    observer.on_step(step, pred)
                yield pred
