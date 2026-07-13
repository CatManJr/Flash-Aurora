from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from flash_aurora.models.aurora import Batch


@dataclass(frozen=True)
class ForecastStep:
    step_index: int
    valid_time: datetime
    batch: Batch

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        step_index: int,
        base_time: datetime,
        timestep_hours: int,
    ) -> ForecastStep:
        times = getattr(batch.metadata, "time", None) or ()
        if times:
            valid_time = times[-1]
        else:
            valid_time = base_time + timedelta(hours=timestep_hours * (step_index + 1))
        return cls(step_index=step_index, valid_time=valid_time, batch=batch)
