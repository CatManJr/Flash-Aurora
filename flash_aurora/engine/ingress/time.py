from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class TimeHistoryPolicy:
    name: str
    time_index: int = 1

    def select_pair(self, values: np.ndarray) -> np.ndarray:
        if self.name == "first_two":
            return values[:2]
        if self.name == "pair":
            idx = self.time_index
            return values[[idx - 1, idx]]
        raise ValueError(f"Unsupported time policy: {self.name}")

    def select_times(self, times: tuple[datetime, ...], batch_size: int) -> tuple[datetime, ...]:
        if batch_size == 1:
            if self.name == "first_two":
                return (times[1],)
            if self.name == "pair":
                return (times[self.time_index],)
        return times[:batch_size]
