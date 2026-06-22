from __future__ import annotations

import numpy as np
import torch

from flash_aurora.engine.core.config import SourceProfile
from flash_aurora.engine.ingress.time import TimeHistoryPolicy


class TimeHistoryBuilder:
    def __init__(self, profile: SourceProfile, *, time_index: int | None = None) -> None:
        self._profile = profile
        self._policy = TimeHistoryPolicy(
            profile.time_policy,
            time_index=1 if time_index is None else time_index,
        )

    def build_surf_history(self, values: np.ndarray) -> torch.Tensor:
        pair = self._policy.select_pair(values)
        array = np.ascontiguousarray(pair)[None]
        if self._profile.flip_lat:
            array = array[..., ::-1, :].copy()
        return torch.from_numpy(array)

    def build_atmos_history(self, values: np.ndarray) -> torch.Tensor:
        pair = self._policy.select_pair(values)
        array = np.ascontiguousarray(pair)[None]
        if self._profile.flip_lat:
            array = array[..., ::-1, :].copy()
        return torch.from_numpy(array)

    def build_wave_history(self, values: np.ndarray) -> torch.Tensor:
        pair = self._policy.select_pair(values)
        array = np.ascontiguousarray(pair)[None]
        if self._profile.flip_lat_wave:
            array = array[..., ::-1, :].copy()
        return torch.from_numpy(array)
