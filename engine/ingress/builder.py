from __future__ import annotations

import numpy as np
import torch

from engine.core.config import SourceProfile
from engine.ingress.time import TimeHistoryPolicy


class TimeHistoryBuilder:
    def __init__(self, profile: SourceProfile) -> None:
        self._profile = profile
        self._policy = TimeHistoryPolicy(profile.time_policy)

    def build_surf_history(self, values: np.ndarray) -> torch.Tensor:
        pair = self._policy.select_pair(values)
        tensor = torch.from_numpy(pair)[None]
        if self._profile.flip_lat:
            tensor = tensor[..., ::-1, :].contiguous()
        return tensor

    def build_atmos_history(self, values: np.ndarray) -> torch.Tensor:
        pair = self._policy.select_pair(values)
        tensor = torch.from_numpy(pair)[None]
        if self._profile.flip_lat:
            tensor = tensor[..., ::-1, :].contiguous()
        return tensor
