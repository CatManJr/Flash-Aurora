from __future__ import annotations

import torch


class StaticVarsCache:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, key: str) -> dict[str, torch.Tensor] | None:
        return self._cache.get(key)

    def put(self, key: str, static_vars: dict[str, torch.Tensor]) -> None:
        self._cache[key] = static_vars
