from __future__ import annotations

import pickle
from pathlib import Path

from aurora import Batch


class BatchDeserializer:
    @staticmethod
    def from_pickle(path: Path) -> Batch:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, Batch):
            return payload
        raise TypeError(f"Expected Batch in pickle, got {type(payload)!r}")

    @staticmethod
    def from_netcdf(path: Path) -> Batch:
        return Batch.from_netcdf(str(path))
