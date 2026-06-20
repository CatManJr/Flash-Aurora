from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aurora import Batch, Metadata

from engine.core.trusted_pickle import load_trusted_pickle, resolve_trusted_path
from engine.core.netcdf_codec import read_batch_netcdf


class BatchDeserializer:
    @staticmethod
    def from_pickle(path: Path, *, allowed_roots: tuple[Path, ...]) -> Batch:
        payload = load_trusted_pickle(path, allowed_roots)
        if isinstance(payload, Batch):
            return payload
        if isinstance(payload, dict):
            return BatchDeserializer._batch_from_saved_dict(payload)
        raise TypeError(f"Unsupported pickle payload type: {type(payload)!r}")

    @staticmethod
    def _batch_from_saved_dict(payload: dict) -> Batch:
        metadata = payload["metadata"]
        atmos_levels = metadata["atmos_levels"]
        if isinstance(atmos_levels, list):
            atmos_levels = tuple(atmos_levels)
        times = metadata["time"]
        if isinstance(times, list):
            times = tuple(times)
        return Batch(
            surf_vars={k: torch.from_numpy(np.asarray(v)) for k, v in payload["surf_vars"].items()},
            static_vars={
                k: torch.from_numpy(np.asarray(v))
                for k, v in payload.get("static_vars", {}).items()
            },
            atmos_vars={
                k: torch.from_numpy(np.asarray(v)) for k, v in payload["atmos_vars"].items()
            },
            metadata=Metadata(
                lat=torch.from_numpy(np.asarray(metadata["lat"])),
                lon=torch.from_numpy(np.asarray(metadata["lon"])),
                time=times,
                atmos_levels=atmos_levels,
            ),
        )

    @staticmethod
    def from_netcdf(path: Path, *, allowed_roots: tuple[Path, ...]) -> Batch:
        resolve_trusted_path(path, allowed_roots)
        return read_batch_netcdf(path)
