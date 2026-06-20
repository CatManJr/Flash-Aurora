from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora import Batch, Metadata

NETCDF_ENGINE = "scipy"


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def batch_to_dataset(batch: Batch) -> xr.Dataset:
    return xr.Dataset(
        {
            **{
                f"surf_{k}": (("batch", "history", "latitude", "longitude"), _tensor_to_numpy(v))
                for k, v in batch.surf_vars.items()
            },
            **{
                f"static_{k}": (("latitude", "longitude"), _tensor_to_numpy(v))
                for k, v in batch.static_vars.items()
            },
            **{
                f"atmos_{k}": (
                    ("batch", "history", "level", "latitude", "longitude"),
                    _tensor_to_numpy(v),
                )
                for k, v in batch.atmos_vars.items()
            },
        },
        coords={
            "latitude": _tensor_to_numpy(batch.metadata.lat),
            "longitude": _tensor_to_numpy(batch.metadata.lon),
            "time": list(batch.metadata.time),
            "level": list(batch.metadata.atmos_levels),
            "rollout_step": batch.metadata.rollout_step,
        },
    )


def _numpy_array(values) -> np.ndarray:
    array = np.asarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return array


def dataset_to_batch(dataset: xr.Dataset) -> Batch:
    surf_vars: list[str] = []
    static_vars: list[str] = []
    atmos_vars: list[str] = []

    for name in dataset:
        if name.startswith("surf_"):
            surf_vars.append(name.removeprefix("surf_"))
        elif name.startswith("static_"):
            static_vars.append(name.removeprefix("static_"))
        elif name.startswith("atmos_"):
            atmos_vars.append(name.removeprefix("atmos_"))

    return Batch(
        surf_vars={k: torch.from_numpy(_numpy_array(dataset[f"surf_{k}"].values)) for k in surf_vars},
        static_vars={
            k: torch.from_numpy(_numpy_array(dataset[f"static_{k}"].values)) for k in static_vars
        },
        atmos_vars={k: torch.from_numpy(_numpy_array(dataset[f"atmos_{k}"].values)) for k in atmos_vars},
        metadata=Metadata(
            lat=torch.from_numpy(_numpy_array(dataset.latitude.values)),
            lon=torch.from_numpy(_numpy_array(dataset.longitude.values)),
            time=tuple(dataset.time.values.astype("datetime64[s]").tolist()),
            atmos_levels=tuple(dataset.level.values),
            rollout_step=int(dataset.rollout_step.values),
        ),
    )


def write_batch_netcdf(batch: Batch, path: str | Path) -> None:
    batch_to_dataset(batch).to_netcdf(path, engine=NETCDF_ENGINE)


def read_batch_netcdf(path: str | Path) -> Batch:
    dataset = xr.load_dataset(path, engine=NETCDF_ENGINE)
    try:
        return dataset_to_batch(dataset)
    finally:
        dataset.close()
