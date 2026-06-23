"""Disk cache for deterministic ingress: replay the exact post-processed ``Batch``.

Optional via ``EngineConfig.ic_cache``. Used when the same initial field (same day /
same NetCDF file) is prepared multiple times.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from flash_aurora.aurora import Batch, Metadata

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.adapters.base import resolve_cache_dir
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.download.layout import cache_subdir, day_token, expected_paths

IC_CACHE_VERSION = 2
IC_CACHE_DIRNAME = ".ic-cache"


@dataclass(frozen=True)
class IcCacheFingerprint:
    """Inputs that must match for a cached batch to be valid."""

    version: int
    variant: str
    source: str
    day: str
    time_index: int
    regrid_res: float | None
    levels: tuple[int | float, ...]
    input_digests: tuple[tuple[str, str], ...]
    static_pickle_digest: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "variant": self.variant,
            "source": self.source,
            "day": self.day,
            "time_index": self.time_index,
            "regrid_res": self.regrid_res,
            "levels": list(self.levels),
            "input_digests": [list(item) for item in self.input_digests],
            "static_pickle_digest": self.static_pickle_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IcCacheFingerprint:
        raw_levels = payload["levels"]
        raw_digests = payload["input_digests"]
        regrid = payload.get("regrid_res")
        static = payload.get("static_pickle_digest")
        return cls(
            version=int(payload["version"]),
            variant=str(payload["variant"]),
            source=str(payload["source"]),
            day=str(payload["day"]),
            time_index=int(payload["time_index"]),
            regrid_res=None if regrid is None else float(regrid),
            levels=tuple(int(x) if float(x).is_integer() else float(x) for x in raw_levels),  # type: ignore[arg-type]
            input_digests=tuple((str(row[0]), str(row[1])) for row in raw_digests),  # type: ignore[misc]
            static_pickle_digest=None if static is None else str(static),
        )


@dataclass(frozen=True)
class NetcdfIcCacheFingerprint:
    version: int
    variant: str
    netcdf_digest: str
    levels: tuple[int | float, ...]
    static_pickle_digest: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "variant": self.variant,
            "netcdf_digest": self.netcdf_digest,
            "levels": list(self.levels),
            "static_pickle_digest": self.static_pickle_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NetcdfIcCacheFingerprint:
        raw_levels = payload["levels"]
        static = payload.get("static_pickle_digest")
        return cls(
            version=int(payload["version"]),
            variant=str(payload["variant"]),
            netcdf_digest=str(payload["netcdf_digest"]),
            levels=tuple(int(x) if float(x).is_integer() else float(x) for x in raw_levels),  # type: ignore[arg-type]
            static_pickle_digest=None if static is None else str(static),
        )


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _variant_label(config: EngineConfig) -> str:
    return config.preset_name or config.variant.name


def _static_pickle_digest(config: EngineConfig) -> str | None:
    assets = AssetStore(root=config.asset_root)
    path = assets.join(
        config.variant.static_pickle,
        explicit=config.asset_root,
        user_cwd=config.user_cwd,
    )
    if not path.is_file():
        return f"missing:{config.variant.static_pickle}"
    return sha256_file(path)


def resolve_source_input_paths(request: IngestRequest, config: EngineConfig) -> dict[str, Path]:
    if request.raw_paths:
        return {
            key: Path(value).expanduser().resolve()
            for key, value in request.raw_paths.items()
        }
    subdir = cache_subdir(config.source)
    cache_dir = resolve_cache_dir(request, config, subdir)
    return expected_paths(config.source, request.valid_time, cache_dir)


def fingerprint_for_source(request: IngestRequest, config: EngineConfig) -> IcCacheFingerprint:
    paths = resolve_source_input_paths(request, config)
    digests = tuple(
        (role, sha256_file(path))
        for role, path in sorted(paths.items())
        if path.is_file()
    )
    uses_hf_static = "static" not in paths
    return IcCacheFingerprint(
        version=IC_CACHE_VERSION,
        variant=_variant_label(config),
        source=config.source.name,
        day=day_token(request.valid_time),
        time_index=request.time_index,
        regrid_res=config.source.regrid_res,
        levels=config.variant.levels,
        input_digests=digests,
        static_pickle_digest=_static_pickle_digest(config) if uses_hf_static else None,
    )


def fingerprint_for_netcdf(path: Path, config: EngineConfig) -> NetcdfIcCacheFingerprint:
    resolved = path.expanduser().resolve()
    return NetcdfIcCacheFingerprint(
        version=IC_CACHE_VERSION,
        variant=_variant_label(config),
        netcdf_digest=sha256_file(resolved),
        levels=config.variant.levels,
        static_pickle_digest=_static_pickle_digest(config),
    )


def source_cache_location(request: IngestRequest, config: EngineConfig) -> tuple[Path, str]:
    if request.raw_paths:
        first = next(iter(request.raw_paths.values()))
        cache_root = Path(first).expanduser().resolve().parent
    else:
        subdir = cache_subdir(config.source)
        cache_root = resolve_cache_dir(request, config, subdir)
    day = day_token(request.valid_time)
    cache_id = f"{_variant_label(config)}-{day}-t{request.time_index}"
    return cache_root, cache_id


def netcdf_cache_location(path: Path, config: EngineConfig) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    cache_id = f"{_variant_label(config)}-{resolved.stem}"
    return resolved.parent, cache_id


def ic_cache_paths(cache_root: Path, cache_id: str) -> tuple[Path, Path]:
    root = Path(cache_root) / IC_CACHE_DIRNAME
    stem = root / cache_id
    return stem.with_suffix(".pt"), stem.with_suffix(".meta.json")


def _batch_to_state(batch: Batch) -> dict[str, object]:
    return {
        "surf_vars": {k: v.detach().cpu() for k, v in batch.surf_vars.items()},
        "static_vars": {k: v.detach().cpu() for k, v in batch.static_vars.items()},
        "atmos_vars": {k: v.detach().cpu() for k, v in batch.atmos_vars.items()},
        "metadata": {
            "lat": batch.metadata.lat.detach().cpu(),
            "lon": batch.metadata.lon.detach().cpu(),
            "time": batch.metadata.time,
            "atmos_levels": batch.metadata.atmos_levels,
            "rollout_step": batch.metadata.rollout_step,
        },
    }


def _batch_from_state(payload: dict[str, object]) -> Batch:
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    atmos_levels = metadata["atmos_levels"]
    if isinstance(atmos_levels, list):
        atmos_levels = tuple(atmos_levels)
    times = metadata["time"]
    if isinstance(times, list):
        times = tuple(times)
    surf_vars = payload["surf_vars"]
    static_vars = payload["static_vars"]
    atmos_vars = payload["atmos_vars"]
    assert isinstance(surf_vars, dict)
    assert isinstance(static_vars, dict)
    assert isinstance(atmos_vars, dict)
    return Batch(
        surf_vars={str(k): v for k, v in surf_vars.items()},  # type: ignore[misc]
        static_vars={str(k): v for k, v in static_vars.items()},  # type: ignore[misc]
        atmos_vars={str(k): v for k, v in atmos_vars.items()},  # type: ignore[misc]
        metadata=Metadata(
            lat=metadata["lat"],  # type: ignore[arg-type]
            lon=metadata["lon"],  # type: ignore[arg-type]
            time=times,  # type: ignore[arg-type]
            atmos_levels=atmos_levels,  # type: ignore[arg-type]
            rollout_step=int(metadata.get("rollout_step", 0)),
        ),
    )


def _load_cached(
    cache_root: Path,
    cache_id: str,
    fingerprint: IcCacheFingerprint | NetcdfIcCacheFingerprint,
) -> Batch | None:
    cache_pt, cache_meta = ic_cache_paths(cache_root, cache_id)
    if not cache_pt.is_file() or not cache_meta.is_file():
        return None
    try:
        stored_payload = json.loads(cache_meta.read_text())
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if isinstance(fingerprint, IcCacheFingerprint):
        try:
            stored = IcCacheFingerprint.from_dict(stored_payload)
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            stored = NetcdfIcCacheFingerprint.from_dict(stored_payload)
        except (KeyError, TypeError, ValueError):
            return None

    if stored != fingerprint:
        return None

    state = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return None
    return _batch_from_state(state)


def _store_cached(
    cache_root: Path,
    cache_id: str,
    fingerprint: IcCacheFingerprint | NetcdfIcCacheFingerprint,
    batch: Batch,
) -> None:
    cache_pt, cache_meta = ic_cache_paths(cache_root, cache_id)
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    pt_tmp = cache_pt.with_suffix(".pt.tmp")
    meta_tmp = cache_meta.with_suffix(".meta.json.tmp")
    try:
        torch.save(_batch_to_state(batch), pt_tmp)
        meta_tmp.write_text(json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True))
        pt_tmp.replace(cache_pt)
        meta_tmp.replace(cache_meta)
    finally:
        pt_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)


def load_cached_source_batch(
    request: IngestRequest,
    config: EngineConfig,
) -> Batch | None:
    cache_root, cache_id = source_cache_location(request, config)
    fingerprint = fingerprint_for_source(request, config)
    return _load_cached(cache_root, cache_id, fingerprint)


def store_cached_source_batch(
    request: IngestRequest,
    config: EngineConfig,
    batch: Batch,
) -> None:
    cache_root, cache_id = source_cache_location(request, config)
    fingerprint = fingerprint_for_source(request, config)
    _store_cached(cache_root, cache_id, fingerprint, batch)


def load_cached_netcdf_batch(path: Path, config: EngineConfig) -> Batch | None:
    cache_root, cache_id = netcdf_cache_location(path, config)
    fingerprint = fingerprint_for_netcdf(path, config)
    return _load_cached(cache_root, cache_id, fingerprint)


def store_cached_netcdf_batch(path: Path, config: EngineConfig, batch: Batch) -> None:
    cache_root, cache_id = netcdf_cache_location(path, config)
    fingerprint = fingerprint_for_netcdf(path, config)
    _store_cached(cache_root, cache_id, fingerprint, batch)
