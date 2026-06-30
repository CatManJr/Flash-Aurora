"""Ensure checkpoints and ingress cache for distributed rollout benchmarks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from _preset_ic import (
    _DEFAULT_VALID_TIME,
    checkpoint_path,
    preset_engine_config,
)
from flash_aurora.engine.core.hub import HF_MIRROR_ENDPOINT
from flash_aurora.engine.core.paths import AssetStore
from flash_aurora.engine.ingress.download import DataDownloader
from flash_aurora.engine.ingress.download.layout import cache_subdir
from flash_aurora.hub import apply_hub_endpoint


def preset_assets_ready(preset: str, asset_root: Path) -> tuple[bool, str]:
    root = asset_root.expanduser().resolve()
    if preset == "era5_pretrained":
        from _pretrained_era5 import _CHECKPOINT_NAME  # noqa: PLC0415

        ckpt = root / _CHECKPOINT_NAME
        if not ckpt.is_file():
            return False, f"missing checkpoint: {ckpt}"
        config = preset_engine_config(preset, root)
        cache = root / cache_subdir(config.source)
        missing = DataDownloader(config).missing(_DEFAULT_VALID_TIME[preset], cache_dir=cache)
        if missing:
            return False, f"missing ingress under {cache}: {missing}"
        return True, ""

    config = preset_engine_config(preset, root)
    ckpt = checkpoint_path(config, root)
    if not ckpt.is_file():
        return False, f"missing checkpoint: {ckpt}"
    static_name = config.variant.static_pickle
    if static_name:
        static_path = root / static_name
        if not static_path.is_file():
            return False, f"missing static pickle: {static_path}"
    cache = root / cache_subdir(config.source)
    missing = DataDownloader(config).missing(_DEFAULT_VALID_TIME[preset], cache_dir=cache)
    if missing:
        return False, f"missing ingress under {cache}: {missing}"
    return True, ""


def ensure_preset_assets(
    preset: str,
    asset_root: Path,
    *,
    hf_mirror: bool,
    prompt: bool,
    verbose: bool = True,
    download_workers: int | None = None,
) -> None:
    root = asset_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if preset == "era5_pretrained":
        from _pretrained_era5 import ensure_pretrained_assets  # noqa: PLC0415

        ensure_pretrained_assets(
            root,
            hf_mirror=hf_mirror,
            download_era5=True,
            prompt=prompt,
            verbose=verbose,
            download_workers=download_workers,
        )
        return

    config = replace(
        preset_engine_config(preset, root),
        allow_hub_download=True,
        hf_endpoint=HF_MIRROR_ENDPOINT if hf_mirror else None,
    )
    if hf_mirror:
        apply_hub_endpoint(HF_MIRROR_ENDPOINT)

    store = AssetStore(root=root)
    ckpt = checkpoint_path(config, root)
    if not ckpt.is_file():
        if verbose:
            print(f"[download] {preset} checkpoint -> {ckpt}", flush=True)
        fetched = store.fetch_hub_file(
            config.variant.checkpoint_filename,
            repo=config.variant.hf_repo,
            allow_download=True,
            explicit=root,
            hub=config.hub_download_options(),
        )
        if verbose:
            print(f"[download] checkpoint ready: {fetched}", flush=True)

    static_name = config.variant.static_pickle
    if static_name:
        static_path = root / static_name
        if not static_path.is_file():
            if verbose:
                print(f"[download] {preset} static -> {static_path}", flush=True)
            fetched = store.fetch_hub_file(
                static_name,
                repo=config.variant.hf_repo,
                allow_download=True,
                explicit=root,
                hub=config.hub_download_options(),
            )
            if verbose:
                print(f"[download] static ready: {fetched}", flush=True)

    downloader = DataDownloader(config, workers=download_workers)
    valid_time = _DEFAULT_VALID_TIME[preset]
    cache = root / cache_subdir(config.source)
    missing = downloader.missing(valid_time, cache_dir=cache)
    if missing:
        if verbose:
            print(
                f"[download] {preset} ingress ({', '.join(missing)}) -> {cache} "
                f"({downloader.download_workers} workers)",
                flush=True,
            )
        result = downloader.ensure(
            valid_time,
            cache_dir=cache,
            prompt=prompt,
            workers=download_workers,
        )
        if verbose:
            print(
                f"[download] ingress ready: downloaded={result.downloaded} "
                f"skipped={result.skipped}",
                flush=True,
            )

    ready, reason = preset_assets_ready(preset, root)
    if not ready:
        raise FileNotFoundError(reason)
