from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.core.asset_root import normalize_asset_root
from flash_aurora.engine.core.paths import AssetStore, normalize_user_path
from flash_aurora.engine.core.redaction import redact_text, safe_path, sanitize_exception
from flash_aurora.engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from flash_aurora.engine.ingress.adapters.request import IngestRequest
from flash_aurora.engine.ingress.download.backends import DownloadBackendError, DownloadOutcome, get_backend
from flash_aurora.engine.ingress.download.credentials import (
    DownloadCredentials,
    merge_credentials,
    prompt_ecmwf_credentials,
    use_download_credentials,
)
from flash_aurora.engine.ingress.download.layout import cache_subdir, expected_paths, missing_keys
from flash_aurora.engine.ingress.download.options import DownloadOptions, resolve_download_workers
from flash_aurora.engine.ingress.download.paths import ensure_directory, normalize_path


@dataclass(frozen=True)
class DownloadRequest:
    valid_time: datetime
    cache_dir: Path | None = None


@dataclass(frozen=True)
class DownloadResult:
    cache_dir: Path
    paths: dict[str, Path]
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return all(path.is_file() for path in self.paths.values())

    def __repr__(self) -> str:
        return (
            "DownloadResult("
            f"cache_dir={safe_path(self.cache_dir)!r}, "
            f"downloaded={self.downloaded!r}, "
            f"skipped={self.skipped!r}, "
            f"keys={tuple(self.paths)})"
        )


class DataDownloader:
    """Download and cache ingress data for a preset/source profile.

    Paths are resolved with ``pathlib`` and work on Linux, macOS, and Windows.
    Model checkpoints/static pickles still use ``AssetStore`` / Hugging Face.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        credentials: DownloadCredentials | None = None,
        workers: int | None = None,
    ) -> None:
        self.config = config
        self._credentials = credentials
        self._download_options = DownloadOptions.resolve(workers)
        if self.config.user_cwd is None:
            self.config.user_cwd = Path.cwd()

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        asset_root: Path | str | None = None,
        user_cwd: Path | str | None = None,
        presets: PresetRegistry | None = None,
        credentials: DownloadCredentials | None = None,
        cds_api_key: str | None = None,
        cds_api_url: str | None = None,
        ads_api_key: str | None = None,
        ads_api_url: str | None = None,
        ecmwf_api_key: str | None = None,
        ecmwf_api_url: str | None = None,
        ecmwf_email: str | None = None,
        workers: int | None = None,
    ) -> DataDownloader:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        user_cwd = normalize_path(user_cwd) if user_cwd is not None else Path.cwd()
        config.user_cwd = user_cwd
        config.asset_root = normalize_asset_root(asset_root, user_cwd=user_cwd)
        preset_credentials = merge_credentials(
            credentials,
            DownloadCredentials(
                cds_api_key=cds_api_key,
                cds_api_url=cds_api_url,
                ads_api_key=ads_api_key,
                ads_api_url=ads_api_url,
                ecmwf_api_key=ecmwf_api_key,
                ecmwf_api_url=ecmwf_api_url,
                ecmwf_email=ecmwf_email,
            ),
        )
        return cls(config, credentials=preset_credentials, workers=workers)

    @property
    def download_workers(self) -> int:
        return self._download_options.workers

    def with_workers(self, workers: int) -> DataDownloader:
        """Return a downloader that uses a different worker count."""
        return DataDownloader(
            self.config,
            credentials=self._credentials,
            workers=workers,
        )

    def _effective_credentials(
        self,
        credentials: DownloadCredentials | None = None,
        *,
        cds_api_key: str | None = None,
        cds_api_url: str | None = None,
        ads_api_key: str | None = None,
        ads_api_url: str | None = None,
        ecmwf_api_key: str | None = None,
        ecmwf_api_url: str | None = None,
        ecmwf_email: str | None = None,
        prompt: bool = False,
    ) -> DownloadCredentials:
        creds = merge_credentials(
            self._credentials,
            credentials,
            DownloadCredentials(
                cds_api_key=cds_api_key,
                cds_api_url=cds_api_url,
                ads_api_key=ads_api_key,
                ads_api_url=ads_api_url,
                ecmwf_api_key=ecmwf_api_key,
                ecmwf_api_url=ecmwf_api_url,
                ecmwf_email=ecmwf_email,
            ),
        )
        if prompt and self.config.source.name == "wb2_wam" and creds.ecmwf_settings() is None:
            creds = prompt_ecmwf_credentials(creds)
        if self.config.source.name == "wb2_wam" and creds.ecmwf_settings() is None:
            from flash_aurora.engine.ingress.download.mars import _mars_config_error

            raise _mars_config_error()
        return creds

    def resolve_cache_dir(self, request: DownloadRequest | None = None) -> Path:
        if request is not None and request.cache_dir is not None:
            return normalize_path(request.cache_dir)
        store = AssetStore(root=self.config.asset_root)
        root = store.resolve_root(self.config.asset_root, self.config.user_cwd)
        return root / cache_subdir(self.config.source)

    def expected_paths(
        self,
        valid_time: datetime,
        *,
        cache_dir: Path | None = None,
    ) -> dict[str, Path]:
        directory = normalize_path(cache_dir) if cache_dir is not None else self.resolve_cache_dir(
            DownloadRequest(valid_time=valid_time)
        )
        return expected_paths(self.config.source, valid_time, directory)

    def missing(
        self,
        valid_time: datetime,
        *,
        cache_dir: Path | None = None,
    ) -> tuple[str, ...]:
        directory = normalize_path(cache_dir) if cache_dir is not None else self.resolve_cache_dir(
            DownloadRequest(valid_time=valid_time)
        )
        return missing_keys(self.config.source, valid_time, directory)

    def ensure(
        self,
        valid_time: datetime,
        *,
        cache_dir: Path | None = None,
        credentials: DownloadCredentials | None = None,
        cds_api_key: str | None = None,
        cds_api_url: str | None = None,
        ads_api_key: str | None = None,
        ads_api_url: str | None = None,
        ecmwf_api_key: str | None = None,
        ecmwf_api_url: str | None = None,
        ecmwf_email: str | None = None,
        prompt: bool = False,
        workers: int | None = None,
    ) -> DownloadResult:
        directory = ensure_directory(
            normalize_path(cache_dir)
            if cache_dir is not None
            else self.resolve_cache_dir(DownloadRequest(valid_time=valid_time))
        )
        paths = expected_paths(self.config.source, valid_time, directory)
        if not missing_keys(self.config.source, valid_time, directory):
            return DownloadResult(
                cache_dir=directory,
                paths=paths,
                downloaded=(),
                skipped=tuple(paths),
            )

        effective_workers = resolve_download_workers(
            workers if workers is not None else self._download_options.workers
        )
        creds = self._effective_credentials(
            credentials,
            cds_api_key=cds_api_key,
            cds_api_url=cds_api_url,
            ads_api_key=ads_api_key,
            ads_api_url=ads_api_url,
            ecmwf_api_key=ecmwf_api_key,
            ecmwf_api_url=ecmwf_api_url,
            ecmwf_email=ecmwf_email,
            prompt=prompt,
        )
        backend = get_backend(self.config.source)
        try:
            with use_download_credentials(creds):
                outcome: DownloadOutcome = backend.ensure(
                    self.config.source,
                    valid_time,
                    directory,
                    workers=effective_workers,
                )
        except DownloadBackendError:
            raise
        except Exception as exc:
            raise RuntimeError(
                redact_text(
                    f"Download failed for source {self.config.source.name!r} "
                    f"into {safe_path(directory)}: {sanitize_exception(exc)}"
                )
            ) from None

        paths = expected_paths(self.config.source, valid_time, directory)
        still_missing = missing_keys(self.config.source, valid_time, directory)
        if still_missing:
            raise FileNotFoundError(
                redact_text(
                    f"Download finished but files are still missing: {still_missing} "
                    f"under {safe_path(directory)}"
                )
            )
        return DownloadResult(
            cache_dir=directory,
            paths=paths,
            downloaded=outcome.downloaded,
            skipped=outcome.skipped,
        )

    def ingest_request(
        self,
        valid_time: datetime,
        *,
        cache_dir: Path | None = None,
        time_index: int = 1,
        download: bool = True,
        credentials: DownloadCredentials | None = None,
        cds_api_key: str | None = None,
        cds_api_url: str | None = None,
        ads_api_key: str | None = None,
        ads_api_url: str | None = None,
        ecmwf_api_key: str | None = None,
        ecmwf_api_url: str | None = None,
        ecmwf_email: str | None = None,
        prompt: bool = False,
        workers: int | None = None,
    ) -> IngestRequest:
        directory = normalize_path(cache_dir) if cache_dir is not None else self.resolve_cache_dir(
            DownloadRequest(valid_time=valid_time)
        )
        if download and self.missing(valid_time, cache_dir=directory):
            self.ensure(
                valid_time,
                cache_dir=directory,
                credentials=credentials,
                cds_api_key=cds_api_key,
                cds_api_url=cds_api_url,
                ads_api_key=ads_api_key,
                ads_api_url=ads_api_url,
                ecmwf_api_key=ecmwf_api_key,
                ecmwf_api_url=ecmwf_api_url,
                ecmwf_email=ecmwf_email,
                prompt=prompt,
                workers=workers,
            )
        return IngestRequest(
            valid_time=valid_time,
            cache_dir=directory,
            time_index=time_index,
        )
