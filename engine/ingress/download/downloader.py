from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from engine.core.config import EngineConfig
from engine.core.paths import AssetStore
from engine.core.redaction import redact_text, safe_path, sanitize_exception
from engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from engine.ingress.adapters.request import IngestRequest
from engine.ingress.download.backends import DownloadBackendError, DownloadOutcome, get_backend
from engine.ingress.download.layout import cache_subdir, expected_paths, missing_keys
from engine.ingress.download.paths import ensure_directory, normalize_path


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

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        if self.config.user_cwd is None:
            self.config.user_cwd = Path.cwd()

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        asset_root: Path | None = None,
        user_cwd: Path | None = None,
        presets: PresetRegistry | None = None,
    ) -> DataDownloader:
        registry = presets or DEFAULT_PRESETS
        config = registry.get(name)
        if asset_root is not None:
            config.asset_root = normalize_path(asset_root)
        if user_cwd is not None:
            config.user_cwd = normalize_path(user_cwd)
        return cls(config)

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
    ) -> DownloadResult:
        directory = ensure_directory(
            normalize_path(cache_dir)
            if cache_dir is not None
            else self.resolve_cache_dir(DownloadRequest(valid_time=valid_time))
        )
        paths = expected_paths(self.config.source, valid_time, directory)
        missing = missing_keys(self.config.source, valid_time, directory)
        if not missing:
            return DownloadResult(
                cache_dir=directory,
                paths=paths,
                downloaded=(),
                skipped=tuple(paths),
            )

        backend = get_backend(self.config.source)
        try:
            outcome: DownloadOutcome = backend.ensure(self.config.source, valid_time, directory)
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
    ) -> IngestRequest:
        directory = normalize_path(cache_dir) if cache_dir is not None else self.resolve_cache_dir(
            DownloadRequest(valid_time=valid_time)
        )
        if download and self.missing(valid_time, cache_dir=directory):
            self.ensure(valid_time, cache_dir=directory)
        return IngestRequest(
            valid_time=valid_time,
            cache_dir=directory,
            time_index=time_index,
        )
