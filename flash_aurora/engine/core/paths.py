from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flash_aurora.engine.core.hub import HubDownloadOptions, download_hub_file
from flash_aurora.engine.core.redaction import redact_text, safe_path


class AssetRootRequiredError(ValueError):
    """Raised when callers omit the required ``asset_root``."""


def safe_filename(filename: str) -> str:
    """Reject path-like hub filenames; callers must pass a bare name."""
    name = Path(filename).name
    if name != filename or name in {"", ".", ".."}:
        raise ValueError(f"Unsafe filename: {filename!r}")
    return name


def basename_only(filename: str) -> str:
    """Strip directory components; used when joining under a trusted root."""
    name = Path(filename).name
    if name in {"", ".", ".."}:
        raise ValueError(f"Unsafe filename: {filename!r}")
    return name


def normalize_asset_path(path: Path | str) -> Path:
    """Expand ``~`` and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def normalize_user_path(
    path: Path | str,
    *,
    user_cwd: Path | None = None,
) -> Path:
    """Resolve a user path against a stable working directory.

  Absolute paths are normalized as-is. Relative paths are joined to ``user_cwd``
  (default: process cwd at call time), which avoids surprises when Jupyter's cwd
  differs from the repository root.
    """
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    base = user_cwd or Path.cwd()
    return (base / expanded).resolve()


def require_asset_root(
    explicit: Path | str | None,
    *,
    store_root: Path | str | None = None,
) -> Path:
    """Return a normalized asset root supplied by the caller."""
    if explicit is not None:
        return normalize_asset_path(explicit)
    if store_root is not None:
        return normalize_asset_path(store_root)
    raise AssetRootRequiredError(
        "asset_root is required. Pass asset_root= to from_preset(), for example "
        "AuroraEngine.from_preset('era5_pretrained', asset_root='~/aurora/assets')."
    )


@dataclass(frozen=True)
class AssetStore:
    """Resolve weight and cache files under a caller-provided root only."""

    root: Path | None = None

    def resolve_root(
        self,
        explicit: Path | str | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        del user_cwd  # kept for call-site compatibility; not used for discovery
        return require_asset_root(explicit, store_root=self.root)

    def allowed_roots(
        self,
        explicit: Path | str | None = None,
        user_cwd: Path | None = None,
    ) -> tuple[Path, ...]:
        return (self.resolve_root(explicit, user_cwd),)

    def ensure_root(
        self,
        explicit: Path | str | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        root = self.resolve_root(explicit, user_cwd)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def join(
        self,
        filename: str,
        explicit: Path | str | None = None,
        user_cwd: Path | None = None,
    ) -> Path:
        safe_name = basename_only(filename)
        return self.resolve_root(explicit, user_cwd) / safe_name

    def fetch_hub_file(
        self,
        filename: str,
        *,
        repo: str,
        allow_download: bool,
        explicit: Path | str | None = None,
        user_cwd: Path | None = None,
        hub: HubDownloadOptions | None = None,
    ) -> Path:
        safe_name = safe_filename(filename)
        root = self.ensure_root(explicit, user_cwd)
        local = root / safe_name
        if local.is_file():
            return local
        if not allow_download:
            raise FileNotFoundError(
                redact_text(
                    f"Missing {filename!r} under {safe_path(root)}. "
                    "Place the file there, pass checkpoint_path=, or enable allow_hub_download "
                    "(optionally with hf_endpoint= for a Hugging Face mirror)."
                )
            )
        options = hub or HubDownloadOptions()
        return download_hub_file(
            filename,
            repo=repo,
            local_dir=root,
            revision=options.revision,
            endpoint=options.endpoint,
            token=options.token,
        )
