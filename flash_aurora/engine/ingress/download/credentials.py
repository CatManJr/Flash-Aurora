from __future__ import annotations

import getpass
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from flash_aurora.engine.ingress.download.paths import ecmwfapirc_path, read_cdsapirc_key, read_ecmwfapirc

CDS_DEFAULT_URL = "https://cds.climate.copernicus.eu/api"
ADS_DEFAULT_URL = "https://ads.atmosphere.copernicus.eu/api"
ECMWF_DEFAULT_URL = "https://api.ecmwf.int/v1"


def normalize_secret(value: str | None) -> str | None:
    """Strip whitespace and accidental ``key:`` / prompt prefixes from pasted secrets."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered.startswith("cds api key:"):
        stripped = stripped.split(":", 1)[1].strip()
    elif lowered.startswith("key:"):
        stripped = stripped.split(":", 1)[1].strip()
    return stripped or None


@dataclass(frozen=True)
class DownloadCredentials:
    """Optional API credentials for ingress downloads.

    Resolution order for CDS: explicit fields here -> ``CDSAPI_*`` env vars ->
    ``~/.cdsapirc``. The key is the CDS API key (no legacy ``UID:`` prefix).
    Keys are never logged when ``use_download_credentials`` is active.

    ADS (CAMS) uses ``ADSAPI_*`` or the same Copernicus API key via
    ``CDSAPI_*`` with the ADS endpoint forced in
    :func:`flash_aurora.engine.ingress.download.ads.ads_client`.

    ECMWF MARS uses ``ECMWF_API_*`` env vars, then ``~/.ecmwfapirc`` JSON
    (``url``, ``key``, ``email``). Pass ``prompt=True`` to
    :meth:`flash_aurora.engine.ingress.download.downloader.DataDownloader.ensure`
    to fill missing CDS, ADS, or MARS fields interactively.
    """

    cds_api_key: str | None = None
    cds_api_url: str | None = None
    ads_api_key: str | None = None
    ads_api_url: str | None = None
    ecmwf_api_key: str | None = None
    ecmwf_api_url: str | None = None
    ecmwf_email: str | None = None

    @classmethod
    def _env_value(cls, name: str) -> str | None:
        value = os.environ.get(name)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def from_env(cls) -> DownloadCredentials:
        return cls(
            cds_api_key=cls._env_value("CDSAPI_KEY"),
            cds_api_url=cls._env_value("CDSAPI_URL"),
            ads_api_key=cls._env_value("ADSAPI_KEY"),
            ads_api_url=cls._env_value("ADSAPI_URL"),
            ecmwf_api_key=cls._env_value("ECMWF_API_KEY"),
            ecmwf_api_url=cls._env_value("ECMWF_API_URL"),
            ecmwf_email=cls._env_value("ECMWF_API_EMAIL"),
        )

    @classmethod
    def from_config_files(cls) -> DownloadCredentials:
        ecmwf = read_ecmwfapirc()
        return cls(
            cds_api_key=read_cdsapirc_key(),
            ecmwf_api_key=ecmwf.get("key") if ecmwf else None,
            ecmwf_api_url=ecmwf.get("url") if ecmwf else None,
            ecmwf_email=ecmwf.get("email") if ecmwf else None,
        )

    def merge(self, override: DownloadCredentials | None) -> DownloadCredentials:
        if override is None:
            return self
        return DownloadCredentials(
            cds_api_key=override.cds_api_key or self.cds_api_key,
            cds_api_url=override.cds_api_url or self.cds_api_url,
            ads_api_key=override.ads_api_key or self.ads_api_key,
            ads_api_url=override.ads_api_url or self.ads_api_url,
            ecmwf_api_key=override.ecmwf_api_key or self.ecmwf_api_key,
            ecmwf_api_url=override.ecmwf_api_url or self.ecmwf_api_url,
            ecmwf_email=override.ecmwf_email or self.ecmwf_email,
        )

    def fill_missing(self, fallback: DownloadCredentials | None) -> DownloadCredentials:
        if fallback is None:
            return self
        return DownloadCredentials(
            cds_api_key=self.cds_api_key or fallback.cds_api_key,
            cds_api_url=self.cds_api_url or fallback.cds_api_url,
            ads_api_key=self.ads_api_key or fallback.ads_api_key,
            ads_api_url=self.ads_api_url or fallback.ads_api_url,
            ecmwf_api_key=self.ecmwf_api_key or fallback.ecmwf_api_key,
            ecmwf_api_url=self.ecmwf_api_url or fallback.ecmwf_api_url,
            ecmwf_email=self.ecmwf_email or fallback.ecmwf_email,
        )

    def cds_settings(self) -> tuple[str, str] | None:
        key = normalize_secret(self.cds_api_key)
        if not key:
            return None
        url = self.cds_api_url or CDS_DEFAULT_URL
        return url, key

    def ads_settings(self) -> tuple[str, str] | None:
        key = normalize_secret(self.ads_api_key) or normalize_secret(self.cds_api_key)
        if not key:
            return None
        url = self.ads_api_url or ADS_DEFAULT_URL
        return url, key

    def ecmwf_settings(self) -> tuple[str, str, str] | None:
        key = self.ecmwf_api_key
        email = self.ecmwf_email
        if not key or not email:
            return None
        url = self.ecmwf_api_url or ECMWF_DEFAULT_URL
        return url, key, email

    def secret_literals(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.cds_api_key, self.ads_api_key, self.ecmwf_api_key)
            if value and len(value) >= 4
        )


_active_credentials: ContextVar[DownloadCredentials | None] = ContextVar(
    "flash_aurora_download_credentials",
    default=None,
)

T = TypeVar("T")


def active_download_credentials() -> DownloadCredentials | None:
    return _active_credentials.get()


def run_with_credentials(credentials: DownloadCredentials | None, fn: Callable[[], T]) -> T:
    """Run ``fn`` with ``credentials`` installed in the current thread."""
    if credentials is None:
        return fn()
    token = _active_credentials.set(credentials)
    try:
        return fn()
    finally:
        _active_credentials.reset(token)


def run_with_active_credentials(fn: Callable[[], T]) -> T:
    """Run ``fn`` with download credentials from the current thread."""
    return run_with_credentials(active_download_credentials(), fn)


@contextmanager
def use_download_credentials(credentials: DownloadCredentials | None) -> Iterator[None]:
    from flash_aurora.engine.core.redaction import pop_ephemeral_literals, push_ephemeral_literals

    token = _active_credentials.set(credentials)
    redaction_token = None
    if credentials is not None:
        redaction_token = push_ephemeral_literals(*credentials.secret_literals())
    try:
        yield
    finally:
        if redaction_token is not None:
            pop_ephemeral_literals(redaction_token)
        _active_credentials.reset(token)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
        return ipython.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _read_secret(prompt: str) -> str | None:
    """Read a secret interactively.

    Jupyter kernels do not reliably capture ``getpass`` input; use ``input`` there.
    """
    if _in_notebook():
        value = input(prompt).strip()
    else:
        value = getpass.getpass(prompt).strip()
    return value or None


def merge_credentials(*layers: DownloadCredentials | None) -> DownloadCredentials:
    merged = DownloadCredentials.from_env()
    for layer in layers:
        merged = merged.merge(layer)
    return merged.fill_missing(DownloadCredentials.from_config_files())


def prompt_cds_credentials(credentials: DownloadCredentials) -> DownloadCredentials:
    """Fill missing CDS fields interactively (terminal or notebook stdin)."""
    key = normalize_secret(credentials.cds_api_key)
    if key is None:
        key = normalize_secret(_read_secret("CDS API key: "))
    return credentials.merge(DownloadCredentials(cds_api_key=key))


def prompt_ads_credentials(credentials: DownloadCredentials) -> DownloadCredentials:
    """Fill missing ADS fields interactively (terminal or notebook stdin)."""
    key = credentials.ads_api_key or credentials.cds_api_key
    if key is None:
        key = _read_secret("ADS API key: ")
    return credentials.merge(DownloadCredentials(ads_api_key=key))


def prompt_ecmwf_credentials(credentials: DownloadCredentials) -> DownloadCredentials:
    """Fill missing ECMWF MARS fields interactively (terminal or notebook stdin)."""
    key = credentials.ecmwf_api_key
    email = credentials.ecmwf_email
    if key is None:
        key = _read_secret("ECMWF API key (MARS): ")
    if email is None:
        email = _read_secret("ECMWF account email: ")
    return credentials.merge(
        DownloadCredentials(ecmwf_api_key=key, ecmwf_email=email),
    )


def ecmwf_credential_status(
    credentials: DownloadCredentials | None = None,
) -> tuple[bool, str, str | None]:
    """Return ``(ready, message, email)`` without exposing secrets."""
    merged = merge_credentials(credentials)
    settings = merged.ecmwf_settings()
    if settings is None:
        return (
            False,
            "ECMWF MARS credentials missing. Set ECMWF_API_KEY + ECMWF_API_EMAIL, "
            "create ~/.ecmwfapirc (see https://api.ecmwf.int/v1/key), or call "
            "ensure(..., prompt=True).",
            None,
        )
    _, _, email = settings
    sources: list[str] = []
    if merged.ecmwf_api_key and os.environ.get("ECMWF_API_KEY", "").strip():
        sources.append("ECMWF_API_KEY env")
    if ecmwfapirc_path().is_file() and read_ecmwfapirc() is not None:
        sources.append("~/.ecmwfapirc")
    if credentials is not None and credentials.ecmwf_settings() is not None:
        sources.append("explicit argument")
    source_label = ", ".join(sources) if sources else "merged config"
    return True, f"ECMWF credentials ready ({source_label})", email
