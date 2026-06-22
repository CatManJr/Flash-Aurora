from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

CDS_DEFAULT_URL = "https://cds.climate.copernicus.eu/api"
ECMWF_DEFAULT_URL = "https://api.ecmwf.int/v1"


@dataclass(frozen=True)
class DownloadCredentials:
    """Optional API credentials for ingress downloads.

    Resolution order for CDS: explicit fields here -> ``CDSAPI_*`` env vars ->
    ``~/.cdsapirc``. Keys are never logged when ``use_download_credentials`` is active.
    """

    cds_api_key: str | None = None
    cds_api_url: str | None = None
    ecmwf_api_key: str | None = None
    ecmwf_api_url: str | None = None
    ecmwf_email: str | None = None

    @classmethod
    def from_env(cls) -> DownloadCredentials:
        return cls(
            cds_api_key=os.environ.get("CDSAPI_KEY"),
            cds_api_url=os.environ.get("CDSAPI_URL"),
            ecmwf_api_key=os.environ.get("ECMWF_API_KEY"),
            ecmwf_api_url=os.environ.get("ECMWF_API_URL"),
            ecmwf_email=os.environ.get("ECMWF_API_EMAIL"),
        )

    def merge(self, override: DownloadCredentials | None) -> DownloadCredentials:
        if override is None:
            return self
        return DownloadCredentials(
            cds_api_key=override.cds_api_key or self.cds_api_key,
            cds_api_url=override.cds_api_url or self.cds_api_url,
            ecmwf_api_key=override.ecmwf_api_key or self.ecmwf_api_key,
            ecmwf_api_url=override.ecmwf_api_url or self.ecmwf_api_url,
            ecmwf_email=override.ecmwf_email or self.ecmwf_email,
        )

    def cds_settings(self) -> tuple[str, str] | None:
        key = self.cds_api_key
        if not key:
            return None
        url = self.cds_api_url or CDS_DEFAULT_URL
        return url, key

    def secret_literals(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.cds_api_key, self.ecmwf_api_key)
            if value and len(value) >= 4
        )


_active_credentials: ContextVar[DownloadCredentials | None] = ContextVar(
    "flash_aurora_download_credentials",
    default=None,
)


def active_download_credentials() -> DownloadCredentials | None:
    return _active_credentials.get()


@contextmanager
def use_download_credentials(credentials: DownloadCredentials | None) -> Iterator[None]:
    from engine.core.redaction import pop_ephemeral_literals, push_ephemeral_literals

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


def merge_credentials(*layers: DownloadCredentials | None) -> DownloadCredentials:
    merged = DownloadCredentials.from_env()
    for layer in layers:
        merged = merged.merge(layer)
    return merged
