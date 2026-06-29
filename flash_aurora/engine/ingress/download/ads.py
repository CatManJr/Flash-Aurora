from __future__ import annotations

from flash_aurora.engine.core.redaction import safe_config_label, sanitize_exception
from flash_aurora.engine.ingress.download.credentials import (
    ADS_DEFAULT_URL,
    DownloadCredentials,
    active_download_credentials,
    merge_credentials,
)
from flash_aurora.engine.ingress.download.paths import cdsapirc_path, read_cdsapirc_key


class AdsConfigError(FileNotFoundError):
    """Raised when ADS credentials are missing."""


def require_cdsapi():
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "ADS/CAMS downloads require cdsapi. Install with: uv pip install cdsapi netcdf4"
        ) from exc
    return cdsapi


def ads_client():
    """Build a cdsapi client pointed at the Atmosphere Data Store (CAMS).

    Credentials are never written to the repository. Resolution order:

    1. ``ads_api_key`` / ``ADSAPI_KEY`` (optional ``ads_api_url`` / ``ADSAPI_URL``)
    2. ``cds_api_key`` / ``CDSAPI_KEY`` with the ADS endpoint (same API key as CDS)
    3. ``key:`` from ``~/.cdsapirc`` with the ADS endpoint (URL line in the file is ignored)
    """
    active = active_download_credentials()
    merged = merge_credentials(active)
    settings = merged.ads_settings()
    cdsapi = require_cdsapi()
    if settings is not None:
        url, key = settings
        try:
            return cdsapi.Client(url=url, key=key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ADS client: {sanitize_exception(exc)}") from None

    key = read_cdsapirc_key()
    if key:
        try:
            return cdsapi.Client(url=ADS_DEFAULT_URL, key=key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ADS client: {sanitize_exception(exc)}") from None

    raise AdsConfigError(
        "Missing ADS credentials. Pass ads_api_key to DataDownloader.ensure(), "
        "set ADSAPI_KEY (or CDSAPI_KEY with the same API key), "
        f"or create {safe_config_label(cdsapirc_path())} with a key: line "
        "(the ADS API URL is applied automatically). "
        "If you used getpass(), the string in parentheses is only a prompt—not your API key."
    )


def ads_retrieve(client, dataset: str, params: dict, target: str) -> None:
    try:
        client.retrieve(dataset, params, target)
    except Exception as exc:
        raise RuntimeError(f"ADS retrieve failed: {sanitize_exception(exc)}") from None
