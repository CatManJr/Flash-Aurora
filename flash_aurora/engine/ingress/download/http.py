from __future__ import annotations

import os
import warnings
from typing import Iterator
from urllib.parse import urlparse

from flash_aurora.engine.core.redaction import sanitize_exception
from flash_aurora.engine.ingress.download.progress import download_progress_enabled

UCAR_RDA_HOST = "data.rda.ucar.edu"
_ucar_insecure_tls: bool = False
_insecure_warnings_suppressed: bool = False


def ssl_verify_enabled() -> bool:
    """Return whether HTTPS downloads should verify TLS certificates."""
    value = os.environ.get("FLASH_AURORA_SSL_VERIFY", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _require_requests():
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "HTTP downloads require requests. Install with: uv pip install requests"
        ) from exc
    return requests


def _is_ucar_host(url: str) -> bool:
    return urlparse(url).hostname == UCAR_RDA_HOST


def _allow_insecure_retry(url: str, *, verify: bool, exc: Exception) -> bool:
    if not verify:
        return False
    if not _is_ucar_host(url):
        return False
    requests = _require_requests()
    return isinstance(exc, requests.exceptions.SSLError)


def _suppress_insecure_request_warnings() -> None:
    """Silence urllib3 once we intentionally skip TLS verification."""
    global _insecure_warnings_suppressed
    if _insecure_warnings_suppressed:
        return
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass
    _insecure_warnings_suppressed = True


def _activate_ucar_insecure_tls() -> None:
    """Remember UCAR cert issues for this process and warn the user once."""
    global _ucar_insecure_tls
    if _ucar_insecure_tls:
        return
    warnings.warn(
        "UCAR RDA TLS certificate verification failed; continuing without "
        f"verification for {UCAR_RDA_HOST} only in this process. "
        "Set FLASH_AURORA_SSL_VERIFY=0 to skip verification from the first request.",
        stacklevel=4,
    )
    _ucar_insecure_tls = True
    _suppress_insecure_request_warnings()


def _iter_response_chunks(response, *, label: str | None, show_progress: bool) -> Iterator[bytes]:
    total_header = response.headers.get("Content-Length")
    total = int(total_header) if total_header else None
    if not show_progress or total is None:
        yield from response.iter_content(chunk_size=1024 * 1024)
        return

    from tqdm.auto import tqdm

    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=label or "download",
        leave=False,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            bar.update(len(chunk))
            yield chunk


def _read_response_body(response, *, label: str | None, show_progress: bool) -> bytes:
    chunks = list(_iter_response_chunks(response, label=label, show_progress=show_progress))
    return b"".join(chunks)


def _get_with_optional_insecure_retry(
    requests,
    url: str,
    *,
    timeout: float,
    verify: bool,
    stream: bool,
    label: str | None,
    show_progress: bool,
) -> bytes:
    global _ucar_insecure_tls

    if not verify:
        _suppress_insecure_request_warnings()

    if _ucar_insecure_tls and _is_ucar_host(url):
        verify = False

    try:
        response = requests.get(url, timeout=timeout, verify=verify, stream=stream)
        response.raise_for_status()
        if stream:
            return _read_response_body(response, label=label, show_progress=show_progress)
        return response.content
    except Exception as exc:
        if not _allow_insecure_retry(url, verify=verify, exc=exc):
            raise RuntimeError(f"HTTP GET failed for {url}: {sanitize_exception(exc)}") from None

        _activate_ucar_insecure_tls()

        try:
            response = requests.get(url, timeout=timeout, verify=False, stream=stream)
            response.raise_for_status()
            if stream:
                return _read_response_body(response, label=label, show_progress=show_progress)
            return response.content
        except Exception as retry_exc:
            raise RuntimeError(
                f"HTTP GET failed for {url}: {sanitize_exception(retry_exc)}"
            ) from None


def fetch_bytes(
    url: str,
    *,
    timeout: float = 120,
    progress: bool | None = None,
    label: str | None = None,
) -> bytes:
    """GET ``url`` and return the response body.

    UCAR RDA (``data.rda.ucar.edu``) has intermittently shipped expired TLS certificates.
    When verification fails for that host only, later requests in the same process skip
    verification automatically so tutorial ingress keeps working. Set
    ``FLASH_AURORA_SSL_VERIFY=0`` to skip TLS verification for all HTTP downloads.

    Progress bars follow ``FLASH_AURORA_DOWNLOAD_PROGRESS`` (default ``auto``: on in
    terminals and Jupyter).
    """
    requests = _require_requests()
    show_progress = download_progress_enabled() if progress is None else progress
    return _get_with_optional_insecure_retry(
        requests,
        url,
        timeout=timeout,
        verify=ssl_verify_enabled(),
        stream=show_progress,
        label=label,
        show_progress=show_progress,
    )
