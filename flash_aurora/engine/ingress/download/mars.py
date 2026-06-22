from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from flash_aurora.engine.core.redaction import safe_config_label, sanitize_exception
from flash_aurora.engine.ingress.download.credentials import (
    ECMWF_DEFAULT_URL,
    active_download_credentials,
    merge_credentials,
)
from flash_aurora.engine.ingress.download.paths import ecmwfapirc_path, ensure_directory, normalize_path

WAVE_MARS_PARAMS: dict[str, str] = {
    "swh": "229.140",
    "pp1d": "231.140",
    "mwp": "232.140",
    "mwd": "230.140",
    "shww": "234.140",
    "mdww": "235.140",
    "mpww": "236.140",
    "shts": "237.140",
    "mdts": "238.140",
    "mpts": "239.140",
    "swh1": "121.140",
    "mwd1": "122.140",
    "mwp1": "123.140",
    "swh2": "124.140",
    "mwd2": "125.140",
    "mwp2": "126.140",
    "dwi": "249.140",
    "wind": "245.140",
}


class MarsConfigError(FileNotFoundError):
    """Raised when the ECMWF API config file is missing."""


def require_ecmwfapi():
    try:
        import ecmwfapi
    except ImportError as exc:
        raise ImportError(
            "MARS wave downloads require ecmwf-api-client. "
            "Install with: uv pip install ecmwf-api-client"
        ) from exc
    return ecmwfapi


@contextmanager
def _ecmwf_api_config(path: Path) -> Iterator[None]:
    env_key = "ECMWF_API_CONFIG"
    previous = os.environ.get(env_key)
    os.environ[env_key] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


@contextmanager
def mars_service() -> Iterator[object]:
    """Yield an initialized MARS client for the duration of a download."""
    active = active_download_credentials()
    merged = merge_credentials(active)
    settings = merged.ecmwf_settings()
    ecmwfapi = require_ecmwfapi()

    if settings is not None:
        url, key, email = settings
        payload = json.dumps(
            {"url": url or ECMWF_DEFAULT_URL, "key": key, "email": email},
            indent=4,
        ) + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            handle.write(payload)
            config_path = Path(handle.name)
        try:
            with _ecmwf_api_config(config_path):
                yield ecmwfapi.ECMWFService("mars")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MARS client: {sanitize_exception(exc)}") from None
        finally:
            config_path.unlink(missing_ok=True)
        return

    if not ecmwfapirc_path().is_file():
        raise MarsConfigError(
            "Missing ECMWF credentials. Pass ecmwf_api_key and ecmwf_email to DataDownloader.ensure(), "
            f"set ECMWF_API_KEY and ECMWF_API_EMAIL, or create {safe_config_label(ecmwfapirc_path())} "
            "(see https://api.ecmwf.int/v1/key)."
        )
    try:
        yield ecmwfapi.ECMWFService("mars")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize MARS client: {sanitize_exception(exc)}") from None


def download_wave_grib(cache_dir: Path | str, day: str) -> Path:
    target = normalize_path(cache_dir) / f"{day}-wave.grib"
    if target.is_file():
        return target

    ensure_directory(target.parent)
    with mars_service() as client:
        try:
            client.execute(
                f"""
                request,
                    class=od,
                    date={day}/to/{day},
                    domain=g,
                    expver=1,
                    param={"/".join(WAVE_MARS_PARAMS.values())},
                    stream=wave,
                    time=00:00:00/06:00:00/12:00:00/18:00:00,
                    grid=0.25/0.25,
                    type=an,
                    target="{day}-wave.grib"
                """,
                str(target),
            )
        except Exception as exc:
            raise RuntimeError(
                f"MARS wave download failed for {day}: {sanitize_exception(exc)}"
            ) from None
    return target
