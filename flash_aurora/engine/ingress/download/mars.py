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
def _ecmwf_rc_file(path: Path) -> Iterator[None]:
    env_key = "ECMWF_API_RC_FILE"
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
def _ecmwf_env_credentials(url: str, key: str, email: str) -> Iterator[None]:
    """Expose credentials via the env vars read by ``ecmwf-api-client``."""
    overrides = {
        "ECMWF_API_KEY": key,
        "ECMWF_API_URL": url or ECMWF_DEFAULT_URL,
        "ECMWF_API_EMAIL": email,
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _mars_config_error() -> MarsConfigError:
    rc_path = ecmwfapirc_path()
    if rc_path.is_file():
        return MarsConfigError(
            f"Invalid ECMWF credentials in {safe_config_label(rc_path)}. "
            "The file must be JSON with non-empty 'key' and 'email' fields "
            "(see https://api.ecmwf.int/v1/key). "
            "Alternatively pass ecmwf_api_key and ecmwf_email to DataDownloader.ensure(), "
            "set ECMWF_API_KEY and ECMWF_API_EMAIL, or call ensure(..., prompt=True)."
        )
    return MarsConfigError(
        "Missing ECMWF credentials. Pass ecmwf_api_key and ecmwf_email to DataDownloader.ensure(), "
        f"set ECMWF_API_KEY and ECMWF_API_EMAIL, create {safe_config_label(rc_path)} "
        "(see https://api.ecmwf.int/v1/key), or call ensure(..., prompt=True). "
        "If you used getpass(), the string in parentheses is only a prompt—not your API key."
    )


@contextmanager
def _mars_client_from_settings(url: str, key: str, email: str) -> Iterator[object]:
    ecmwfapi = require_ecmwfapi()
    payload = json.dumps({"url": url or ECMWF_DEFAULT_URL, "key": key, "email": email}, indent=4) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        handle.write(payload)
        config_path = Path(handle.name)
    try:
        with _ecmwf_env_credentials(url, key, email), _ecmwf_rc_file(config_path):
            try:
                client = ecmwfapi.ECMWFService("mars")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize MARS client: {sanitize_exception(exc)}") from None
            yield client
    finally:
        config_path.unlink(missing_ok=True)


@contextmanager
def mars_service() -> Iterator[object]:
    """Yield an initialized MARS client for the duration of a download."""
    active = active_download_credentials()
    merged = merge_credentials(active)
    settings = merged.ecmwf_settings()
    if settings is None:
        raise _mars_config_error()
    url, key, email = settings
    with _mars_client_from_settings(url, key, email) as client:
        yield client


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
            message = sanitize_exception(exc)
            if "no access to services/mars" in message.lower():
                raise RuntimeError(
                    "MARS wave download failed: your ECMWF account is authenticated but "
                    "not authorised for the MARS archive service. Microsoft Aurora uses the "
                    "same MARS request; a registered API key alone may be insufficient. "
                    "See https://www.ecmwf.int/en/forecasts/accessing-forecasts and "
                    "https://confluence.ecmwf.int/display/UDOC/ecmwf.API+error+1%3A+User+has+no+access+to+services+mars+-+Web+API+FAQ. "
                    "Workaround: place the GRIB at "
                    f"{safe_path(target)} and re-run ensure()."
                ) from None
            raise RuntimeError(
                f"MARS wave download failed for {day}: {message}"
            ) from None
    return target
