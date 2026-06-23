"""ZMQ wire protocol for single-worker forecast scheduling (P1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


PROTOCOL_VERSION = 1

ForecastOutputMode = Literal["export_paths", "metadata_only"]
ForecastCommandKind = Literal["forecast", "health", "shutdown"]
ForecastEventKind = Literal[
    "accepted",
    "preparing",
    "running",
    "step",
    "completed",
    "failed",
    "health",
]


class SchedulerError(Exception):
    """Raised when the worker reports a failed forecast job."""


@dataclass(frozen=True)
class ForecastRequest:
    """Scheduler-layer job descriptor (metadata and paths only; no GPU tensors)."""

    request_id: str
    preset: str
    steps: int
    valid_time: str | None = None
    cache_dir: str | None = None
    time_index: int = 1
    download: bool = True
    netcdf_path: str | None = None
    export_dir: str | None = None
    async_export: bool | None = None
    overlap: bool | None = None
    output_mode: ForecastOutputMode = "export_paths"

    def parsed_valid_time(self) -> datetime:
        if self.valid_time is None:
            raise ValueError("valid_time is required when netcdf_path is not set")
        return datetime.fromisoformat(self.valid_time)


@dataclass(frozen=True)
class ForecastCommand:
    kind: ForecastCommandKind
    request: ForecastRequest | None = None


@dataclass(frozen=True)
class ForecastEvent:
    kind: ForecastEventKind
    request_id: str | None = None
    step: int | None = None
    export_path: str | None = None
    valid_time: str | None = None
    error: str | None = None
    worker_preset: str | None = None
    message: str | None = None


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _require_str(payload, key)


def forecast_request_to_dict(request: ForecastRequest) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.request_id,
        "preset": request.preset,
        "steps": request.steps,
        "valid_time": request.valid_time,
        "cache_dir": request.cache_dir,
        "time_index": request.time_index,
        "download": request.download,
        "netcdf_path": request.netcdf_path,
        "export_dir": request.export_dir,
        "async_export": request.async_export,
        "overlap": request.overlap,
        "output_mode": request.output_mode,
    }


def forecast_request_from_dict(payload: dict[str, Any]) -> ForecastRequest:
    version = int(payload.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")

    output_mode = payload.get("output_mode", "export_paths")
    if output_mode not in ("export_paths", "metadata_only"):
        raise ValueError(f"unsupported output_mode {output_mode!r}")

    async_export = payload.get("async_export")
    overlap = payload.get("overlap")
    return ForecastRequest(
        request_id=_require_str(payload, "request_id"),
        preset=_require_str(payload, "preset"),
        steps=int(payload["steps"]),
        valid_time=_optional_str(payload, "valid_time"),
        cache_dir=_optional_str(payload, "cache_dir"),
        time_index=int(payload.get("time_index", 1)),
        download=bool(payload.get("download", True)),
        netcdf_path=_optional_str(payload, "netcdf_path"),
        export_dir=_optional_str(payload, "export_dir"),
        async_export=None if async_export is None else bool(async_export),
        overlap=None if overlap is None else bool(overlap),
        output_mode=output_mode,
    )


def forecast_command_to_dict(command: ForecastCommand) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": command.kind,
    }
    if command.request is not None:
        payload["request"] = forecast_request_to_dict(command.request)
    return payload


def forecast_command_from_dict(payload: dict[str, Any]) -> ForecastCommand:
    version = int(payload.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")

    kind = _require_str(payload, "kind")
    if kind not in ("forecast", "health", "shutdown"):
        raise ValueError(f"unsupported command kind {kind!r}")

    request_payload = payload.get("request")
    request = None if request_payload is None else forecast_request_from_dict(request_payload)
    return ForecastCommand(kind=kind, request=request)


def forecast_event_to_dict(event: ForecastEvent) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": event.kind,
        "request_id": event.request_id,
        "step": event.step,
        "export_path": event.export_path,
        "valid_time": event.valid_time,
        "error": event.error,
        "worker_preset": event.worker_preset,
        "message": event.message,
    }


def forecast_event_from_dict(payload: dict[str, Any]) -> ForecastEvent:
    version = int(payload.get("protocol_version", PROTOCOL_VERSION))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")

    kind = _require_str(payload, "kind")
    if kind not in (
        "accepted",
        "preparing",
        "running",
        "step",
        "completed",
        "failed",
        "health",
    ):
        raise ValueError(f"unsupported event kind {kind!r}")

    step = payload.get("step")
    return ForecastEvent(
        kind=kind,
        request_id=_optional_str(payload, "request_id"),
        step=None if step is None else int(step),
        export_path=_optional_str(payload, "export_path"),
        valid_time=_optional_str(payload, "valid_time"),
        error=_optional_str(payload, "error"),
        worker_preset=_optional_str(payload, "worker_preset"),
        message=_optional_str(payload, "message"),
    )


def encode_command(command: ForecastCommand) -> bytes:
    return json.dumps(forecast_command_to_dict(command), sort_keys=True).encode("utf-8")


def decode_command(data: bytes) -> ForecastCommand:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("command payload must be a JSON object")
    return forecast_command_from_dict(payload)


def encode_event(event: ForecastEvent) -> bytes:
    return json.dumps(forecast_event_to_dict(event), sort_keys=True).encode("utf-8")


def decode_event(data: bytes) -> ForecastEvent:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("event payload must be a JSON object")
    return forecast_event_from_dict(payload)
