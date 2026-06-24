from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from flash_aurora.scheduler.protocol import (
    ForecastCommand,
    ForecastEvent,
    ForecastRequest,
    decode_array,
    decode_command,
    decode_event,
    encode_array,
    encode_command,
    encode_event,
    forecast_command_from_dict,
    forecast_command_to_dict,
    forecast_event_from_dict,
    forecast_event_to_dict,
    forecast_request_from_dict,
    forecast_request_to_dict,
)


def test_forecast_request_json_round_trip() -> None:
    request = ForecastRequest(
        request_id="job-1",
        preset="era5_pretrained",
        steps=4,
        valid_time="2024-06-01T06:00:00",
        cache_dir="/data/cache",
        time_index=1,
        download=False,
        export_dir="/tmp/out",
        async_export=True,
        overlap=False,
        output_mode="last_step_array",
        preview_var="2t",
    )
    restored = forecast_request_from_dict(forecast_request_to_dict(request))
    assert restored == request
    assert restored.parsed_valid_time() == datetime(2024, 6, 1, 6, 0, 0)


def test_forecast_command_json_round_trip() -> None:
    command = ForecastCommand(
        kind="forecast",
        request=ForecastRequest(
            request_id="job-2",
            preset="hres_0.1",
            steps=2,
            netcdf_path="/data/input.nc",
        ),
    )
    restored = forecast_command_from_dict(forecast_command_to_dict(command))
    assert restored == command


def test_forecast_event_json_round_trip() -> None:
    event = ForecastEvent(
        kind="step",
        request_id="job-2",
        step=0,
        export_path="/tmp/prediction-000.nc",
        valid_time="2024-06-01T12:00:00",
    )
    restored = forecast_event_from_dict(forecast_event_to_dict(event))
    assert restored == event


def test_forecast_event_array_round_trip() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    event = ForecastEvent(
        kind="step",
        request_id="job-array",
        step=0,
        array_name="2t",
        array_data_b64=encode_array(array),
    )

    restored = forecast_event_from_dict(forecast_event_to_dict(event))

    assert restored.array_name == "2t"
    np.testing.assert_array_equal(restored.array(), array)
    np.testing.assert_array_equal(decode_array(restored.array_data_b64 or ""), array)


def test_encode_decode_bytes() -> None:
    command = ForecastCommand(kind="health")
    assert decode_command(encode_command(command)) == command

    event = ForecastEvent(kind="failed", request_id="job-3", error="boom")
    assert decode_event(encode_event(event)) == event


def test_unsupported_protocol_version() -> None:
    payload = forecast_request_to_dict(
        ForecastRequest(request_id="x", preset="era5_pretrained", steps=1, valid_time="2024-01-01T00:00:00")
    )
    payload["protocol_version"] = 99
    with pytest.raises(ValueError, match="unsupported protocol version"):
        forecast_request_from_dict(payload)
