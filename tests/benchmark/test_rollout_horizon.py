"""Closed-loop medium-range defaults: 240 h, 3xTF32, Perceiver FP32."""

from __future__ import annotations

import pytest

from bench_rollout_drift import (
    CLOSEDLOOP_WORKER,
    MEDIUM_RANGE_LEAD_HOURS,
    _DEFAULT_TIERS,
    format_forecast_timing_table,
    require_perceiver_fp32,
    run_tier_isolated,
    steps_for_horizon,
)


def test_medium_range_is_ten_days() -> None:
    assert MEDIUM_RANGE_LEAD_HOURS == 240
    assert steps_for_horizon(240, 6.0) == 40
    assert steps_for_horizon(240, 12.0) == 20


def test_horizon_rejects_non_integer_steps() -> None:
    with pytest.raises(ValueError, match="integer number"):
        steps_for_horizon(240, 7.0)


def test_default_closedloop_tiers_include_tf32x3_with_fp32_perceiver() -> None:
    assert "tf32x3@fp32" in _DEFAULT_TIERS
    assert "tf32@fp32" in _DEFAULT_TIERS
    assert "tf32x3" not in _DEFAULT_TIERS
    assert "tf32" not in _DEFAULT_TIERS
    assert "bf16_mixed" not in _DEFAULT_TIERS
    for name in _DEFAULT_TIERS:
        require_perceiver_fp32(name)


def test_named_tf32_presets_rejected_for_closedloop() -> None:
    for name in ("tf32", "tf32x3", "bf16_mixed"):
        with pytest.raises(ValueError, match="Perceiver"):
            require_perceiver_fp32(name)


def test_tf32x3_at_tf32_rejected_for_closedloop() -> None:
    with pytest.raises(ValueError, match="Perceiver"):
        require_perceiver_fp32("tf32x3@tf32")


def test_hres_closedloop_uses_isolated_workers() -> None:
    assert CLOSEDLOOP_WORKER.is_file()
    assert callable(run_tier_isolated)


def test_forecast_timing_table_marks_baseline() -> None:
    lines = format_forecast_timing_table(
        {
            "pytorch_backbone_fp32_encoder_decoder_fp32": {
                "ok": True,
                "load_s": 10.0,
                "forecast_s": 100.0,
                "per_step_s": 2.5,
                "peak_gib": 26.6,
            },
            "tf32x3@fp32": {
                "ok": True,
                "load_s": 11.0,
                "forecast_s": 50.0,
                "per_step_s": 1.25,
                "peak_gib": 26.6,
            },
        },
        baseline="pytorch_backbone_fp32_encoder_decoder_fp32",
    )
    joined = "\n".join(lines)
    assert "base" in joined
    assert "2.00x" in joined
    assert "tf32x3@fp32" in joined
