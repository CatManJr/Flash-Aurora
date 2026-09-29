"""Closed-loop comparison keeps one step pair, not the full trajectory."""

from __future__ import annotations

import torch

from bench_rollout_drift import (
    compare_step,
    representative_variable,
    variable_drift_series,
)


def test_compare_step_reports_mean_relative_error() -> None:
    reference = {"surf_vars.2t": torch.ones(2, 2)}
    candidate = {"surf_vars.2t": torch.ones(2, 2) * 1.1}
    stats = compare_step(reference, candidate, (("surf_vars", "2t", 1.0),))
    assert stats["n_fail"] == 0
    assert abs(stats["vars"][0]["mean_rel"] - 0.1) < 1e-6


def test_variable_drift_series_uses_named_channel_and_tolerance() -> None:
    assert representative_variable("hres_t0_finetuned") == "10v"
    assert representative_variable("cams") == "pm10"
    assert representative_variable("aurora_v1p5") == "scaled_sf_1h"
    series = [
        {
            "lead_hours": 6,
            "vars": [
                {"name": "10v", "mean_rel": 1.1e-3, "tol": 5e-3},
                {"name": "2t", "mean_rel": 1e-5, "tol": 1e-4},
            ],
        }
    ]
    lead, values, tolerance = variable_drift_series(series, "10v")
    assert lead == [6]
    assert values == [1.1e-3]
    assert tolerance == 5e-3
