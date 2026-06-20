from __future__ import annotations

from datetime import datetime

import pytest
import torch

from engine.core.presets import DEFAULT_PRESETS
from engine.ingress.validator import BatchValidator
from tests.helpers import matching_small_batch


def test_validator_accepts_matching_small_batch() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    issues = validator.collect_issues(matching_small_batch())
    assert issues == []


def test_validator_reports_missing_variable() -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    validator = BatchValidator(config.variant)
    batch = matching_small_batch()
    batch.surf_vars.pop("msl")
    issues = validator.collect_issues(batch)
    assert any(issue.field == "surf_vars" for issue in issues)


def test_validator_rejects_non_monotonic_lat() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    batch = matching_small_batch()
    batch.metadata.lat = batch.metadata.lat.flip(0)
    issues = validator.collect_issues(batch)
    assert any(issue.field == "lat" for issue in issues)


def test_validator_rejects_history_length_not_two() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    batch = matching_small_batch()
    batch.surf_vars["2t"] = torch.randn(1, 1, 400, 800)
    issues = validator.collect_issues(batch)
    assert any(issue.field == "surf_vars.2t" for issue in issues)


def test_validator_rejects_nonzero_rollout_step() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    batch = matching_small_batch()
    batch.metadata.rollout_step = 1
    issues = validator.collect_issues(batch)
    assert any(issue.field == "rollout_step" for issue in issues)


def test_validator_rejects_width_not_divisible_by_four() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    batch = matching_small_batch()
    bad_width = 802
    for key in batch.surf_vars:
        batch.surf_vars[key] = torch.randn(1, 2, 400, bad_width)
    for key in batch.static_vars:
        batch.static_vars[key] = torch.randn(400, bad_width)
    for key in batch.atmos_vars:
        batch.atmos_vars[key] = torch.randn(1, 2, 7, 400, bad_width)
    issues = validator.collect_issues(batch)
    assert any("divisible by 4" in issue.message for issue in issues)
