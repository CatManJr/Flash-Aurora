from __future__ import annotations

import pytest

from engine.core.presets import DEFAULT_PRESETS
from engine.ingress.validator import BatchValidator
from tests.helpers import smoke_batch


def test_validator_accepts_matching_small_batch() -> None:
    config = DEFAULT_PRESETS.get("small_pretrained")
    validator = BatchValidator(config.variant)
    batch = smoke_batch()
    issues = validator.collect_issues(batch)
    assert any(issue.field == "atmos_levels" for issue in issues)


def test_validator_reports_missing_variable() -> None:
    config = DEFAULT_PRESETS.get("era5_pretrained")
    validator = BatchValidator(config.variant)
    batch = smoke_batch()
    batch.surf_vars.pop("msl")
    issues = validator.collect_issues(batch)
    assert any(issue.field == "surf_vars" for issue in issues)
