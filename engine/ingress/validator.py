from __future__ import annotations

from dataclasses import dataclass

from aurora import Batch

from engine.core.config import ModelVariantSpec


@dataclass
class ValidationIssue:
    field: str
    message: str


class BatchValidator:
    def __init__(self, variant: ModelVariantSpec) -> None:
        self._variant = variant

    def validate(self, batch: Batch) -> None:
        issues = list(self.collect_issues(batch))
        if issues:
            lines = [f"{item.field}: {item.message}" for item in issues]
            raise ValueError("Batch validation failed:\n" + "\n".join(lines))

    def collect_issues(self, batch: Batch) -> list[ValidationIssue]:
        variant = self._variant
        issues: list[ValidationIssue] = []

        for name in variant.surf_vars:
            if name not in batch.surf_vars:
                issues.append(ValidationIssue("surf_vars", f"missing {name}"))
        for name in variant.static_vars:
            if name not in batch.static_vars:
                issues.append(ValidationIssue("static_vars", f"missing {name}"))
        for name in variant.atmos_vars:
            if name not in batch.atmos_vars:
                issues.append(ValidationIssue("atmos_vars", f"missing {name}"))

        if batch.metadata.atmos_levels != variant.levels:
            issues.append(
                ValidationIssue(
                    "atmos_levels",
                    f"expected {variant.levels}, got {batch.metadata.atmos_levels}",
                )
            )

        if not isinstance(batch.metadata.atmos_levels, tuple):
            issues.append(ValidationIssue("atmos_levels", "must be a tuple"))

        height = batch.spatial_shape[0]
        width = batch.spatial_shape[1]
        expected_h, expected_w = variant.resolution
        if height not in {expected_h, expected_h - 1}:
            issues.append(
                ValidationIssue("resolution", f"height {height} not compatible with {expected_h}")
            )
        if width != expected_w:
            issues.append(
                ValidationIssue("resolution", f"width {width} expected {expected_w}")
            )

        return issues
