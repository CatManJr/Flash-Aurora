from __future__ import annotations

from dataclasses import dataclass

import torch
from flash_aurora.aurora import Batch

from flash_aurora.engine.core.config import ModelVariantSpec


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

        issues.extend(self._collect_variable_issues(batch, variant))
        issues.extend(self._collect_metadata_issues(batch))
        issues.extend(self._collect_shape_issues(batch, variant))
        return issues

    def _collect_variable_issues(
        self,
        batch: Batch,
        variant: ModelVariantSpec,
    ) -> list[ValidationIssue]:
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

        return issues

    def _collect_metadata_issues(self, batch: Batch) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        metadata = batch.metadata

        if not (torch.all(metadata.lat <= 90) and torch.all(metadata.lat >= -90)):
            issues.append(ValidationIssue("lat", "values must lie in [-90, 90]"))
        if not (torch.all(metadata.lon >= 0) and torch.all(metadata.lon < 360)):
            issues.append(ValidationIssue("lon", "values must lie in [0, 360)"))

        if metadata.lat.dim() == metadata.lon.dim() == 1:
            if not torch.all(metadata.lat[1:] - metadata.lat[:-1] < 0):
                issues.append(ValidationIssue("lat", "must be strictly decreasing"))
            if not torch.all(metadata.lon[1:] - metadata.lon[:-1] > 0):
                issues.append(ValidationIssue("lon", "must be strictly increasing"))
        elif metadata.lat.dim() == metadata.lon.dim() == 2:
            if not torch.all(metadata.lat[1:, :] - metadata.lat[:-1, :] < 0):
                issues.append(ValidationIssue("lat", "must be strictly decreasing along columns"))
            if not torch.all(metadata.lon[:, 1:] - metadata.lon[:, :-1] > 0):
                issues.append(ValidationIssue("lon", "must be strictly increasing along rows"))
        else:
            issues.append(
                ValidationIssue("lat_lon", "lat and lon must both be vectors or both be matrices")
            )

        batch_size = next(iter(batch.surf_vars.values())).shape[0]
        if len(metadata.time) != batch_size:
            issues.append(
                ValidationIssue(
                    "time",
                    f"length {len(metadata.time)} must match batch size {batch_size}",
                )
            )

        if metadata.rollout_step != 0:
            issues.append(
                ValidationIssue(
                    "rollout_step",
                    f"initial condition must use rollout_step=0, got {metadata.rollout_step}",
                )
            )

        return issues

    def _collect_shape_issues(
        self,
        batch: Batch,
        variant: ModelVariantSpec,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        height, width = batch.spatial_shape
        expected_h, expected_w = variant.resolution

        if height not in {expected_h, expected_h - 1}:
            issues.append(
                ValidationIssue("resolution", f"height {height} not compatible with {expected_h}")
            )
        if width != expected_w:
            issues.append(
                ValidationIssue("resolution", f"width {width} expected {expected_w}")
            )
        if width % 4 != 0:
            issues.append(ValidationIssue("resolution", f"width {width} must be divisible by 4"))

        batch_size = next(iter(batch.surf_vars.values())).shape[0]
        level_count = len(batch.metadata.atmos_levels)

        for name, tensor in batch.surf_vars.items():
            issues.extend(self._tensor_issues(f"surf_vars.{name}", tensor, (batch_size, 2, height, width)))
        for name, tensor in batch.static_vars.items():
            issues.extend(self._tensor_issues(f"static_vars.{name}", tensor, (height, width)))
        for name, tensor in batch.atmos_vars.items():
            issues.extend(
                self._tensor_issues(
                    f"atmos_vars.{name}",
                    tensor,
                    (batch_size, 2, level_count, height, width),
                )
            )

        return issues

    @staticmethod
    def _tensor_issues(name: str, tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if tuple(tensor.shape) != expected_shape:
            issues.append(
                ValidationIssue(name, f"expected shape {expected_shape}, got {tuple(tensor.shape)}")
            )
        if tensor.dtype not in (torch.float32, torch.float64):
            issues.append(ValidationIssue(name, f"expected float32/float64, got {tensor.dtype}"))
        if not tensor.is_contiguous():
            issues.append(ValidationIssue(name, "tensor must be contiguous"))
        return issues
