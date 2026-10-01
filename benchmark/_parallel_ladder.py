"""Trajectory files and per-variable scores for the Group C parallel drift ladder.

A trajectory is one directory: ``manifest.json`` plus ``step_001.pt`` ... with the
full FP32 prediction of each rollout step, keyed ``surf_vars.<name>`` and
``atmos_vars.<name>``. Rung 1 writes one; every other rung on the node scores
against it step by step, so only one full rollout per (preset, IC) is on disk.
Group A can freeze a rollout reference on the PRO 6000 in the same format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

MANIFEST_NAME = "manifest.json"
# Ensemble moments (mean and spread per step) sit next to the member trajectories.
ENSEMBLE_DIRECTORY = "ensemble"
# The ladder compares the same preset, initial condition, and ensemble member; other
# manifest fields (scheme, GPU, precision) are what the comparison is about.
_MATCHING_FIELDS = ("preset", "ic", "member_seed")


def step_file_name(step: int) -> str:
    return f"step_{step:03d}.pt"


def member_directory(root: Path, member: int, num_members: int) -> Path:
    """A deterministic run keeps its steps in ``root``; ensemble member ``k`` in ``member_k``."""
    return root if num_members == 1 else root / f"member_{member:03d}"


class TrajectoryWriter:
    """Writes one FP32 field dictionary per rollout step."""

    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.glob("step_*.pt")):
            raise FileExistsError(f"trajectory {directory} already has steps; use a new directory")
        self.directory = directory
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def write_step(self, step: int, fields: dict[str, torch.Tensor]) -> None:
        torch.save(fields, self.directory / step_file_name(step))


class TrajectoryReader:
    """Reads a trajectory written by :class:`TrajectoryWriter`."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))

    def require_same_case(self, manifest: dict[str, Any]) -> None:
        for field in _MATCHING_FIELDS:
            if self.manifest.get(field) != manifest.get(field):
                raise ValueError(
                    f"reference {self.directory} has {field}={self.manifest.get(field)!r}, "
                    f"this run has {manifest.get(field)!r}"
                )

    def read_step(self, step: int) -> dict[str, torch.Tensor]:
        return torch.load(self.directory / step_file_name(step), map_location="cpu")


@dataclass(frozen=True)
class VariableScore:
    name: str
    mean_rel: float
    max_abs: float
    bitwise_equal: bool
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        return self.mean_rel <= self.tolerance


def score_variable(
    name: str, reference: torch.Tensor, candidate: torch.Tensor, tolerance: float
) -> VariableScore:
    """Mean-rel ``mean|c - r| / mean|r|`` and max-abs, accumulated in FP64.

    FP64 keeps reassociation-level differences (near FP32 roundoff) resolvable.
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"{name}: shape {tuple(candidate.shape)} != reference {tuple(reference.shape)}")
    error = (candidate.double() - reference.double()).abs()
    scale = reference.double().abs().mean().clamp_min(1e-12)
    return VariableScore(
        name=name,
        mean_rel=float(error.mean() / scale),
        max_abs=float(error.max()),
        bitwise_equal=bool(torch.equal(candidate, reference)),
        tolerance=tolerance,
    )


def score_step(
    step: int,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Scores of every output variable at one rollout step."""
    scores = [
        score_variable(key, reference[key], candidate[key], tolerances[key])
        for key in sorted(tolerances)
    ]
    return {
        "step": step,
        "bitwise_equal": all(score.bitwise_equal for score in scores),
        "n_fail": sum(not score.within_tolerance for score in scores),
        "variables": [
            {**asdict(score), "within_tolerance": score.within_tolerance} for score in scores
        ],
    }


def tolerance_by_key(var_specs: tuple[tuple[str, str, float], ...]) -> dict[str, float]:
    """``(group, name, tol)`` triples to ``{"group.name": tol}``."""
    return {f"{group}.{name}": tol for group, name, tol in var_specs}


def peak_memory_gib(device: torch.device) -> dict[str, float]:
    gib = 1024.0**3
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }
