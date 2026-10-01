from __future__ import annotations

from pathlib import Path

import pytest
import torch

from _parallel_ladder import (
    TrajectoryReader,
    TrajectoryWriter,
    score_step,
    score_variable,
    tolerance_by_key,
)

_MANIFEST = {"preset": "era5_pretrained", "ic": "2023-01-01T06:00:00", "scheme": "pp"}


def test_identical_fields_are_bitwise_equal_with_zero_error() -> None:
    field = torch.randn(4, 5)
    score = score_variable("surf_vars.2t", field, field.clone(), tolerance=1e-4)
    assert score.bitwise_equal and score.mean_rel == 0.0 and score.max_abs == 0.0


def test_mean_rel_is_mean_abs_error_over_mean_abs_reference() -> None:
    reference = torch.tensor([2.0, -2.0])
    candidate = torch.tensor([2.5, -2.0])
    score = score_variable("surf_vars.2t", reference, candidate, tolerance=0.1)
    assert score.mean_rel == pytest.approx(0.125)
    assert score.max_abs == pytest.approx(0.5)
    assert not score.within_tolerance


def test_step_counts_variables_outside_tolerance() -> None:
    tolerances = tolerance_by_key((("surf_vars", "2t", 1e-4), ("atmos_vars", "q", 5e-3)))
    reference = {"surf_vars.2t": torch.ones(3), "atmos_vars.q": torch.ones(3)}
    candidate = {"surf_vars.2t": torch.full((3,), 1.01), "atmos_vars.q": torch.ones(3)}
    step = score_step(1, reference, candidate, tolerances)
    assert step["n_fail"] == 1 and not step["bitwise_equal"]


def test_trajectory_round_trip(tmp_path: Path) -> None:
    fields = {"surf_vars.2t": torch.randn(2, 3)}
    TrajectoryWriter(tmp_path / "traj", _MANIFEST).write_step(1, fields)
    reader = TrajectoryReader(tmp_path / "traj")
    torch.testing.assert_close(reader.read_step(1), fields)


def test_reader_rejects_a_different_initial_condition(tmp_path: Path) -> None:
    TrajectoryWriter(tmp_path / "traj", _MANIFEST)
    reader = TrajectoryReader(tmp_path / "traj")
    with pytest.raises(ValueError, match="ic"):
        reader.require_same_case({**_MANIFEST, "ic": "2023-04-01T06:00:00"})


def test_writer_refuses_to_mix_two_runs(tmp_path: Path) -> None:
    TrajectoryWriter(tmp_path / "traj", _MANIFEST).write_step(1, {"surf_vars.2t": torch.zeros(1)})
    with pytest.raises(FileExistsError):
        TrajectoryWriter(tmp_path / "traj", _MANIFEST)
