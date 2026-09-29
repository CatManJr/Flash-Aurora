"""Closed-loop comparison keeps one step pair, not the full trajectory."""

from __future__ import annotations

import torch

from bench_rollout_drift import _atomic_save, compare_step_files


def test_compare_step_files_reads_one_pair(tmp_path) -> None:
    reference = tmp_path / "ref"
    candidate = tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    ref_step = {"surf_vars.2t": torch.ones(2, 2)}
    cand_step = {"surf_vars.2t": torch.ones(2, 2) * 1.1}
    _atomic_save(ref_step, reference / "step_0001.pt")
    _atomic_save(cand_step, candidate / "step_0001.pt")
    rows = compare_step_files(
        reference,
        candidate,
        steps=1,
        hours=6.0,
        var_specs=(("surf_vars", "2t", 1.0),),
    )
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["lead_hours"] == 6.0
    assert rows[0]["n_fail"] == 0
    assert abs(rows[0]["vars"][0]["mean_rel"] - 0.1) < 1e-6
