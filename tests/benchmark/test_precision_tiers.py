"""Mean-rel tiers pair each backbone with Perceiver FP32 and Perceiver TF32."""

from __future__ import annotations

import re
from pathlib import Path

from bench_aurora_precision_all import _DEFAULT_PRECISION_TIERS

_REPO = Path(__file__).resolve().parents[2]
_PAIRS = (
    ("bf16_mixed@fp32", "bf16_mixed@tf32"),
    ("tf32@fp32", "tf32@tf32"),
    ("tf32x3@fp32", "tf32x3@tf32"),
    ("fp32@fp32", "fp32@tf32"),
)


def _shell_array(name: str) -> list[str]:
    text = (_REPO / "benchmark/run_group_a.sh").read_text()
    match = re.search(rf"{name}=\((.*?)\)", text, re.S)
    assert match is not None, name
    return re.findall(r"[A-Za-z0-9_@]+", match.group(1))


def test_default_precision_tiers_pair_perceiver_tf32() -> None:
    for perceiver_fp32, perceiver_tf32 in _PAIRS:
        index = _DEFAULT_PRECISION_TIERS.index(perceiver_fp32)
        assert _DEFAULT_PRECISION_TIERS[index + 1] == perceiver_tf32


def test_group_a_contract_matches_perceiver_pairs() -> None:
    tiers = _shell_array("CONTRACT_TIERS")
    for perceiver_fp32, perceiver_tf32 in _PAIRS:
        index = tiers.index(perceiver_fp32)
        assert tiers[index + 1] == perceiver_tf32


def test_group_a_measures_025_finetuned_and_tc_tracking() -> None:
    presets = _shell_array("PAPER_PRESETS")
    assert "hres_t0_finetuned" in presets
    assert "tc_tracking" in presets
