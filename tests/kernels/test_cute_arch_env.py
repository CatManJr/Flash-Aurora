"""Tests for CuTe DSL arch auto-detection."""

from __future__ import annotations

import os

import pytest

from flash_aurora.aurora.ops.cute._arch_env import (
    cute_dsl_arch_for_capability,
    detect_cute_dsl_arch,
    ensure_cute_dsl_arch,
)


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (12, 0, "sm_120a"),
        (10, 0, "sm_100a"),
        (9, 0, "sm_90"),
        (8, 9, "sm_89"),
        (8, 6, "sm_86"),
        (8, 0, "sm_80"),
    ],
)
def test_cute_dsl_arch_for_capability(major: int, minor: int, expected: str) -> None:
    assert cute_dsl_arch_for_capability(major, minor) == expected


def test_ensure_cute_dsl_arch_respects_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_80")
    assert ensure_cute_dsl_arch() == "sm_80"


def test_ensure_cute_dsl_arch_sets_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    detected = detect_cute_dsl_arch()
    if detected is None:
        pytest.skip("CUDA not available")
    assert ensure_cute_dsl_arch() == detected
    assert os.environ.get("CUTE_DSL_ARCH") == detected
