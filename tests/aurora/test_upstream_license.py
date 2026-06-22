"""Upstream license and attribution files must ship with flash_aurora.aurora."""

from __future__ import annotations

from pathlib import Path

_AURORA_PKG = Path(__file__).resolve().parents[2] / "flash_aurora" / "aurora"


def test_microsoft_aurora_license_present() -> None:
    license_path = _AURORA_PKG / "LICENSE.txt"
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8")
    assert "Copyright (c) Microsoft Corporation" in text
    assert "MIT License" in text


def test_upstream_notice_present() -> None:
    notice_path = _AURORA_PKG / "NOTICE.md"
    assert notice_path.is_file()
    text = notice_path.read_text(encoding="utf-8")
    assert "github.com/microsoft/aurora" in text


def test_repo_root_license_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    license_path = repo_root / "LICENSE"
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8")
    assert "Catman Jr." in text
    assert "flash_aurora/aurora/LICENSE.txt" in text


def test_flash_aurora_package_license_present() -> None:
    root_license = Path(__file__).resolve().parents[2] / "flash_aurora" / "LICENSE"
    assert root_license.is_file()
    assert "Catman Jr." in root_license.read_text(encoding="utf-8")
