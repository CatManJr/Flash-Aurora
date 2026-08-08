"""Upstream license and attribution files must ship with flash_aurora.models.aurora."""

from __future__ import annotations

from pathlib import Path

_AURORA_PKG = Path(__file__).resolve().parents[2] / "flash_aurora" / "models" / "aurora"
_AURORA_V1P5_PKG = Path(__file__).resolve().parents[2] / "flash_aurora" / "models" / "aurora_v1p5"


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


def test_aurora_v1p5_license_and_notice_present() -> None:
    license_path = _AURORA_V1P5_PKG / "LICENSE.txt"
    notice_path = _AURORA_V1P5_PKG / "NOTICE.md"
    assert license_path.is_file()
    assert "Copyright (c) Microsoft Corporation" in license_path.read_text(encoding="utf-8")
    assert notice_path.is_file()
    notice = notice_path.read_text(encoding="utf-8")
    assert "github.com/microsoft/aurora" in notice
    assert "v2.0.1" in notice


def test_legacy_aurora_notice_caps_at_v1_8_0() -> None:
    notice = (_AURORA_PKG / "NOTICE.md").read_text(encoding="utf-8")
    assert "v1.8.0" in notice
    assert "freeze" in notice.lower() or "capped" in notice.lower()


def test_repo_root_license_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    license_path = repo_root / "LICENSE"
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8")
    assert text.startswith("MIT License\n")
    assert "Copyright (c) Catman Jr." in text
    assert "Permission is hereby granted" in text


def test_flash_aurora_package_license_present() -> None:
    root_license = Path(__file__).resolve().parents[2] / "flash_aurora" / "LICENSE"
    assert root_license.is_file()
    assert "Catman Jr." in root_license.read_text(encoding="utf-8")
