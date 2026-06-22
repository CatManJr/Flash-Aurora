#!/usr/bin/env python3
"""Remove legacy sys.path hacks for the old aurora/ tree."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    re.compile(r"^\s*_AURORA_PKG = _REPO / \"aurora\"\n\s*if _AURORA_PKG\.is_dir\(\):\n\s*sys\.path\.insert\(0, str\(_AURORA_PKG\)\)\n", re.M),
    re.compile(r"^\s*_AURORA_ROOT = _REPO / \"aurora\"\n\s*if _AURORA_ROOT\.is_dir\(\):\n\s*sys\.path\.insert\(0, str\(_AURORA_ROOT\)\)\n", re.M),
    re.compile(r"^\s*sys\.path\.insert\(0, os\.path\.join\([^\n]*\"aurora\"\)\)\n", re.M),
    re.compile(r"^\s*sys\.path\.insert\(0, os\.path\.join\(_BENCH_DIR, \"\.\.\", \"aurora\"\)\)\n", re.M),
]


def clean(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for pat in PATTERNS:
        text = pat.sub("", text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    for path in list((ROOT / "benchmark").glob("*.py")) + list((ROOT / "profiling").glob("*.py")):
        if clean(path):
            n += 1
            print(path.relative_to(ROOT))
    print(f"cleaned {n} files")


if __name__ == "__main__":
    main()
