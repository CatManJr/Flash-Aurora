#!/usr/bin/env python3
"""Rewrite legacy ``aurora`` / ``engine`` imports to ``flash_aurora.*``."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "cutlass", "__pycache__", "node_modules", ".pytest_cache"}

# Order matters: longer / more specific patterns first.
REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfrom aurora\."), "from flash_aurora.aurora."),
    (re.compile(r"\bfrom aurora import\b"), "from flash_aurora.aurora import"),
    (re.compile(r"\bimport aurora\b"), "import flash_aurora.aurora as aurora"),
    (re.compile(r"\bimport aurora\."), "import flash_aurora.aurora."),
    (re.compile(r"\bfrom engine\."), "from flash_aurora.engine."),
    (re.compile(r"\bfrom engine import\b"), "from flash_aurora.engine import"),
    (re.compile(r"\bimport engine\."), "import flash_aurora.engine."),
)


def iter_py_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def rewrite_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for pattern, repl in REPLACEMENTS:
        text = pattern.sub(repl, text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    changed: list[str] = []
    for path in iter_py_files(ROOT):
        if rewrite_file(path):
            changed.append(str(path.relative_to(ROOT)))
    for name in sorted(changed):
        print(name)
    print(f"Updated {len(changed)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
