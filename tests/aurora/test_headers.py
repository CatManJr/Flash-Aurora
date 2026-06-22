"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from pathlib import Path

import pytest

COPYRIGHT_NOTICE_MICROSOFT: str = (
    '"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.'
)
COPYRIGHT_NOTICE_CATMAN: str = '"""Copyright (c) Catman Jr. Licensed under the MIT license.'

PYTHON_FILES: list[Path] = []
"""list[Path]: Python files to scan for headers."""

_root = Path(__file__).resolve().parents[2] / "flash_aurora" / "aurora"
for path in _root.rglob("**/*.py"):
    relative_path = path.relative_to(_root)

    # Ignore virtual environments and tool caches under the package tree.
    if any(p in {".venv", "venv", "node_modules"} for p in relative_path.parts):
        continue

    # Ignore the automatically generated version file.
    if relative_path.name in {"_version.py"}:
        continue

    PYTHON_FILES.append(path)


@pytest.mark.parametrize("python_file", PYTHON_FILES)
def test_presence_of_copyright_header(python_file: Path) -> None:
    with open(python_file) as f:
        lines = list(f.read().splitlines())

    # Allow an optional Unix shebang on line 0; copyright must be on the next line.
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]

    if not lines or not (
        lines[0].startswith(COPYRIGHT_NOTICE_MICROSOFT)
        or lines[0].startswith(COPYRIGHT_NOTICE_CATMAN)
    ):
        raise AssertionError(f"`{python_file}` must start with the copyright notice.")
