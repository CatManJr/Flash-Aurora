from __future__ import annotations

from pathlib import Path

import pytest

from flash_aurora.engine.core.paths import AssetStore, safe_filename
from flash_aurora.engine.core.trusted_pickle import UntrustedPicklePathError, resolve_trusted_path


def test_safe_filename_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        safe_filename("../secret.pickle")
    with pytest.raises(ValueError):
        safe_filename("/etc/passwd")


def test_resolve_trusted_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "fetched"
    root.mkdir()
    outside = tmp_path / "outside.pickle"
    outside.write_bytes(b"data")
    with pytest.raises(UntrustedPicklePathError):
        resolve_trusted_path(outside, (root,))


def test_join_uses_basename_only(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    path = store.join("../escape.pickle", explicit=tmp_path)
    assert path.parent.resolve() == tmp_path.resolve()
    assert path.name == "escape.pickle"
