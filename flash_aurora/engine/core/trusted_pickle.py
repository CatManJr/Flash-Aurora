from __future__ import annotations

import pickle
import warnings
from pathlib import Path


class UntrustedPicklePathError(PermissionError):
    pass


def resolve_trusted_path(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    resolved = path.expanduser().resolve()
    for root in allowed_roots:
        root_resolved = root.expanduser().resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue
    raise UntrustedPicklePathError(
        f"Refusing to read {resolved}. Path must stay under: "
        + ", ".join(str(r.resolve()) for r in allowed_roots)
    )


def load_trusted_pickle(path: Path, allowed_roots: tuple[Path, ...]) -> object:
    """Load a pickle that must live under an allowed asset root.

    Pickle can execute code during load. Only use this for files you fetched
    yourself or placed under the engine asset directory.
    """
    trusted_path = resolve_trusted_path(path, allowed_roots)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*numpy(\._core)?\.core(\.numeric)?.*",
        )
        with open(trusted_path, "rb") as handle:
            return pickle.load(handle)
