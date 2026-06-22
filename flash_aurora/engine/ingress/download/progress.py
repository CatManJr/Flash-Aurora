from __future__ import annotations

import os
import sys


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def download_progress_enabled() -> bool:
    """Whether HTTP / multi-file download progress bars should be shown."""
    value = os.environ.get("FLASH_AURORA_DOWNLOAD_PROGRESS", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return sys.stderr.isatty() or _running_in_ipython()
