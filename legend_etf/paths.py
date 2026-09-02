"""Machine-stable runtime paths for the Legend ETF sleeve."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def machine_runtime_dir() -> Path:
    """Return one per-user path shared by every checkout on this host."""

    base = str(os.environ.get("LOCALAPPDATA", "")).strip()
    if not base:
        # Test/non-Windows fallback. Production Windows tasks always provide
        # LOCALAPPDATA for the interactive task principal.
        base = tempfile.gettempdir()
    return Path(base) / "NewSeasonals" / "legend_etf"


def runtime_env_path() -> Path:
    return machine_runtime_dir() / "runtime.env"
