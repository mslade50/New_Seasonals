"""Where New_Seasonals finds trading_ibkr's code, state and secrets.

trading_ibkr's own rule (runtime_paths.py): TRADING_IBKR_STATE_DIR and
TRADING_IBKR_SECRETS_DIR when set, else its code dir. Since the 2026-10-11
cutover the code runs from a pinned worktree (TRADING_IBKR_SOURCE) and state
and secrets live under C:\\trading_state. Each setting resolves:

    1. the process environment (setx user variables, task env);
    2. the config root's .env (dev\\New_Seasonals\\.env). The cutover writes a
       managed block there, so a scheduled task that does not see the new
       user variables still resolves the new locations;
    3. the pre-cutover default: ~/OneDrive/trading_ibkr for the code dir, and
       the code dir for state and secrets.

With nothing set every path is exactly what it was before. Values only:
nothing here reads a secret file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent
SOURCE_VAR = "TRADING_IBKR_SOURCE"
STATE_VAR = "TRADING_IBKR_STATE_DIR"
SECRETS_VAR = "TRADING_IBKR_SECRETS_DIR"


def _dotenv(config_root: Path) -> dict[str, str]:
    path = Path(config_root) / ".env"
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(key: str, environ: Mapping[str, str] | None = None, config_root: Path | None = None) -> str | None:
    """Environment first, then the config root's .env; None when neither has it."""
    env = os.environ if environ is None else environ
    if env.get(key):
        return env[key]
    return _dotenv(ROOT if config_root is None else config_root).get(key) or None


def onedrive_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("USERPROFILE") or Path.home()) / "OneDrive" / "trading_ibkr"


MARKER_VAR = "TRADING_IBKR_RUNTIME_MARKER"
DEFAULT_MARKER = Path("C:/trading/runtime/trading_ibkr.current.json")


def runtime_root(environ: Mapping[str, str] | None = None) -> Path | None:
    """The pinned worktree named by the cutover's marker json, when it exists."""
    env = os.environ if environ is None else environ
    marker = Path(env.get(MARKER_VAR) or DEFAULT_MARKER)
    try:
        root = json.loads(marker.read_text(encoding="utf-8")).get("runtime_root")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(root) if root and Path(root).is_dir() else None


def source_dir(environ: Mapping[str, str] | None = None, config_root: Path | None = None,
               default: str | Path | None = None) -> Path:
    """trading_ibkr code dir: TRADING_IBKR_SOURCE (env, else .env), else the
    runtime root from the cutover marker, else `default` (callers that pinned
    an absolute OneDrive path keep it), else ~/OneDrive/trading_ibkr."""
    v = setting(SOURCE_VAR, environ, config_root)
    if v:
        return Path(v)
    return runtime_root(environ) or (Path(default) if default else onedrive_dir(environ))


def state_dir(environ: Mapping[str, str] | None = None, config_root: Path | None = None) -> Path | None:
    """TRADING_IBKR_STATE_DIR when configured, else None (caller falls back to the code dir)."""
    v = setting(STATE_VAR, environ, config_root)
    return Path(v) if v else None


def secrets_dir(environ: Mapping[str, str] | None = None, config_root: Path | None = None) -> Path | None:
    v = setting(SECRETS_VAR, environ, config_root)
    return Path(v) if v else None
