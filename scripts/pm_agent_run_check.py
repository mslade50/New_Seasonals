"""Run one PM Weekly check script, inside its sandbox rules.

    python scripts/pm_agent_run_check.py <script.py> [--timeout 600]

The headless PM session may execute Python only through this wrapper. It runs
a script only when it is a .py file inside PM_AGENT_HOME/checks/<date>/ and
its source names nothing on the forbidden list (book objects, the Risk Agent's
working files: pm_agent_universe.FORBIDDEN_SOURCE_TOKENS). The script runs in
a child interpreter with the repo on sys.path and the checks folder as cwd.
This is a scope guard for an LLM-written script, not a security boundary.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pm_agent_grammar as G  # noqa: E402
import pm_agent_universe as U  # noqa: E402


def check_path(script: str) -> tuple[Path | None, str]:
    root = U.checks_root().resolve()
    p = Path(script).expanduser()
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None, f"refused: {p} is not inside {root}"
    if len(rel.parts) != 2 or p.suffix != ".py":
        return None, f"refused: expected {root}/<date>/<name>.py, got {p}"
    if not p.is_file():
        return None, f"refused: {p} does not exist"
    bad = G.forbidden_tokens(p.read_text(encoding="utf-8", errors="replace"))
    if bad:
        return None, f"refused: {p.name} names objects outside the PM read boundary: {bad}"
    return p, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("script")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)
    p, msg = check_path(a.script)
    if p is None:
        print(msg)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        return subprocess.run([sys.executable, str(p)], cwd=str(p.parent), env=env,
                              timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"check {p.name} exceeded {a.timeout}s")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
