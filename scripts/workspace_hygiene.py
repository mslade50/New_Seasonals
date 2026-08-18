"""Keep task-local changes separate from pre-existing workspace dirtiness.

Typical use::

    python scripts/workspace_hygiene.py start
    # work on the task
    python scripts/workspace_hygiene.py check --allow path/to/intended_change.py

The baseline lives under ``.local/`` (gitignored).  Existing dirty files are
recorded rather than cleaned, so this tool is safe to use in a long-lived dirty
worktree.  ``check`` fails only for paths created or changed after the baseline
unless the path is explicitly allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / ".local" / "workspace_hygiene" / "baseline.json"
ARTIFACT_ROOT = ROOT / "artifacts"


def _git(*args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def _paths_from_git(*args: str) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in _git(*args).splitlines()
        if line.strip()
    }


def _fingerprint(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"exists": False}

    result: dict[str, object] = {
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # Dirty tracked files are normally few and deserve content-level checks.
    # For the potentially large untracked set, size + mtime avoids hashing
    # thousands of browser-cache files during every task preflight.
    return result


def _tracked_fingerprint(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    basic = _fingerprint(relative_path)
    if not basic.get("exists") or not path.is_file():
        return basic
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    basic["sha256"] = digest.hexdigest()
    return basic


def collect_state() -> dict[str, object]:
    tracked = _paths_from_git("diff", "--name-only", "HEAD", "--")
    untracked = _paths_from_git(
        "ls-files", "--others", "--exclude-standard", "--"
    )
    return {
        "head": _git("rev-parse", "HEAD").strip(),
        "branch": _git("branch", "--show-current").strip(),
        "tracked": {
            path: _tracked_fingerprint(path) for path in sorted(tracked)
        },
        "untracked": {
            path: _fingerprint(path) for path in sorted(untracked)
        },
    }


def _baseline_path(value: str | None) -> Path:
    if value:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate
    configured = os.environ.get("NEW_SEASONALS_HYGIENE_BASELINE")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else ROOT / candidate
    return DEFAULT_BASELINE


def write_baseline(path: Path, *, force: bool) -> int:
    if path.exists() and not force:
        print(f"Baseline already exists: {path}")
        print("Use --force only when intentionally starting a new task baseline.")
        return 2

    state = collect_state()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Workspace baseline saved: {path}")
    print(
        "Recorded "
        f"{len(state['tracked'])} tracked change(s) and "
        f"{len(state['untracked'])} untracked file(s)."
    )
    return 0


def _normalize_allow(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        path = value.strip().replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        if path:
            normalized.append(path.rstrip("/"))
    return tuple(normalized)


def _is_allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == item or path.startswith(f"{item}/") for item in allowed)


def _changed_paths(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> set[str]:
    return {
        path
        for path in before.keys() & after.keys()
        if before[path] != after[path]
    }


def check_baseline(path: Path, *, allowed_values: Iterable[str]) -> int:
    if not path.exists():
        print(f"No workspace baseline found: {path}")
        print("Run: python scripts/workspace_hygiene.py start")
        return 2

    baseline = json.loads(path.read_text(encoding="utf-8"))
    current = collect_state()
    allowed = _normalize_allow(allowed_values)
    before_tracked = baseline.get("tracked", {})
    before_untracked = baseline.get("untracked", {})
    after_tracked = current["tracked"]
    after_untracked = current["untracked"]

    candidates = {
        *(set(after_tracked) - set(before_tracked)),
        *(set(after_untracked) - set(before_untracked)),
        *_changed_paths(before_tracked, after_tracked),
        *_changed_paths(before_untracked, after_untracked),
    }
    unexpected = sorted(path for path in candidates if not _is_allowed(path, allowed))

    if unexpected:
        print("Unexpected workspace changes since the baseline:")
        for relative_path in unexpected:
            print(f"  {relative_path}")
        if allowed:
            print(f"Allowed scope: {', '.join(allowed)}")
        print("Move generated output under artifacts/ or explicitly allow intended source paths.")
        return 1

    print("Workspace hygiene check passed.")
    if allowed:
        print(f"Allowed scope: {', '.join(allowed)}")
    return 0


def print_summary() -> int:
    state = collect_state()
    print(f"Branch: {state['branch'] or '(detached)'}")
    print(f"HEAD: {state['head']}")
    print(f"Tracked changes: {len(state['tracked'])}")
    print(f"Untracked files: {len(state['untracked'])}")
    print(f"Local artifact root: {ARTIFACT_ROOT}")
    return 0


def prepare_artifact_dir(category: str | None) -> int:
    target = ARTIFACT_ROOT
    if category:
        category_path = Path(category)
        if category_path.is_absolute() or ".." in category_path.parts:
            print("Artifact category must be a relative path inside artifacts/.")
            return 2
        target /= category_path
    target.mkdir(parents=True, exist_ok=True)
    print(target)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        help="Baseline path (default: .local/workspace_hygiene/baseline.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Record the current dirty state")
    start.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing baseline for a deliberately new task",
    )

    check = subparsers.add_parser("check", help="Find changes made after start")
    check.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PATH",
        help="Intended file or directory scope; repeat as needed",
    )

    subparsers.add_parser("summary", help="Summarize the current workspace")
    artifact = subparsers.add_parser(
        "artifact-dir", help="Create and print an ignored artifact directory"
    )
    artifact.add_argument("category", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = _baseline_path(args.baseline)
    try:
        if args.command == "start":
            return write_baseline(path, force=args.force)
        if args.command == "check":
            return check_baseline(path, allowed_values=args.allow)
        if args.command == "summary":
            return print_summary()
        if args.command == "artifact-dir":
            return prepare_artifact_dir(args.category)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Workspace hygiene error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
