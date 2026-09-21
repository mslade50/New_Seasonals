"""Read-only EP runtime guard: protect executable code, tolerate known logs."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_runtime(root: Path, expected_commit: str) -> None:
    root = root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("expected commit must be a full SHA")

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
        )

    if git("rev-parse", "HEAD").decode().strip() != expected_commit:
        raise ValueError("runtime commit does not match pin")
    if git("diff", "--name-only", "HEAD", "--"):
        raise ValueError("tracked runtime content differs from pinned commit")
    untracked = (
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    for name in filter(None, untracked):
        path = root / name
        # Chromium can emit this non-executable file during report QA. Preserve
        # it; do not delete it, change ignore rules, or hide unknown source files.
        if name != "debug.log" or path.is_symlink() or not path.is_file():
            raise ValueError("unexpected untracked runtime file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        validate_runtime(ROOT, args.expected_commit)
    except (OSError, ValueError, subprocess.CalledProcessError):
        print(
            "EP runtime validation failed: commit or executable-content integrity mismatch."
        )
        return 1
    print(
        "EP runtime verified: pinned tracked code intact; known non-executable logs do not block scanning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
