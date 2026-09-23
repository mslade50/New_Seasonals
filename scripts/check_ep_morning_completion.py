"""Inspect or checkpoint EP morning work. Read-only by default; never sends email."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from episodic_pivot.morning_completion import STAGES, checkpoint, claim_resume, inspect_morning


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-date", default=datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "episodic_pivot")
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--claim-resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.artifact_root.resolve()
    if (ROOT / "artifacts").resolve() not in root.parents:
        parser.error("Artifact root must be below this runtime's artifacts")
    try:
        if args.stage:
            checkpoint(root, args.session_date, args.stage, {k: Path(v) for k, v in (item.split("=", 1) for item in args.file)})
        elif args.file:
            parser.error("--file requires --stage")
        result = inspect_morning(root, args.session_date)
        if args.claim_resume:
            result["resume_claimed"] = claim_resume(root, args.session_date)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"EP completion check unavailable: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
