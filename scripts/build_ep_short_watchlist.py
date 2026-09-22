"""Capture or seal the ATR Extended Gap Up morning research supplement."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.short_watchlist import capture, seal


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", action="store_true", help="download fresh daily prices; no broker access")
    action.add_argument("--seal", action="store_true", help="validate completed Google reviews and render the supplement")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--session-date", default=datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if (ROOT / "artifacts").resolve() not in run_dir.parents:
        parser.error("--run-dir must be below this worktree's artifacts directory")
    try:
        if args.capture:
            queue = capture(run_dir, args.session_date)
            print(f"Frozen short research queue: {run_dir / 'queue.json'}")
            print(f"Verified {queue['coverage']['verified']}/{queue['coverage']['requested']} histories; {len(queue['candidates'])} setups require review.")
        else:
            if args.notes is None:
                parser.error("--seal requires --notes (use an empty JSON list for a verified zero-candidate queue)")
            print(f"Validated short watchlist: {seal(run_dir, args.notes)}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"SHORT WATCHLIST UNAVAILABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("Research only. No email, broker, staging or order action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
