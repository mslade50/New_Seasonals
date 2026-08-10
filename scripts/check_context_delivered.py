"""Did tonight's Market Context brief actually go out? Exit non-zero if not.

The agent step is a long unattended session, and an agent that gives up
politely still exits 0. Task Scheduler would show green on an evening with no
Slack post. This checks the only durable evidence of delivery: a journal
record dated for tonight's run.

A QUIET TAPE evening counts as delivered. It is a verdict, it posts, and it
journals a `quiet` record. What this still catches is the failure it exists
for: a run that finished having published nothing at all.

    python scripts/check_context_delivered.py [--run-date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_PATH = ROOT / "data" / "context_journal.jsonl"
BRIEF_DIR = ROOT / "data" / "context_briefs"
CELL_MAP_DIR = ROOT / "scratch" / "context_checks"

MIN_NUGGETS = 4


def load_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date",
                    default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    ap.add_argument("--journal", default=str(JOURNAL_PATH))
    args = ap.parse_args()

    tonight = [r for r in load_journal(Path(args.journal))
               if str(r.get("run_date")) == args.run_date]
    nuggets = [r for r in tonight if r.get("kind") == "nugget"]
    quiet = [r for r in tonight if r.get("kind") == "quiet"]

    brief = BRIEF_DIR / f"{args.run_date}.md"
    cell_map = CELL_MAP_DIR / args.run_date / "00_cell_map.md"

    if quiet and not nuggets:
        print(f"OK: QUIET TAPE brief delivered for {args.run_date}.")
        return 0
    if len(nuggets) >= MIN_NUGGETS:
        print(f"OK: {len(nuggets)} nuggets journaled for {args.run_date} "
              f"({brief.name}).")
        return 0

    print(f"FAILED: {len(nuggets)} nugget record(s) journaled for "
          f"{args.run_date}, expected at least {MIN_NUGGETS} or a quiet "
          f"record. The brief did not deliver.")
    print(f"  brief on disk:    {brief.exists()}  ({brief})")
    print(f"  cell map on disk: {cell_map.exists()}  ({cell_map})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
