"""Did today's Daily Pitch actually deliver? Exit non-zero when it did not.

The headless pitch run is a long agent session, and an agent that gives up
politely still exits 0. Task Scheduler would then show green on a morning with
no email. This checks the only durable evidence of delivery: three idea
records dated today in the journal.

    python scripts/check_pitch_delivered.py [--asof YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pitch_journal  # noqa: E402
from pitch_grammar import IDEA_COUNT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=str(dt.date.today()))
    args = ap.parse_args()

    ideas = [r for r in pitch_journal.load(pull=False)
             if r.get("kind") == "idea" and str(r.get("date")) == args.asof]
    if len(ideas) == IDEA_COUNT:
        print(f"OK: {IDEA_COUNT} ideas journaled for {args.asof}")
        return 0
    print(f"FAILED: {len(ideas)} idea record(s) journaled for {args.asof}, "
          f"expected {IDEA_COUNT}. The pitch did not deliver.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
