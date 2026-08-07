"""Did today's Daily Pitch actually deliver? Exit non-zero when it did not.

The headless pitch run is a long agent session, and an agent that gives up
politely still exits 0. Task Scheduler would then show green on a morning with
no email. This checks the only durable evidence of delivery: three idea
records dated today in the journal, or one stand_down record.

A stand-down counts as delivery because it IS a delivered verdict: the email
goes out, the tab is cleared, and the journal records what was swept. What
this check still catches is the failure it was written for, an agent that
finished without publishing anything at all.

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
    ap.add_argument("--journal", default=str(pitch_journal.JOURNAL_PATH),
                    help="journal path; matches daily_pitch.py's flag so a "
                         "test or dev run never reads the shared trail")
    args = ap.parse_args()

    today = [r for r in pitch_journal.load(Path(args.journal), pull=False)
             if str(r.get("date")) == args.asof]
    ideas = [r for r in today if r.get("kind") == "idea"]
    if len(ideas) == IDEA_COUNT:
        print(f"OK: {IDEA_COUNT} ideas journaled for {args.asof}")
        return 0

    stand_down = [r for r in today if r.get("kind") == "stand_down"]
    if stand_down and not ideas:
        killed = sum(1 for r in today if r.get("kind") == "killed")
        print(f"OK: stand-down journaled for {args.asof} "
              f"({killed} kill record(s)). Nothing shipped, by verdict.")
        return 0

    # A stand-down later amended by a DIRECTED idea is a legitimate sequence:
    # the sweep found nothing, McKinley overruled a specific kill, and that
    # idea shipped. Mixed model-composed ideas alongside a stand-down remain a
    # half-published run and still fail.
    if stand_down and ideas and all(str(r.get("directed_by", "")).strip()
                                    for r in ideas):
        print(f"OK: stand-down for {args.asof} amended by {len(ideas)} "
              f"directed idea(s).")
        return 0

    print(f"FAILED: {len(ideas)} idea record(s) journaled for {args.asof}, "
          f"expected {IDEA_COUNT} or a stand-down. The pitch did not deliver.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
