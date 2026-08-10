"""Lint a Daily Posts queue json against posts_grammar, exactly as a
publish would. Hard findings exit 1; the fix is always the queue, never
this gate.

    python scripts/lint_posts.py content/queue/2026-08-10.json
                                 [--journal PATH] [--journal-drafts]

--journal-drafts: on a clean pass, append one `draft` record per draft to
the posts journal (idempotent by draft_id). The drafting skill's last step
runs with this flag so every draft is on the record BEFORE the human
decides what to post - the unposted ones are how the scoreboard measures
the filter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import posts_journal  # noqa: E402
from posts_grammar import REPEAT_BLOCK_TD, fingerprint, validate_queue  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", help="content/queue/<date>.json")
    ap.add_argument("--journal", default=str(posts_journal.JOURNAL_PATH))
    ap.add_argument("--journal-drafts", action="store_true")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    journal_path = Path(args.journal)
    records = posts_journal.load(journal_path, pull=False)

    asof = pd.Timestamp(str(payload.get("asof", dt.date.today()))).normalize()
    since = str((asof - REPEAT_BLOCK_TD * TRADING_DAY).date())
    recent = posts_journal.recent_fingerprints(records, since)

    hard, soft = validate_queue(payload, recent)
    for w in soft:
        print(f"  warn: {w}")
    for h in hard:
        print(f"  HARD: {h}")
    drafts = payload.get("drafts") or []
    print(f"{queue_path.name}: {len(drafts)} draft(s), "
          f"{len(hard)} hard, {len(soft)} soft")
    if hard:
        return 1

    if args.journal_drafts:
        known = {r.get("draft_id") for r in records if r.get("kind") == "draft"}
        fresh = []
        for draft in drafts:
            if draft["id"] in known:
                continue
            rec = {"kind": "draft", "draft_id": draft["id"],
                   "date": str(payload["asof"]), "type": draft["type"],
                   "text": draft.get("text"), "texts": draft.get("texts"),
                   "long": bool(draft.get("long")),
                   "source": draft.get("source"),
                   "evidence": draft.get("evidence"),
                   "changed_since": draft.get("changed_since")}
            if draft["type"] == "idea":
                rec["idea"] = draft["idea"]
                rec["fingerprint"] = fingerprint(draft["idea"])
            fresh.append(rec)
        n = posts_journal.append(fresh, journal_path)
        print(f"journaled {n} new draft record(s) "
              f"({len(drafts) - n} already on record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
