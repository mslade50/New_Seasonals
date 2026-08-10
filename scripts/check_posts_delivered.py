"""Did the Daily Posts run actually deliver a queue? Loud on failure.

Delivery means: content/queue/<today>.json exists, parses, carries >= 1
draft, the md review file exists beside it, and the journal holds a draft
record for every queue id (proof the lint --journal-drafts step ran and
passed - the lint refuses to journal a queue with hard findings).

    python scripts/check_posts_delivered.py [--asof YYYY-MM-DD]
                                            [--journal PATH] [--queue-dir PATH]

Exit 0 delivered, 1 not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import posts_journal  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--journal", default=str(posts_journal.JOURNAL_PATH))
    ap.add_argument("--queue-dir", default=str(ROOT / "content" / "queue"))
    args = ap.parse_args()

    day = str(args.asof or dt.date.today())
    qdir = Path(args.queue_dir)
    jpath, mpath = qdir / f"{day}.json", qdir / f"{day}.md"

    problems = []
    drafts = []
    if not jpath.exists():
        problems.append(f"missing {jpath}")
    else:
        try:
            drafts = json.loads(jpath.read_text(encoding="utf-8")).get("drafts") or []
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"unreadable queue json: {exc}")
        if jpath.exists() and not drafts:
            problems.append("queue json has no drafts")
    if not mpath.exists():
        problems.append(f"missing {mpath} (the review file)")

    if drafts:
        records = posts_journal.load(Path(args.journal), pull=False)
        on_record = {r.get("draft_id") for r in records if r.get("kind") == "draft"}
        missing = [d.get("id") for d in drafts if d.get("id") not in on_record]
        if missing:
            problems.append(f"drafts not journaled (lint step skipped or "
                            f"failed?): {missing}")

    if problems:
        for p in problems:
            print(f"POSTS-DELIVERY: {p}")
        return 1
    kinds = [d.get("type") for d in drafts]
    print(f"delivered: {len(drafts)} draft(s) for {day} ({', '.join(kinds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
