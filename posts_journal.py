"""Append-only journal for the Daily Posts pipeline (the X account).

Sibling of pitch_journal, same discipline: every draft, every posted mark and
every graded outcome lands here and nothing is edited in place. The
scoreboard measures the FILTER as well as the ideas: unposted idea drafts are
graded too, so "the ones I chose not to post" accrues its own record.

Record kinds:
    draft    one per queue draft at generation time (full spec, incl. the
             frozen idea legs for later replay)
    posted   the human marked the draft posted in the queue md; carries the
             text as posted (they may have edited it) and an optional url
    outcome  the hypothetical replay booked by scripts/posts_scoreboard.py

A non-default path never touches R2 (pitch_journal's rule): tests and dev
runs must not pollute the evidence trail.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from research_io import append_jsonl, read_jsonl, file_lock

ROOT = Path(__file__).resolve().parent
JOURNAL_PATH = ROOT / "data" / "posts_journal.jsonl"
JOURNAL_R2_KEY = "posts_journal.jsonl"

KINDS = {"draft", "posted", "outcome"}


def sync_down(path: Path = JOURNAL_PATH) -> None:
    if path != JOURNAL_PATH or path.exists():
        return
    from cache_io import download_to_local, is_configured
    if is_configured():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not download_to_local(JOURNAL_R2_KEY, str(path)):
            raise RuntimeError("journal mirror download failed; refusing to create replacement history")
        read_jsonl(path)



def sync_up(path: Path = JOURNAL_PATH) -> None:
    if path != JOURNAL_PATH:
        return
    from cache_io import is_configured, upload_from_local
    if is_configured() and path.exists():
        with file_lock(path):
            read_jsonl(path)
            if not upload_from_local(str(path), JOURNAL_R2_KEY):
                raise RuntimeError("journal mirror upload failed; local evidence was preserved")



def load(path: Path = JOURNAL_PATH, pull: bool = True) -> list[dict]:
    if pull:
        sync_down(path)
    return read_jsonl(path)


def append(records: list[dict], path: Path = JOURNAL_PATH,
           push: bool = True) -> int:
    if not records:
        return 0
    bad = sorted({r.get("kind") for r in records} - KINDS)
    if bad:
        raise ValueError(f"unknown journal record kind(s): {bad}")
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with file_lock(path):
        sync_down(path)
        append_jsonl(path, [{**record, "written_at": stamp} for record in records])
        if push:
            sync_up(path)
    return len(records)


def fold_drafts(records: list[dict]) -> list[dict]:
    """Draft records with posted marks and outcomes merged on, ordered by
    (date, id). Later records win, so a re-grade supersedes cleanly."""
    drafts: dict[str, dict] = {}
    for record in records:
        if record.get("kind") == "draft" and record.get("draft_id"):
            drafts[record["draft_id"]] = dict(record)
    for record in records:
        draft = drafts.get(record.get("draft_id", ""))
        if draft is None:
            continue
        if record.get("kind") == "posted":
            draft["posted"] = True
            draft["posted_text"] = record.get("text")
            draft["posted_url"] = record.get("url")
            draft["posted_at"] = record.get("marked_at") or record.get("written_at")
        elif record.get("kind") == "outcome":
            draft["outcome"] = record.get("outcome")
            draft["graded_at"] = record.get("graded_at")
    return sorted(drafts.values(),
                  key=lambda r: (str(r.get("date", "")), str(r.get("draft_id", ""))))


def recent_fingerprints(records: list[dict], since: str) -> dict[str, str]:
    """fingerprint -> most recent POSTED date for idea drafts on/after
    `since`. Repetition is judged on what the audience saw, so only posted
    drafts count."""
    posted_ids = {r.get("draft_id") for r in records if r.get("kind") == "posted"}
    out: dict[str, str] = {}
    for record in records:
        if record.get("kind") != "draft" or record.get("draft_id") not in posted_ids:
            continue
        date, fp = str(record.get("date", "")), record.get("fingerprint")
        if fp and date >= since:
            out[fp] = max(out.get(fp, ""), date)
    return out
