"""Append-only journal for the Daily Pitch pipeline.

Spec: daily_pitch_agent_spec_2026-08-06.html section 8. "False positives are
fine" only stays fine if they are counted, so every idea, every kill, every
approval and every graded outcome lands here and nothing is ever edited in
place. Records are one JSON object per line in data/pitch_journal.jsonl, and
the file round-trips through R2 so a run on another machine sees the history.

Record kinds:
    idea      the full delivered spec (legs, entry, exit, sizing, orders,
              thesis, evidence, fingerprint)
    killed    a finalist the falsification stage killed, with the reason
    approval  McKinley's Approve cell, captured off the Pitch tab the
              morning AFTER (the tab is cleared and rewritten daily)
    outcome   the hypothetical replay booked by scripts/grade_pitch_journal.py
    stand_down  a morning that shipped nothing, with the sweep's size, the
              axes covered and the near-misses. One per stand-down day, and
              the record check_pitch_delivered accepts in place of three
              ideas. Before this kind existed an all-kill morning wrote
              NOTHING to the journal (kills are journaled by the publisher,
              which never ran), so the day's work was invisible to the
              scoreboard and to the model split.
    short_slate  a morning that shipped ONE or TWO ideas, with the same sweep
              record a stand-down carries. It sits beside the idea records
              rather than replacing them, so the scoreboard still grades the
              ideas and the journal still explains the empty slots.

`fold_ideas` merges the later approval/outcome records onto their idea, which
is what every consumer actually wants. Later records win, so a re-grade
supersedes an earlier one without rewriting history.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOURNAL_PATH = ROOT / "data" / "pitch_journal.jsonl"
JOURNAL_R2_KEY = "pitch_journal.jsonl"

KINDS = {"idea", "killed", "approval", "outcome", "stand_down", "short_slate"}


def sync_down(path: Path = JOURNAL_PATH) -> None:
    """Pull the journal from R2 when this machine has no local copy. Never
    overwrites a local file: local is the writer, R2 is the shared mirror.

    Only the default path syncs. A caller that passes its own path is a test
    or a dev run, and those must never touch the shared evidence trail."""
    if path != JOURNAL_PATH or path.exists():
        return
    try:
        from cache_io import download_to_local, is_configured
        if is_configured():
            path.parent.mkdir(parents=True, exist_ok=True)
            download_to_local(JOURNAL_R2_KEY, str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: pitch journal R2 pull skipped ({exc})")


def sync_up(path: Path = JOURNAL_PATH) -> None:
    if path != JOURNAL_PATH:
        return
    try:
        from cache_io import is_configured, upload_from_local
        if is_configured() and path.exists():
            upload_from_local(str(path), JOURNAL_R2_KEY)
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: pitch journal R2 push skipped ({exc})")


def load(path: Path = JOURNAL_PATH, pull: bool = True) -> list[dict]:
    if pull:
        sync_down(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append(records: list[dict], path: Path = JOURNAL_PATH,
           push: bool = True) -> int:
    if not records:
        return 0
    bad = sorted({r.get("kind") for r in records} - KINDS)
    if bad:
        raise ValueError(f"unknown journal record kind(s): {bad}")
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "written_at": stamp}) + "\n")
    if push:
        sync_up(path)
    return len(records)


def fold_ideas(records: list[dict]) -> list[dict]:
    """Idea records with their approval and outcome merged on. Ordered by
    (date, rank)."""
    ideas: dict[str, dict] = {}
    for record in records:
        if record.get("kind") == "idea" and record.get("idea_id"):
            ideas[record["idea_id"]] = dict(record)
    for record in records:
        idea = ideas.get(record.get("idea_id", ""))
        if idea is None:
            continue
        if record.get("kind") == "approval":
            idea["approve"] = record.get("approve", "")
            idea["approve_captured_at"] = record.get("captured_at")
        elif record.get("kind") == "outcome":
            idea["outcome"] = record.get("outcome")
            idea["graded_at"] = record.get("graded_at")
    return sorted(ideas.values(),
                  key=lambda r: (str(r.get("date", "")), int(r.get("rank", 0))))


def recent_fingerprints(records: list[dict], since: str) -> dict[str, str]:
    """fingerprint -> most recent pitch date, for ideas pitched on/after
    `since` (YYYY-MM-DD). Drives the repetition rule."""
    out: dict[str, str] = {}
    for record in records:
        if record.get("kind") != "idea":
            continue
        date, fp = str(record.get("date", "")), record.get("fingerprint")
        if fp and date >= since:
            out[fp] = max(out.get(fp, ""), date)
    return out


def approved(idea: dict) -> bool:
    return str(idea.get("approve", "")).strip().upper() in ("Y", "YES")
