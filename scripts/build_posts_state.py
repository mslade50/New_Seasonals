"""Stage A of the Daily Posts pipeline: queue ingest + deterministic state.

    python scripts/build_posts_state.py [--asof YYYY-MM-DD] [--out PATH]
                                        [--queue-dir PATH] [--journal PATH]

Two jobs, in order:

1. INGEST yesterday's queue marks. The drafting run writes
   content/queue/<date>.md (human) + .json (authoritative). McKinley flips
   `Posted: no` to `yes` (or pastes the post URL) on whatever he published.
   This run captures those marks into the posts journal BEFORE anything is
   overwritten - the same one-window convention as the Pitch tab's Approve
   column. The text is captured from the MD (he may have edited wording
   before posting), the spec from the journal's draft record.

2. ASSEMBLE data/posts_state.json for the drafting skill: session dates,
   macro calendar, tape metrics (through TONIGHT's close - this is an
   evening product, unlike the pitch), risk dial state, the context
   journal's recent vetted nuggets, recent pitch activity, the posts
   scoreboard, and repetition state.

Every block except prices is best effort: a missing input adds a warning and
leaves the block partial. Reuses build_pitch_state's calendar/tape/risk
builders one-way (the context engine's precedent); the book never imports
any of this.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import posts_journal  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402
from build_pitch_state import (  # noqa: E402
    build_calendar, build_risk, build_tape, td_ahead,
)
from pitch_grammar import load_prices  # noqa: E402
from posts_grammar import REPEAT_BLOCK_TD  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "posts_state.json"
QUEUE_DIR = ROOT / "content" / "queue"
CONTEXT_JOURNAL = ROOT / "data" / "context_journal.jsonl"
SCOREBOARD_PATH = ROOT / "data" / "posts_scoreboard.json"

HEAD = re.compile(r"^## \d+\. \[(?P<type>[a-z]+)\] id=(?P<id>\S+)\s*$")
POSTED = re.compile(r"^Posted:\s*(?P<mark>\S.*?)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# queue ingest
# ---------------------------------------------------------------------------
def parse_queue_md(text: str) -> list[dict]:
    """[{id, type, mark, text}] from a queue md. The `## N. [type] id=...`
    head and the `Posted:` line are the parser contract."""
    blocks: list[dict] = []
    current: dict | None = None
    body: list[str] = []

    def _close():
        if current is not None:
            current["text"] = "\n".join(body).strip()
            blocks.append(current)

    for line in text.splitlines():
        m = HEAD.match(line)
        if m:
            _close()
            current = {"id": m.group("id"), "type": m.group("type"),
                       "mark": None}
            body = []
            continue
        if current is not None and current["mark"] is None:
            p = POSTED.match(line)
            if p:
                current["mark"] = p.group("mark")
                continue
        if current is not None:
            body.append(line)
    _close()
    return blocks


def _mark_is_posted(mark: str | None) -> tuple[bool, str | None]:
    if not mark:
        return False, None
    low = mark.strip().lower()
    if low.startswith("http"):
        return True, mark.strip()
    return low in ("yes", "y", "posted"), None


def ingest_queue_marks(queue_dir: Path, journal_path: Path,
                       today: pd.Timestamp, warnings: list[str]) -> dict:
    """Capture Posted marks from every queue md dated before today whose
    drafts have no posted record yet. Idempotent by draft_id."""
    records = posts_journal.load(journal_path)
    already = {r.get("draft_id") for r in records if r.get("kind") == "posted"}
    known = {r.get("draft_id") for r in records if r.get("kind") == "draft"}
    captured, out = [], {"files": 0, "posted": 0, "unmarked": 0}

    if not queue_dir.exists():
        return out
    for md_path in sorted(queue_dir.glob("*.md")):
        date = md_path.stem
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or date >= str(today.date()):
            continue
        out["files"] += 1
        try:
            blocks = parse_queue_md(md_path.read_text(encoding="utf-8"))
        except OSError as exc:
            warnings.append(f"queue {md_path.name} unreadable: {exc}")
            continue
        for block in blocks:
            if block["id"] in already:
                continue
            posted, url = _mark_is_posted(block["mark"])
            if not posted:
                out["unmarked"] += 1
                continue
            if block["id"] not in known:
                warnings.append(f"queue mark for unknown draft {block['id']} "
                                f"({md_path.name}); was lint --journal-drafts "
                                f"skipped that day?")
            captured.append({"kind": "posted", "draft_id": block["id"],
                             "date": date, "text": block["text"], "url": url,
                             "marked_at": dt.datetime.now().isoformat(
                                 timespec="seconds")})
    if captured:
        posts_journal.append(captured, journal_path)
        out["posted"] = len(captured)
    return out


# ---------------------------------------------------------------------------
# state blocks
# ---------------------------------------------------------------------------
def _context_recent(today: pd.Timestamp, warnings: list[str]) -> list[dict]:
    if not CONTEXT_JOURNAL.exists():
        warnings.append("context journal absent; no vetted-nugget block")
        return []
    since = str((today - 14 * TRADING_DAY).date())
    out = []
    for line in CONTEXT_JOURNAL.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("asof", "")) >= since:
            out.append({k: rec.get(k) for k in
                        ("asof", "subject", "cell", "tag", "n", "mean_pct",
                         "hit", "t", "sign_p", "text", "fingerprint")})
    return out[-40:]


def _pitch_recent(today: pd.Timestamp, warnings: list[str]) -> list[dict]:
    try:
        import pitch_journal
        ideas = pitch_journal.fold_ideas(pitch_journal.load(pull=False))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pitch journal unreadable: {exc}")
        return []
    since = str((today - 10 * TRADING_DAY).date())
    out = []
    for idea in ideas:
        if str(idea.get("date", "")) < since:
            continue
        outcome = idea.get("outcome") or {}
        out.append({"date": idea.get("date"), "title": idea.get("title"),
                    "grade": idea.get("grade"),
                    "approved": pitch_journal.approved(idea),
                    "r": outcome.get("r_multiple"),
                    "status": outcome.get("status")})
    return out


def build_state(asof: str | None, queue_dir: Path,
                journal_path: Path) -> dict:
    warnings: list[str] = []
    today = pd.Timestamp(asof or dt.date.today()).normalize()
    next_session = (today + 1 * TRADING_DAY).normalize()

    ingested = ingest_queue_marks(queue_dir, journal_path, today, warnings)

    prices = load_prices()
    # Freshness keys on SPY, not the global max: crypto/FX rows carry
    # weekend-stamped bars (BTC-USD, JPY=X) that make the global max read
    # "Sunday" while every equity is still on Friday's close.
    spy = prices.loc[prices["ticker"] == "SPY", "date"]
    last_bar = pd.Timestamp((spy if len(spy) else prices["date"]).max()).normalize()
    # Evening product: fresh means tonight's close is in the cache. A run on
    # a non-session day (Sunday) is fresh if the last session's bar is there.
    last_session = today if len(pd.date_range(today, today, freq=TRADING_DAY)) \
        else (today - 1 * TRADING_DAY).normalize()
    prices_fresh = last_bar >= last_session

    tape = build_tape(prices, next_session, warnings)
    calendar = build_calendar(next_session)
    risk = build_risk(next_session, warnings)

    records = posts_journal.load(journal_path, pull=False)
    since = str((today - REPEAT_BLOCK_TD * TRADING_DAY).date())
    fps = posts_journal.recent_fingerprints(records, since)
    drafts = posts_journal.fold_drafts(records)
    recent_posted = [
        {"date": d.get("date"), "type": d.get("type"),
         "head": str(d.get("posted_text") or d.get("text") or "")[:90]}
        for d in drafts if d.get("posted")
        and str(d.get("date", "")) >= str((today - 7 * TRADING_DAY).date())]

    scoreboard = None
    if SCOREBOARD_PATH.exists():
        try:
            scoreboard = json.loads(SCOREBOARD_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"posts scoreboard unreadable: {exc}")

    return {
        "meta": {
            "run_date": str(today.date()),
            "next_session": str(next_session.date()),
            "last_price_bar": str(last_bar.date()),
            "prices_fresh": bool(prices_fresh),
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "queue_ingested": ingested,
        "calendar": calendar,
        "tape": tape,
        "risk": risk,
        "context_recent": _context_recent(today, warnings),
        "pitch_recent": _pitch_recent(today, warnings),
        "negative_registry": str(ROOT / "data" / "pitch_negative_registry.md"),
        "scoreboard": scoreboard,
        "repetition": {"recent_fingerprints": fps,
                       "recent_posted": recent_posted,
                       "block_td": REPEAT_BLOCK_TD},
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--queue-dir", default=str(QUEUE_DIR))
    ap.add_argument("--journal", default=str(posts_journal.JOURNAL_PATH))
    args = ap.parse_args()

    state = build_state(args.asof, Path(args.queue_dir), Path(args.journal))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")

    meta, q = state["meta"], state["queue_ingested"]
    print(f"posts state -> {out}")
    print(f"  run {meta['run_date']}, next session {meta['next_session']}, "
          f"last bar {meta['last_price_bar']} "
          f"({'fresh' if meta['prices_fresh'] else 'STALE'})")
    print(f"  queue ingest: {q['posted']} posted mark(s) captured from "
          f"{q['files']} file(s), {q['unmarked']} left unmarked")
    for w in state["warnings"]:
        print(f"  WARN: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
