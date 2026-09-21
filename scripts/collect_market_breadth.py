"""Collect the WSJ Markets Diary new-high/new-low counts over plain HTTP.

The diary page renders itself from a public JSON endpoint that answers a
browser User-Agent. This reads the same document the page reads, the
``marketsDiaryType=diaries`` set, and takes the **Latest Close** column of the
**NYSE** and **NASDAQ** tables. No cookie, login or paywall is involved, and
nothing here bypasses access control: an endpoint that refuses us is an error,
never something to work around.

``marketsDiaryType=overview`` is deliberately NOT used. It answers with an
"Issues At" block whose NYSE counts happen to agree with the diary but whose
NASDAQ counts do not (2026-09-18: overview 72/244, diary 81/246), and its
timestamp is a publication clock ("4:15 PM EDT 9/18/26") rather than the
session the numbers describe. The diaries set names its own session in full
("Friday, September 18, 2026"), which is the only date this collector trusts.

Everything is imported through ``scripts.maintain_market_breadth`` so the
existing validation, the digest-keyed revision retention and the verified R2
publication apply unchanged. A re-pull that returns the same counts for a
session already stored is a no-op; different counts insert a revision and the
export takes the latest observation per session.

Exit codes:
  0  stored a new observation or a revision, or the expected session is
     already current (including an ``--allow-stale`` run that found nothing
     new). ``--publish`` republishes on any such run, so an unchanged store
     still guarantees the canonical R2 objects exist.
  2  the endpoint served an EARLIER session than expected and ``--allow-stale``
     was not set. Loud, but recoverable: the risk producer keeps its documented
     unfloored fallback and the next run collects the session.
  1  network, parse or validation failure
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.maintain_market_breadth import (  # noqa: E402
    COUNT_COLUMNS,
    DEFAULT_DB,
    URL,
    connect,
    export_history,
    import_wsj,
    publish_history,
)
from trading_calendar import TRADING_DAY  # noqa: E402

ET = ZoneInfo("America/New_York")
DEFAULT_EXPORT = ROOT / "data/market_breadth.parquet"
DIARY_ID = {"application": "WSJ", "marketsDiaryType": "diaries"}
COLUMN = "Latest Close"
COLUMN_FIELD = "latestClose"
EXCHANGES = {"nyse": "NYSE", "nasdaq": "NASDAQ"}
ROW_IDS = {"highs": "newhighs", "lows": "newlows"}
POLL_SECONDS = 60
# A blip at 04:10 should not paint the morning red on its own; a genuinely
# unreachable endpoint still exits 1 after these attempts.
RETRY_SECONDS = 10
FETCH_ATTEMPTS = 3
TIMEOUT_SECONDS = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": URL,
}


class CollectionError(RuntimeError):
    """Raised for a network, shape or parse failure worth exiting 1 on."""


def endpoint_url() -> str:
    query = urllib.parse.urlencode(
        {"id": json.dumps(DIARY_ID, separators=(",", ":")), "type": "mdc_marketsdiary"}
    )
    return f"{URL}?{query}"


def fetch_diary(*, timeout: int = TIMEOUT_SECONDS) -> dict:
    request = urllib.request.Request(endpoint_url(), headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CollectionError(f"WSJ diary request failed: {type(exc).__name__}: {exc}") from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"WSJ diary response is not JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("data"), dict):
        raise CollectionError("WSJ diary response has no data object")
    return document["data"]


def parse_session_date(text) -> dt.date:
    """Read the session the diary names, never the clock it was published on."""
    if not isinstance(text, str) or not text.strip():
        raise CollectionError("WSJ diary carries no timestamp")
    value = text.replace("—", " ").replace("–", " ").strip()
    for pattern in ("%A, %B %d, %Y", "%B %d, %Y", "%A %B %d, %Y"):
        try:
            return dt.datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    numeric = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", value)
    if numeric:
        month, day, year = (int(part) for part in numeric.groups())
        if year < 100:
            year += 2000
        try:
            return dt.date(year, month, day)
        except ValueError as exc:
            raise CollectionError(f"WSJ diary timestamp is not a date: {text!r}") from exc
    raise CollectionError(f"Unrecognised WSJ diary timestamp: {text!r}")


def _exchange_set(data: dict, label: str) -> list[dict]:
    """Return the instrument rows of one exchange's Latest Close table.

    The label match is exact, so "NYSE American" and "NYSE Arca" can never be
    read as "NYSE", and the table must actually publish a Latest Close column.
    """
    sets = data.get("instrumentSets")
    if not isinstance(sets, list) or not sets:
        raise CollectionError("WSJ diary response has no instrument sets")
    matches = []
    for group in sets:
        fields = group.get("headerFields") if isinstance(group, dict) else None
        if not isinstance(fields, list):
            continue
        names = {str(f.get("value")): str(f.get("label")) for f in fields if isinstance(f, dict)}
        if names.get("name") != label:
            continue
        if names.get(COLUMN_FIELD) != COLUMN:
            raise CollectionError(f"{label} diary table has no {COLUMN!r} column")
        rows = group.get("instruments")
        if not isinstance(rows, list) or not rows:
            raise CollectionError(f"{label} diary table has no rows")
        matches.append(rows)
    if len(matches) != 1:
        raise CollectionError(f"expected exactly one {label} diary table, found {len(matches)}")
    return matches[0]


def _count(rows: list[dict], row_id: str, label: str) -> int:
    found = [row for row in rows if isinstance(row, dict) and row.get("id") == row_id]
    if len(found) != 1:
        raise CollectionError(f"expected exactly one {label} {row_id} row, found {len(found)}")
    raw = found[0].get(COLUMN_FIELD)
    if not isinstance(raw, str) or not raw.strip():
        raise CollectionError(f"{label} {row_id} has no {COLUMN} value")
    text = raw.strip().replace(",", "")
    if not text.isdigit():
        raise CollectionError(f"{label} {row_id} {COLUMN} is not a count: {raw!r}")
    return int(text)


def parse_diary(data: dict) -> tuple[dt.date, dict[str, int], str, dict]:
    session = parse_session_date(data.get("timestamp"))
    counts: dict[str, int] = {}
    raw_rows: dict[str, list[dict]] = {}
    evidence: list[str] = []
    for prefix, label in EXCHANGES.items():
        rows = _exchange_set(data, label)
        highs = _count(rows, ROW_IDS["highs"], label)
        lows = _count(rows, ROW_IDS["lows"], label)
        counts[f"{prefix}_highs"] = highs
        counts[f"{prefix}_lows"] = lows
        raw_rows[label] = [
            row for row in rows
            if isinstance(row, dict) and row.get("id") in set(ROW_IDS.values())
        ]
        evidence.append(f"{label} {COLUMN}: New highs {highs}; New lows {lows}")
    text = str(data.get("timestamp"))
    return session, counts, f"Diaries {text}. " + ". ".join(evidence) + ".", raw_rows


def build_observation(session: dt.date, counts: dict[str, int], evidence: str,
                      raw_rows: dict, observed: dt.datetime) -> dict:
    payload = {
        "source_url": URL,
        "column": COLUMN,
        "date": session.isoformat(),
        "observed_at": observed.isoformat(),
        "visible_evidence": evidence,
        "collector": "scripts/collect_market_breadth.py",
        "endpoint": endpoint_url(),
        "raw_rows": raw_rows,
    }
    payload.update({key: int(counts[key]) for key in COUNT_COLUMNS})
    return payload


def latest_completed_session(now_utc: dt.datetime) -> dt.date:
    """The session ``validate_wsj`` will accept for a capture made now.

    Identical to the importer's own rule: a session is collectable from 17:00
    ET on its own day, so before that the most recent completed session is the
    previous one. A non-session date (weekend, holiday) always rolls back.
    """
    local = now_utc.astimezone(ET)
    day = pd.Timestamp(local.date())
    if local.hour < 17 or not TRADING_DAY.is_on_offset(day):
        day = day - TRADING_DAY
    return day.date()


def _poll(expect: dt.date, wait_minutes: int, *, fetch, now, sleeper, log):
    deadline = now() + dt.timedelta(minutes=max(0, wait_minutes))
    attempts = 0
    while True:
        observed = now()
        attempts += 1
        try:
            session, counts, evidence, raw_rows = parse_diary(fetch())
        except CollectionError as exc:
            if attempts >= FETCH_ATTEMPTS and observed >= deadline:
                raise
            log(f"WSJ diary attempt {attempts} failed: {exc}; retrying in {RETRY_SECONDS}s")
            sleeper(RETRY_SECONDS)
            continue
        if session >= expect or observed >= deadline:
            return session, counts, evidence, raw_rows, observed
        log(f"WSJ diary still reports {session}; waiting for {expect} "
            f"(retry in {POLL_SECONDS}s, until {deadline.astimezone(ET):%H:%M:%S %Z})")
        sleeper(POLL_SECONDS)


def collect(
    *,
    db_path: Path,
    export_path: Path,
    publish: bool = False,
    dry_run: bool = False,
    expect_session: dt.date | None = None,
    wait_minutes: int = 0,
    allow_stale: bool = False,
    fetch=fetch_diary,
    now=lambda: dt.datetime.now(dt.timezone.utc),
    sleeper=time.sleep,
    log=print,
) -> int:
    expect = expect_session or latest_completed_session(now())
    session, counts, evidence, raw_rows, observed = _poll(
        expect, wait_minutes, fetch=fetch, now=now, sleeper=sleeper, log=log
    )
    log(f"WSJ diary session {session} (expected {expect}); "
        + ", ".join(f"{key}={counts[key]}" for key in COUNT_COLUMNS))

    behind = session < expect
    if behind and not allow_stale:
        log(f"STALE: the WSJ Markets Diary has not published {expect}; newest is "
            f"{session}. Nothing was stored. The risk dial keeps its documented "
            f"unfloored fallback until the next collection.")
        return 2
    if behind:
        log(f"STALE-ACCEPTED: importing {session} because --allow-stale was set.")

    payload = build_observation(session, counts, evidence, raw_rows, observed)
    if dry_run:
        log("DRY RUN: nothing written. Observation that would be imported:")
        log(json.dumps(payload, sort_keys=True)[:2000])
        return 0

    db = connect(db_path)
    try:
        before = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        import_wsj(db, payload, observed, allow_prior_session=behind)
        after = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        stored = after > before
        # Always re-derive the export from the store, even on a no-op. It is
        # the file the risk producer reads, and a manual import that never
        # exported would otherwise leave it behind the database indefinitely.
        table = export_history(db, export_path)
    finally:
        db.close()

    if stored:
        log(f"STORED {session}: observation {after} in the store "
            f"(nyse_net {counts['nyse_highs'] - counts['nyse_lows']:+d}).")
    else:
        log(f"CURRENT {session}: identical counts already stored; nothing changed.")
    if publish:
        # Unconditional on a run that reached here: re-publishing identical
        # bytes is free, and it is what guarantees the canonical database key
        # exists for a machine that has never collected.
        publish_history(db_path, export_path)
    summary = {
        "session": session.isoformat(),
        "expected": expect.isoformat(),
        "stored": bool(stored),
        "stale": bool(behind),
        "observations": after,
        "rows": int(len(table)),
        "published": bool(publish),
        **{key: counts[key] for key in COUNT_COLUMNS},
    }
    log(json.dumps(summary))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument(
        "--expect-session",
        help="ISO session the caller wants. Default: the most recent completed "
             "NYSE session on the Eastern clock, which is today after 17:00 ET "
             "and the previous session before it.",
    )
    ap.add_argument("--wait-minutes", type=int, default=0,
                    help="poll every 60s until the diary reports the expected session")
    ap.add_argument("--allow-stale", action="store_true",
                    help="import the completed session the diary does serve")
    ap.add_argument("--publish", action="store_true",
                    help="publish the verified export and database to canonical R2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    expect = None
    if args.expect_session:
        expect = dt.date.fromisoformat(args.expect_session)
    try:
        return collect(
            db_path=args.db,
            export_path=args.export,
            publish=args.publish,
            dry_run=args.dry_run,
            expect_session=expect,
            wait_minutes=args.wait_minutes,
            allow_stale=args.allow_stale,
        )
    except (CollectionError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
