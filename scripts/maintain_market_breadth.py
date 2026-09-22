"""Append observed WSJ diary counts; export a dated history for the risk producer.

Primary collection is `scripts/collect_market_breadth.py`, which reads the
public Markets Diary JSON the page itself fetches and imports through the
functions here, so every rule below applies unchanged. Capturing the rendered
diary in the in-app browser and importing the JSON by hand with
`--observation` remains the fallback. Neither path circumvents access control.
SQLite retains every distinct source observation; workbook history stays frozen.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from zoneinfo import ZoneInfo
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trading_calendar import TRADING_DAY

URL = "https://www.wsj.com/market-data/stocks/marketsdiary"
DEFAULT_DB = ROOT / "data/market_breadth.sqlite"
CUTOVER = "2026-09-17"
COUNT_COLUMNS = ["nyse_highs", "nyse_lows", "nasdaq_highs", "nasdaq_lows"]
EXPORT_KEY = "market_breadth.parquet"
# Canonical observation store. Both machines read and write one database, so a
# pinned runtime that has never collected can bootstrap instead of starting an
# empty table. Immutable digest-named backups below are unaffected.
DB_KEY = "market_breadth.sqlite"


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS observations (
        date TEXT NOT NULL, source TEXT NOT NULL, observed_at TEXT NOT NULL,
        digest TEXT NOT NULL, nyse_highs INTEGER, nyse_lows INTEGER,
        nasdaq_highs INTEGER, nasdaq_lows INTEGER, payload TEXT NOT NULL,
        PRIMARY KEY (date, source, digest))""")
    return db


def validate_wsj(payload, now=None, *, allow_prior_session=False):
    """Reject anything that is not a completed diary for a real NYSE session.

    ``allow_prior_session`` relaxes exactly one rule: the diary may describe a
    session EARLIER than the most recent completed one. It exists for the
    documented recovery case where the publisher has not yet rolled forward
    (or a collection was missed) and the counts are stored under the session
    the diary itself names. Everything else -- the source, the Latest Close
    column, the trading-session check, the timezone-aware capture timestamp,
    the no-future rule, the integer range and the both-zero quarantine --
    still applies, and a diary dated AHEAD of the clock is always refused.
    """
    if payload.get("source_url") != URL or payload.get("column") != "Latest Close":
        raise ValueError("Use the WSJ Markets Diary Latest Close column only")
    return _validate_session_counts(payload, now, allow_prior_session=allow_prior_session)


def _validate_session_counts(payload, now=None, *, allow_prior_session=False):
    day = pd.Timestamp(payload["date"])
    if day.tzinfo is not None or day != day.normalize() or not TRADING_DAY.is_on_offset(day):
        raise ValueError("Diary date must be an NYSE trading session")
    observed = pd.Timestamp(payload["observed_at"])
    if observed.tzinfo is None:
        raise ValueError("Capture timestamp must include timezone")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if observed > now + pd.Timedelta(minutes=5):
        raise ValueError("Future capture timestamp")
    local = observed.tz_convert("America/New_York")
    if day.date() > local.date() or (day.date() == local.date() and local.hour < 17):
        raise ValueError("Only completed sessions may be imported")
    latest = pd.Timestamp(local.date())
    if local.hour < 17 or not TRADING_DAY.is_on_offset(latest):
        latest = latest - TRADING_DAY
    if day > latest or (day < latest and not allow_prior_session):
        raise ValueError(f"Stale diary date {day.date()}; expected {latest.date()}")
    for key in COUNT_COLUMNS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 20000:
            raise ValueError(f"Missing or invalid integer count: {key}")
    for exchange in ["nyse", "nasdaq"]:
        if payload[f"{exchange}_highs"] + payload[f"{exchange}_lows"] == 0:
            raise ValueError(f"Both {exchange} counts are zero; quarantine this observation")
    if not payload.get("visible_evidence"):
        raise ValueError("Include the visible diary date, exchange labels and counts")
    return day.strftime("%Y-%m-%d"), observed.tz_convert("UTC").isoformat()


def overview_timestamp(text):
    """Parse the overview's explicit Eastern publication timestamp."""
    match = re.fullmatch(r"(\d{1,2}:\d{2} [AP]M) (EDT|EST) (\d{1,2}/\d{1,2}/\d{2,4})", str(text).strip())
    if not match:
        raise ValueError("Overview needs a dated Eastern publication timestamp")
    clock, zone, date = match.groups()
    fmt = "%I:%M %p %m/%d/%Y" if len(date.split("/")[-1]) == 4 else "%I:%M %p %m/%d/%y"
    stamp = dt.datetime.strptime(clock + " " + date, fmt).replace(tzinfo=ZoneInfo("America/New_York"))
    if stamp.tzname() != zone or stamp.time() < dt.time(16, 15):
        raise ValueError("Overview must be a post-close update (16:15 ET or later), with the correct Eastern offset")
    return stamp


def validate_overview(payload, now=None, *, allow_prior_session=False):
    if payload.get("source_url") != URL or payload.get("column") != "Issues At (preliminary)":
        raise ValueError("Use the Dow Jones overview Issues At table")
    stamp = pd.Timestamp(overview_timestamp(payload.get("publisher_timestamp")))
    if str(stamp.date()) != payload.get("date"):
        raise ValueError("Overview publication date must match the stored session")
    if stamp > pd.Timestamp(payload["observed_at"]):
        raise ValueError("Overview publication timestamp is ahead of capture")
    return _validate_session_counts(payload, now, allow_prior_session=allow_prior_session)


def import_overview(db, payload, now=None, *, allow_prior_session=False):
    day, observed = validate_overview(payload, now, allow_prior_session=allow_prior_session)
    with db:
        insert_observation(db, day, "dow_jones_overview", observed,
                           {k: payload[k] for k in COUNT_COLUMNS}, payload)


def insert_observation(db, date, source, observed_at, counts, payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # Re-observing the same dated values is idempotent, while revisions survive.
    digest = hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()
    db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
               (date, source, observed_at, digest, *[counts[k] for k in COUNT_COLUMNS], raw))


def import_wsj(db, payload, now=None, *, allow_prior_session=False):
    day, observed = validate_wsj(payload, now, allow_prior_session=allow_prior_session)
    with db:
        insert_observation(db, day, "wsj", observed, {k: payload[k] for k in COUNT_COLUMNS}, payload)


def seed_workbook(db, path):
    """Import the reviewed, cleaned extract; do not re-clean the source workbook."""
    table = pd.read_csv(path)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    observed = pd.Timestamp.now(tz="UTC").isoformat()
    with db:
        for row in table.to_dict("records"):
            day = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            if day >= CUTOVER:
                raise ValueError("Historical seed reaches beyond the source cutover")
            counts = {}
            for key in COUNT_COLUMNS:
                value = row[key]
                counts[key] = int(value) if pd.notna(value) else None
            # Known zero/zero placeholders in the cleaned workbook are unknown.
            for exchange in ["nyse", "nasdaq"]:
                if pd.isna(row[f"{exchange}_net"]):
                    counts[f"{exchange}_highs"] = counts[f"{exchange}_lows"] = None
            insert_observation(db, day, "workbook", observed, counts,
                               {"source_file_sha256": sha, "source_row": row.get("source_row"), "counts": counts})


def export_history(db, path):
    records = pd.read_sql_query("SELECT * FROM observations ORDER BY observed_at", db)
    if records.empty:
        raise ValueError("No breadth observations")
    # Workbook history is frozen. Live dates may use the preliminary overview
    # until the detailed WSJ diary is available.
    chosen = records.loc[((records.date < CUTOVER) & (records.source == "workbook")) |
                         ((records.date >= CUTOVER) & records.source.isin(["wsj", "dow_jones_overview"]))].copy()
    # The detailed diary supersedes preliminary overview counts, even if an
    # overview is captured later. Preserve both sources in the observation store.
    chosen["priority"] = chosen.source.map({"workbook": 0, "dow_jones_overview": 1, "wsj": 2})
    chosen = chosen.sort_values(["priority", "observed_at"], kind="stable")
    # Within the preferred source, the latest distinct revision wins.
    chosen = chosen.drop_duplicates("date", keep="last").set_index("date").sort_index()
    chosen.index = pd.to_datetime(chosen.index)
    chosen["nyse_net"] = chosen.nyse_highs - chosen.nyse_lows
    chosen["nasdaq_net"] = chosen.nasdaq_highs - chosen.nasdaq_lows
    chosen = chosen[COUNT_COLUMNS + ["nyse_net", "nasdaq_net", "source", "observed_at", "digest"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a unique sibling first, then atomically replace the derived export.
    import uuid
    stage = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    chosen.to_parquet(stage)
    stage.replace(path)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--seed", type=Path)
    ap.add_argument("--observation", type=Path)
    ap.add_argument("--export", type=Path, default=ROOT / "data/market_breadth.parquet")
    ap.add_argument("--publish", action="store_true", help="Publish verified history and an immutable database backup to canonical R2")
    args = ap.parse_args()
    with connect(args.db) as db:
        if args.seed:
            seed_workbook(db, args.seed)
        if args.observation:
            import_wsj(db, json.loads(args.observation.read_text(encoding="utf-8")))
        table = export_history(db, args.export)
    if args.publish:
        publish_history(args.db, args.export)
    print(json.dumps({"rows": len(table), "asof": str(table.index.max().date()),
                      "nyse_net": float(table.nyse_net.iloc[-1]), "database": str(args.db)}))


def publish_history(database, export):
    """A producer publication, never a local private-site build/deployment."""
    import uuid
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import cache_io
    if not cache_io.is_configured():
        raise RuntimeError("R2 credentials unavailable; local database retained")
    destination = Path(export)
    current = pd.read_parquet(destination)
    key = EXPORT_KEY
    remote_meta = cache_io.head(key)
    if remote_meta:
        prior_path = destination.parent / ("breadth_previous_" + uuid.uuid4().hex + ".parquet")
        try:
            if not cache_io.download_to_local(key, str(prior_path)):
                raise RuntimeError("Unable to validate existing canonical history")
            prior = pd.read_parquet(prior_path)
            if not prior.index.isin(current.index).all():
                raise RuntimeError("Publication would lose canonical dates; reconcile the database first")
            historical = prior.index[prior.index < pd.Timestamp(CUTOVER)]
            pd.testing.assert_frame_equal(prior.loc[historical], current.loc[historical])
        finally:
            # Twice-daily automated publication would otherwise litter data/
            # with one comparison copy per run.
            prior_path.unlink(missing_ok=True)
    digest = hashlib.sha256(Path(database).read_bytes()).hexdigest()
    backup_key = f"market_breadth/history/{digest}.sqlite"
    # The immutable backup is written first, so the canonical database key can
    # never be the only copy of a generation.
    for local, target in [(database, backup_key), (database, DB_KEY), (export, key)]:
        if not cache_io.upload_from_local(str(local), target):
            raise RuntimeError(f"R2 publication failed: {target}")
        meta = cache_io.head(target)
        if not meta or meta.get("ContentLength") != Path(local).stat().st_size:
            raise RuntimeError(f"R2 verification failed: {target}")
    print(json.dumps({"published": key, "database": DB_KEY, "database_backup": backup_key,
                      "asof": str(current.index.max().date())}))


if __name__ == "__main__":
    main()
