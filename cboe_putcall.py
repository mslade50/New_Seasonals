"""
CBOE Put/Call Ratio scraper + parquet cache.

CBOE's daily market-statistics page embeds the day's ratio summary as escaped
JSON inside the server-rendered HTML at
`https://www.cboe.com/us/options/market_statistics/daily/?dt=YYYY-MM-DD`.

This module pulls that snapshot, normalizes the ratio fields, and incrementally
backfills missing trading days into `data/cboe_putcall.parquet`.

Columns produced (one row per date):
  total, index, equity, etp, spx, oex
"""

from __future__ import annotations

import os
import re
import time
import datetime as dt
import urllib.request
import urllib.error
from typing import Iterable

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(_DATA_DIR, "cboe_putcall.parquet")

_URL = "https://www.cboe.com/us/options/market_statistics/daily/?dt={dt}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cboe.com/us/options/market_statistics/daily/",
}

# Field-name substring (in CBOE label) -> short column name
_FIELD_MAP = {
    "TOTAL PUT/CALL RATIO": "total",
    "INDEX PUT/CALL RATIO": "index",
    "EQUITY PUT/CALL RATIO": "equity",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp",
    "SPX + SPXW PUT/CALL RATIO": "spx",
    "OEX PUT/CALL RATIO": "oex",
}

# CBOE HTML embeds JSON with the quotes escaped: \"name\":\"...\",\"value\":\"...\"
_PAIR_RE = re.compile(r'\\"name\\":\\"([^"\\]+)\\",\\"value\\":\\"([^"\\]+)\\"')


def _fetch_day(date: dt.date, retries: int = 2, timeout: float = 12.0) -> dict | None:
    """
    Pull one date's snapshot. Returns dict of short_column -> float, or None if
    the page can't be reached / contains no parseable fields (weekend, holiday,
    future date).
    """
    url = _URL.format(dt=date.isoformat())
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="ignore")
            row: dict[str, float] = {}
            for m in _PAIR_RE.finditer(body):
                label = m.group(1).strip().upper()
                if label in _FIELD_MAP:
                    try:
                        row[_FIELD_MAP[label]] = float(m.group(2))
                    except ValueError:
                        continue
            return row if row else None
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    if last_err:
        print(f"[cboe_putcall] {date}: {type(last_err).__name__} {last_err}")
    return None


def _load() -> pd.DataFrame:
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame(columns=list(_FIELD_MAP.values())).rename_axis("date")
    df = pd.read_parquet(CACHE_PATH)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _save(df: pd.DataFrame) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    df = df.sort_index()
    df.to_parquet(CACHE_PATH)


def _trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    rng = pd.bdate_range(start=start, end=end)
    return [d.date() for d in rng]


def backfill(start: str | dt.date, end: str | dt.date | None = None,
             sleep_between: float = 0.4, progress=None,
             max_days: int | None = None) -> pd.DataFrame:
    """
    Fetch any business days in [start, end] missing from the parquet cache.

    progress(done, total, date) optional callback for UI hooks.
    Returns the full cached frame (sorted, ascending).
    """
    if isinstance(start, str):
        start = pd.Timestamp(start).date()
    if end is None:
        end = dt.date.today()
    elif isinstance(end, str):
        end = pd.Timestamp(end).date()

    df = _load()
    have = set(df.index.date) if not df.empty else set()
    targets = [d for d in _trading_days(start, end) if d not in have]
    if max_days is not None:
        targets = targets[:max_days]

    new_rows: dict[dt.date, dict] = {}
    for i, d in enumerate(targets):
        row = _fetch_day(d)
        if row:
            new_rows[d] = row
        if progress:
            try:
                progress(i + 1, len(targets), d)
            except Exception:
                pass
        time.sleep(sleep_between)

    if new_rows:
        add = pd.DataFrame.from_dict(new_rows, orient="index")
        add.index = pd.to_datetime(add.index)
        add.index.name = "date"
        df = add if df.empty else pd.concat([df, add])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _save(df)

    return df


def load_series(start: str | dt.date | None = None,
                column: str = "index") -> pd.Series:
    """
    Return one column from the cache as a Series. Does NOT backfill — call
    backfill() first if fresh data is needed.
    """
    df = _load()
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float, name=column)
    s = df[column].copy()
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    s.name = column
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    def _p(done, total, d):
        print(f"  [{done}/{total}] {d}", flush=True)

    df = backfill(args.start, args.end, sleep_between=args.sleep,
                  progress=_p, max_days=args.max_days)
    print(f"Cache: {len(df)} rows, columns={list(df.columns)}")
    if not df.empty:
        print(df.tail(5))
