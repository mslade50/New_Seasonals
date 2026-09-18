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
import sys
import time
import datetime as dt
import urllib.request
import urllib.error
from typing import Iterable

import numpy as np
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


# ---------------------------------------------------------------------------
# Row validity guard (2026-09-18)
#
# The scraper used to accept whatever the daily page returned. Two ways that
# goes wrong, both observed in the cache:
#   * a date that was never an NYSE session at all. 2025-01-09 (the Carter day
#     of mourning) served a page with equity 0.00, which entered the parquet
#     and dragged the pc_fear 10d MA down for two trading weeks.
#   * a ratio outside anything the series has ever printed, which is a parse
#     or feed artifact rather than a market reading.
# The guard runs on every freshly scraped row AND as a purge over the cache on
# load, so an already-polluted copy (local or the R2 canonical pulled each
# morning) heals itself on the next run. It never raises and never rewrites an
# existing good row; a rejection is one loud line.
# ---------------------------------------------------------------------------

# Dates the NYSE was closed for reasons no calendar rule produces.
NYSE_SPECIAL_CLOSURES: frozenset[dt.date] = frozenset({
    dt.date(2025, 1, 9),    # national day of mourning, Jimmy Carter
    dt.date(2018, 12, 5),   # national day of mourning, George H. W. Bush
    dt.date(2012, 10, 29),  # Hurricane Sandy
    dt.date(2012, 10, 30),  # Hurricane Sandy
    dt.date(2007, 1, 2),    # national day of mourning, Gerald Ford
    dt.date(2004, 6, 11),   # national day of mourning, Ronald Reagan
    dt.date(2001, 9, 11),
    dt.date(2001, 9, 12),
    dt.date(2001, 9, 13),
    dt.date(2001, 9, 14),
})

# Sane band for the equity put/call ratio. Measured over the full cache
# (2006-11 to 2026-09, 5000 rows): min 0.32 (2010-04-14), max 2.40
# (2022-12-28), and the 2008 panic tops out at 1.34. The band sits well
# outside both tails so a genuine capitulation print still passes, while a
# zero, a negative, or an order-of-magnitude parse artifact does not.
EQUITY_MIN = 0.15
EQUITY_MAX = 3.0

# Good Friday is not a federal holiday, so it has to be computed. dateutil is
# a pandas dependency in practice; the table is the belt-and-braces path.
try:  # pragma: no cover - import shape, not logic
    from dateutil.easter import easter as _easter
except ImportError:  # pragma: no cover
    _easter = None

_GOOD_FRIDAY_FALLBACK: dict[int, tuple[int, int]] = {
    2007: (4, 6), 2008: (3, 21), 2009: (4, 10), 2010: (4, 2), 2011: (4, 22),
    2012: (4, 6), 2013: (3, 29), 2014: (4, 18), 2015: (4, 3), 2016: (3, 25),
    2017: (4, 14), 2018: (3, 30), 2019: (4, 19), 2020: (4, 10), 2021: (4, 2),
    2022: (4, 15), 2023: (4, 7), 2024: (3, 29), 2025: (4, 18), 2026: (4, 3),
    2027: (3, 26), 2028: (4, 14), 2029: (3, 30), 2030: (4, 19),
}

_NYSE_HOLIDAY_CACHE: dict[int, frozenset] = {}


def _good_friday(year: int) -> dt.date | None:
    if _easter is not None:
        return _easter(year) - dt.timedelta(days=2)
    md = _GOOD_FRIDAY_FALLBACK.get(year)
    return dt.date(year, md[0], md[1]) if md else None


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    """NYSE observance: Saturday rolls back to Friday, Sunday forward to Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(year: int) -> frozenset:
    """NYSE holiday set for one year.

    Deliberately NOT USFederalHolidayCalendar: Columbus Day and Veterans Day
    are federal holidays on which the NYSE trades normally, and the federal
    calendar has no Good Friday. Using it as-is would reject ~2 real sessions
    a year and accept every Good Friday.
    """
    cached = _NYSE_HOLIDAY_CACHE.get(year)
    if cached is not None:
        return cached
    days: set[dt.date] = set()
    # New Year's Day. A Sunday rolls to Monday; a Saturday is NOT pulled back
    # to the preceding Friday, because that Friday belongs to the prior year
    # and the NYSE stays open (e.g. 2021-12-31).
    new_year = dt.date(year, 1, 1)
    if new_year.weekday() != 5:
        days.add(_observed(new_year))
    days.add(_nth_weekday(year, 1, 0, 3))    # MLK Day, 3rd Monday
    days.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday, 3rd Monday
    good_friday = _good_friday(year)
    if good_friday is not None:
        days.add(good_friday)
    days.add(_last_weekday(year, 5, 0))      # Memorial Day, last Monday
    if year >= 2022:                         # Juneteenth, NYSE from 2022
        days.add(_observed(dt.date(year, 6, 19)))
    days.add(_observed(dt.date(year, 7, 4)))  # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))    # Labor Day, 1st Monday
    days.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving, 4th Thursday
    days.add(_observed(dt.date(year, 12, 25)))  # Christmas
    frozen = frozenset(days)
    _NYSE_HOLIDAY_CACHE[year] = frozen
    return frozen


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def session_reject_reason(value) -> str | None:
    """None when the date is an NYSE session, else a short reason."""
    d = _as_date(value)
    if d.weekday() >= 5:
        return "not an NYSE session (weekend)"
    if d in NYSE_SPECIAL_CLOSURES:
        return "not an NYSE session (special closure)"
    if d in nyse_holidays(d.year):
        return "not an NYSE session (NYSE holiday)"
    return None


def _equity_reject_reason(equity) -> str | None:
    try:
        equity = float(equity)
    except (TypeError, ValueError):
        return "equity missing"
    if not np.isfinite(equity):
        return "equity missing"
    if equity <= 0:
        return f"equity {equity:.2f} <= 0"
    if equity < EQUITY_MIN or equity > EQUITY_MAX:
        return (f"equity {equity:.2f} outside sane band "
                f"[{EQUITY_MIN}, {EQUITY_MAX}]")
    return None


def row_rejection_reason(date, row) -> str | None:
    """None when (date, row) is a keepable cache row, else a short reason."""
    reason = session_reject_reason(date)
    if reason:
        return reason
    equity = row.get("equity") if hasattr(row, "get") else None
    return _equity_reject_reason(equity)


def _log_rejection(date, reason: str) -> None:
    print(f"[cboe_putcall] REJECTED {_as_date(date)} {reason}", flush=True)


def purge_invalid(df: pd.DataFrame, verbose: bool = True):
    """Drop every cache row the guard rejects.

    Returns (clean_frame, [(date, reason), ...]). Dtypes, column order and the
    index name survive because this is a boolean mask, not a rebuild.
    """
    dropped: list[tuple[dt.date, str]] = []
    if df is None or df.empty:
        return df, dropped
    dates = [_as_date(ts) for ts in df.index]
    if "equity" in df.columns:
        equity = pd.to_numeric(df["equity"], errors="coerce").to_numpy(dtype=float)
    else:
        equity = np.full(len(df), np.nan)
    keep = np.ones(len(df), dtype=bool)
    for i, d in enumerate(dates):
        reason = session_reject_reason(d) or _equity_reject_reason(equity[i])
        if reason is None:
            continue
        keep[i] = False
        dropped.append((d, reason))
        if verbose:
            _log_rejection(d, reason)
    if not dropped:
        return df, dropped
    return df.loc[keep], dropped


def _parse_body(body: str) -> dict[str, float]:
    """Extract the mapped ratio fields from the page HTML. Empty dict if the
    page has no parseable fields (weekend, holiday, future date — or a CBOE
    markup change, which the workflow's freshness assertion turns loud)."""
    row: dict[str, float] = {}
    for m in _PAIR_RE.finditer(body):
        label = m.group(1).strip().upper()
        if label in _FIELD_MAP:
            try:
                row[_FIELD_MAP[label]] = float(m.group(2))
            except ValueError:
                continue
    return row


def freshness_age_bdays(df: pd.DataFrame, asof: dt.date | None = None) -> int | None:
    """Business-day age of the newest cached row (None on an empty cache).
    Naive bday count — a single market holiday inflates the age by 1, so
    thresholds should tolerate age 2 (normal steady state is 1: the daily
    page for D populates after the 21:30 UTC run, so each run captures D-1)."""
    if df.empty:
        return None
    if asof is None:
        asof = dt.date.today()
    return int(np.busday_count(df.index.max().date(), asof))


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
            row = _parse_body(body)
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


def _read_cache() -> pd.DataFrame:
    """Raw parquet read, guard not applied."""
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame(columns=list(_FIELD_MAP.values())).rename_axis("date")
    df = pd.read_parquet(CACHE_PATH)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_checked() -> tuple[pd.DataFrame, list]:
    """Cache with the row guard applied. Returns (frame, dropped rows)."""
    return purge_invalid(_read_cache())


def _load() -> pd.DataFrame:
    return _load_checked()[0]


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

    # The guard runs on load too, so a polluted cache (local, or the R2 copy
    # pulled at the top of the morning job) heals itself here and the healed
    # frame is what gets written back and published.
    df, purged = _load_checked()
    have = set(df.index.date) if not df.empty else set()
    targets = [d for d in _trading_days(start, end)
               if d not in have and session_reject_reason(d) is None]
    if max_days is not None:
        targets = targets[:max_days]

    new_rows: dict[dt.date, dict] = {}
    for i, d in enumerate(targets):
        row = _fetch_day(d)
        if row:
            reason = row_rejection_reason(d, row)
            if reason is None:
                new_rows[d] = row
            else:
                _log_rejection(d, reason)
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
        # keep='first' so an existing good row is never overwritten by a
        # re-fetch; new dates are the only ones fetched anyway.
        df = df[~df.index.duplicated(keep="first")].sort_index()

    if new_rows or purged:
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
    ap.add_argument("--assert-fresh-bd", type=int, default=None,
                    help="exit 1 if the newest cached row is older than this "
                         "many business days after the backfill")
    args = ap.parse_args()

    def _p(done, total, d):
        print(f"  [{done}/{total}] {d}", flush=True)

    df = backfill(args.start, args.end, sleep_between=args.sleep,
                  progress=_p, max_days=args.max_days)
    print(f"Cache: {len(df)} rows, columns={list(df.columns)}")
    if not df.empty:
        print(df.tail(5))
    if args.assert_fresh_bd is not None:
        age = freshness_age_bdays(df)
        if age is None or age > args.assert_fresh_bd:
            print(f"[cboe_putcall] STALE: newest row is "
                  f"{'missing' if age is None else f'{age} bdays old'} "
                  f"(threshold {args.assert_fresh_bd}) - scrape or parser "
                  f"is likely broken.")
            sys.exit(1)
        print(f"[cboe_putcall] freshness OK ({age} bdays)")
