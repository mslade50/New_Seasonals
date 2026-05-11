"""
Analyst grades loader + trailing-window helpers — shared by the backtester
and any scanner that wants to filter on rating action flow.

Source: data/analyst_grades.parquet (built by scripts/build_analyst_grades.py
from FMP /stable/grades). One row per grading action with action in
{upgrade, downgrade, maintain}. R2-mirrored at analyst_grades.parquet.

Usage
-----
    from analyst_grades import load_grades_map, trailing_counts

    grades = load_grades_map()                    # {ticker: DataFrame}
    counts = trailing_counts(grades.get('AAPL'),  # arr of (date, ups, downs, net)
                             signal_dates,
                             window_days=30)

The trailing helper is vectorized for one ticker at a time: pass the ticker's
grade DataFrame and a DatetimeIndex of signal dates, get back per-date upgrade
/ downgrade / net counts using calendar-day lookback.
"""

import os
import time
import numpy as np
import pandas as pd

_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "analyst_grades.parquet",
)

# Same staleness threshold as earnings_filter — nightly GHA refresh writes
# both R2 + (locally) disk around 21:30 UTC, so anything older than ~18h on a
# weekday means we should grab the latest from R2 if creds are available.
_STALE_AFTER_SECONDS = 18 * 3600


def _refresh_from_r2_if_needed(local_path):
    """Pull data/analyst_grades.parquet from R2 when missing or stale."""
    try:
        from cache_io import is_configured, download_to_local
    except ImportError:
        return
    if not is_configured():
        return
    needs_pull = False
    if not os.path.exists(local_path):
        needs_pull = True
    else:
        age = time.time() - os.path.getmtime(local_path)
        if age > _STALE_AFTER_SECONDS:
            needs_pull = True
    if needs_pull:
        download_to_local("analyst_grades.parquet", local_path)


def load_grades_map(path=None):
    """Load grades parquet -> {ticker: DataFrame[date, action]} sorted by date.

    Returns empty dict if the parquet is missing or malformed.
    """
    p = path or _PARQUET_PATH
    _refresh_from_r2_if_needed(p)
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}
    if df.empty or "ticker" not in df.columns or "date" not in df.columns:
        return {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.lower().str.strip()
    out = {}
    for tkr, grp in df.groupby("ticker"):
        out[tkr] = grp[["date", "action"]].sort_values("date").reset_index(drop=True)
    return out


def trailing_counts(ticker_grades, signal_dates, window_days=30):
    """For each signal_date, count grading actions in (signal_date - window, signal_date].

    Window is inclusive of signal_date and exclusive of (signal_date - window) so
    a 30d lookback covers exactly 30 calendar days of look-back. We use calendar
    days because analyst actions happen on any day, not just trading days.

    Args:
        ticker_grades: DataFrame with date + action cols (one ticker only) or None.
        signal_dates:  pd.DatetimeIndex of dates to compute counts at.
        window_days:   int lookback window in calendar days.

    Returns:
        DataFrame indexed by signal_dates with cols: upgrades, downgrades, net.
        net = upgrades - downgrades. Maintains are not counted (rating unchanged).
        Returns all-zero DataFrame if ticker_grades is None/empty.
    """
    idx = pd.DatetimeIndex(signal_dates).normalize()
    if ticker_grades is None or len(ticker_grades) == 0:
        return pd.DataFrame(
            {"upgrades": 0, "downgrades": 0, "net": 0},
            index=idx, dtype="int64",
        )

    g = ticker_grades.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy().astype("datetime64[D]")
    is_up = (g["action"].to_numpy() == "upgrade").astype(np.int64)
    is_dn = (g["action"].to_numpy() == "downgrade").astype(np.int64)

    # Cumulative upgrade / downgrade counts at each grade event date (inclusive).
    cum_up = np.cumsum(is_up)
    cum_dn = np.cumsum(is_dn)

    sd = idx.to_numpy().astype("datetime64[D]")
    sd_lo = sd - np.timedelta64(window_days, "D")

    # Count in (sd_lo, sd] = cum_at(sd) - cum_at(sd_lo).
    # cum_at(d) = cum_*[searchsorted(dates, d, side='right') - 1] when in range, else 0.
    def _cum_at(target_d, cum_arr):
        pos = np.searchsorted(dates, target_d, side="right") - 1
        out = np.where(pos >= 0, cum_arr[np.clip(pos, 0, len(cum_arr) - 1)], 0)
        return out.astype(np.int64)

    up_hi = _cum_at(sd, cum_up)
    up_lo = _cum_at(sd_lo, cum_up)
    dn_hi = _cum_at(sd, cum_dn)
    dn_lo = _cum_at(sd_lo, cum_dn)

    ups = up_hi - up_lo
    dns = dn_hi - dn_lo
    return pd.DataFrame(
        {"upgrades": ups, "downgrades": dns, "net": ups - dns},
        index=idx,
    )


def trailing_counts_at(ticker_grades, signal_date, window_days=30):
    """Scalar shortcut: returns (upgrades, downgrades, net) for one date."""
    out = trailing_counts(ticker_grades, pd.DatetimeIndex([signal_date]), window_days)
    row = out.iloc[0]
    return int(row["upgrades"]), int(row["downgrades"]), int(row["net"])
