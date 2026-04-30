"""
Earnings blackout helpers — shared by daily_scan, local_overflow_scan, and
pages/strat_backtester. Mirrors the trading-day arithmetic in
pages/backtester.compute_signed_earnings_offsets so backtest, scan, and report
all see the same offset for a given (ticker, signal_date).

Source: data/earnings_calendar.parquet (built by scripts/build_earnings_calendar.py).

Convention (matches pages/backtester.py):
    offset = signal_date - earnings_date in trading days
        positive → signal is AFTER earnings
        negative → signal is BEFORE earnings
        0        → signal IS the earnings day
        NaN      → ticker has no earnings data (commodity ETF, futures, index, FX)

NaN-as-True behavior: tickers with no earnings data PASS the blackout filter,
mirroring the `Not Between` operator in pages/backtester.py. Commodity ETFs,
indices, futures, and FX should never be silently killed by a stock-only filter.
"""

import os
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "earnings_calendar.parquet",
)

# Cached holidays_d64 used by np.busday_count — same calendar as pages/backtester.
_HOLIDAYS_D64 = pd.DatetimeIndex(
    USFederalHolidayCalendar().holidays(start="1990-01-01", end="2035-12-31")
).to_numpy().astype("datetime64[D]")


def load_earnings_dates_map(path=None):
    """Load earnings calendar parquet → {ticker: np.array of datetime64[D]}.

    Returns empty dict if the parquet is missing or malformed — callers should
    treat that as "filter off" (every ticker passes through).
    """
    p = path or _PARQUET_PATH
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
    out = {}
    for tkr, grp in df.groupby("ticker"):
        sorted_dates = np.sort(
            pd.DatetimeIndex(grp["date"].unique()).to_numpy().astype("datetime64[D]")
        )
        out[tkr] = sorted_dates
    return out


def signed_offset(signal_date, earnings_arr):
    """Trading-day offset from signal_date to nearest earnings.

    Returns NaN when earnings_arr is None/empty (ticker has no data).
    Positive = after earnings, negative = before, 0 = day-of.
    """
    if earnings_arr is None or len(earnings_arr) == 0:
        return float("nan")
    d64 = np.array([pd.Timestamp(signal_date).normalize()], dtype="datetime64[D]")
    pos = np.searchsorted(earnings_arr, d64, side="right")[0]

    past_off = None
    if pos > 0:
        past_e = earnings_arr[pos - 1]
        past_off = int(np.busday_count(past_e, d64[0], holidays=_HOLIDAYS_D64))
    future_off = None
    if pos < len(earnings_arr):
        future_e = earnings_arr[pos]
        future_off = -int(np.busday_count(d64[0], future_e, holidays=_HOLIDAYS_D64))

    if past_off is None and future_off is None:
        return float("nan")
    if past_off is None:
        return float(future_off)
    if future_off is None:
        return float(past_off)
    return float(past_off if abs(past_off) <= abs(future_off) else future_off)


def in_blackout(signal_date, earnings_arr, window=None, lo=None, hi=None):
    """Return True if signal_date sits inside an earnings blackout window.

    Two ways to specify the window:
        symmetric:   window=N  → reject if |offset| <= N (e.g. ±10 TD around earnings)
        asymmetric:  lo=L, hi=H → reject if L <= offset <= H
                                  (e.g. lo=-10, hi=0 blocks 10 TD before earnings
                                  through the announcement day, but allows post-earnings)

    NaN (ticker has no earnings data — commodity ETFs, indices, futures) → False
    (pass through), mirroring `Not Between` semantics in pages/backtester.py.

    If neither `window` nor (lo, hi) is provided, returns False (no blackout).
    """
    off = signed_offset(signal_date, earnings_arr)
    if pd.isna(off):
        return False
    if window is not None:
        return abs(off) <= window
    if lo is not None and hi is not None:
        return lo <= off <= hi
    return False


def signed_offset_series(df_dates, earnings_dates_arr):
    """Vectorized version for a pd.DatetimeIndex of signal dates.

    Used by strat_backtester to compute offsets across all bars in one ticker.
    Returns a pd.Series of float (NaN where no earnings data). Mirrors
    pages/backtester.compute_signed_earnings_offsets.
    """
    if earnings_dates_arr is None or len(earnings_dates_arr) == 0:
        return pd.Series(np.nan, index=df_dates)
    d64 = pd.DatetimeIndex(df_dates).to_numpy().astype("datetime64[D]")
    e_sorted = np.asarray(earnings_dates_arr, dtype="datetime64[D]")
    pos = np.searchsorted(e_sorted, d64, side="right")

    past_mask = pos > 0
    past_e = e_sorted[np.clip(pos - 1, 0, len(e_sorted) - 1)]
    past_off = np.where(
        past_mask,
        np.busday_count(past_e, d64, holidays=_HOLIDAYS_D64),
        2**31 - 1,
    ).astype(np.int64)

    future_mask = pos < len(e_sorted)
    future_e = e_sorted[np.clip(pos, 0, len(e_sorted) - 1)]
    future_off = np.where(
        future_mask,
        -np.busday_count(d64, future_e, holidays=_HOLIDAYS_D64),
        -(2**31 - 1),
    ).astype(np.int64)

    use_past = np.abs(past_off) <= np.abs(future_off)
    nearest = np.where(use_past, past_off, future_off).astype(float)
    no_data = ~past_mask & ~future_mask
    nearest[no_data] = np.nan
    return pd.Series(nearest, index=df_dates)
