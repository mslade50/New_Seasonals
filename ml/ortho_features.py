"""
ml/ortho_features.py — Run-3 features orthogonal to the strategy filters.

The run-1/2 finding was that conditional expectancy is flat in the observables
the strategies already condition on. These features come from data the entry
rules do NOT use:

  Market-level (joined by signal date):
    pc_equity, pc_equity_z63     CBOE equity put/call (data/cboe_putcall.parquet;
                                 2024+ only — thin coverage, disclosed)
    naaim_level, naaim_z52       NAAIM manager exposure (naaim.csv, weekly
                                 2006+, ffilled from publication date)

  Ticker-level (trailing event windows, strictly BEFORE the signal date):
    grades_net_21d               upgrades minus downgrades, trailing 21 cal days
    grades_n_63d                 total grade events, trailing 63 cal days
                                 (data/analyst_grades.parquet, 2012+)
    days_since_earn              calendar days since last earnings (< t), cap 126
    days_to_earn                 calendar days to next scheduled earnings (>= t),
                                 cap 126. Forward dates were typically announced
                                 at signal time; approximation disclosed in the
                                 plan. NaN when the ticker has no earnings data.

All windows are trailing or as-of; nothing uses post-signal information beyond
the disclosed scheduled-earnings approximation.
"""

import os

import numpy as np
import pandas as pd

from ml import config

PUTCALL_PATH = os.path.join(config.DATA_DIR, "cboe_putcall.parquet")
NAAIM_PATH = os.path.join(config.REPO_ROOT, "naaim.csv")
GRADES_PATH = os.path.join(config.DATA_DIR, "analyst_grades.parquet")
EARNINGS_PATH = os.path.join(config.DATA_DIR, "earnings_calendar.parquet")
FRAGILITY_PATHS = [os.path.join(config.DATA_DIR, "rd2_fragility.parquet"),
                   os.path.join(config.DATA_DIR, "rd2_fragility_ts.parquet")]

_CACHE = {}


def market_ortho_frame() -> pd.DataFrame:
    """Daily frame of market-level orthogonal features (ffilled, trailing)."""
    if "market" in _CACHE:
        return _CACHE["market"]
    frames = []

    try:
        pc = pd.read_parquet(PUTCALL_PATH)
        pc.index = pd.to_datetime(pc.index).normalize()
        pc = pc[~pc.index.duplicated(keep="last")]
        eq = pd.to_numeric(pc["equity"], errors="coerce").sort_index()
        f = pd.DataFrame(index=eq.index)
        f["pc_equity"] = eq
        mu, sd = eq.rolling(63).mean(), eq.rolling(63).std()
        f["pc_equity_z63"] = (eq - mu) / sd
        frames.append(f)
    except Exception:
        pass

    try:
        na = pd.read_csv(NAAIM_PATH)
        na["Date"] = pd.to_datetime(na["Date"], errors="coerce")
        na = (na.dropna(subset=["Date"]).sort_values("Date")
                .drop_duplicates(subset="Date", keep="last").set_index("Date"))
        lvl = pd.to_numeric(na["naaim"], errors="coerce")
        f = pd.DataFrame(index=lvl.index)
        f["naaim_level"] = lvl
        mu, sd = lvl.rolling(52).mean(), lvl.rolling(52).std()
        f["naaim_z52"] = (lvl - mu) / sd
        frames.append(f)
    except Exception:
        pass

    if frames:
        out = pd.concat(frames, axis=1).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        # Expand to a daily calendar BEFORE ffilling, so signal dates between
        # weekly NAAIM publications resolve. Staleness cap 14 calendar days.
        cal = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(cal).ffill(limit=14)
    else:
        out = pd.DataFrame(columns=["pc_equity", "pc_equity_z63",
                                    "naaim_level", "naaim_z52"])
    _CACHE["market"] = out
    return out


def fragility_frame() -> pd.DataFrame:
    """Risk-dial fragility scores at 5d/21d/63d windows (risk_dashboard_v2
    composite, data/rd2_fragility.parquet, daily 2016-06 onward; the _ts copy
    is the slightly-stale site fallback).

    Run-4 disclosure: the series is a reconstruction by the current dashboard
    code — historical values use today's signal definitions and in-window
    percentile bands (mild within-window lookahead, inherited from the
    existing risk system the same way seasonal ranks are).
    """
    if "fragility" in _CACHE:
        return _CACHE["fragility"]
    out = pd.DataFrame(columns=config.FRAGILITY_FEATURES)
    for path in FRAGILITY_PATHS:
        try:
            fr = pd.read_parquet(path)
            fr.index = pd.to_datetime(fr.index).normalize()
            fr = fr[~fr.index.duplicated(keep="last")].sort_index()
            fr = fr.rename(columns={"5d": "frag_5d", "21d": "frag_21d",
                                    "63d": "frag_63d"})
            cols = [c for c in config.FRAGILITY_FEATURES if c in fr.columns]
            if not cols:
                continue
            cal = pd.date_range(fr.index.min(), fr.index.max(), freq="D")
            out = fr[cols].reindex(cal).ffill(limit=7)
            break
        except Exception:
            continue
    _CACHE["fragility"] = out
    return out


def _event_lookup(dates: np.ndarray, weights: np.ndarray):
    """(sorted event dates, cumsum of weights) for window-sum queries."""
    order = np.argsort(dates, kind="mergesort")
    d = dates[order]
    cum = np.concatenate([[0.0], np.cumsum(weights[order])])
    return d, cum


def _window_sum(d, cum, start, end):
    """Sum of event weights with start <= date < end."""
    lo = np.searchsorted(d, start, side="left")
    hi = np.searchsorted(d, end, side="left")
    return cum[hi] - cum[lo]


def grades_features(trades: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """grades_net_21d / grades_n_63d per trade row (events strictly < t)."""
    out = pd.DataFrame(index=trades.index,
                       columns=["grades_net_21d", "grades_n_63d"], dtype=float)
    try:
        if "grades" not in _CACHE:
            ag = pd.read_parquet(GRADES_PATH)
            ag["date"] = pd.to_datetime(ag["date"]).dt.normalize()
            ag["ticker"] = ag["ticker"].astype(str).str.upper()
            ag["net"] = np.where(ag["action"] == "upgrade", 1.0,
                        np.where(ag["action"] == "downgrade", -1.0, 0.0))
            _CACHE["grades"] = {
                t: (_event_lookup(g["date"].values.astype("datetime64[ns]"),
                                  g["net"].values),
                    _event_lookup(g["date"].values.astype("datetime64[ns]"),
                                  np.ones(len(g))))
                for t, g in ag.groupby("ticker")
            }
        lookups = _CACHE["grades"]
        first_date = pd.Timestamp("2012-01-01")
        for idx, row in trades.iterrows():
            lk = lookups.get(str(row["Ticker"]).upper())
            t = pd.to_datetime(row[date_col])
            if lk is None or pd.isna(t) or t < first_date:
                continue  # pre-coverage stays NaN rather than fake zero
            (d_net, cum_net), (d_n, cum_n) = lk
            t64 = np.datetime64(t)
            out.at[idx, "grades_net_21d"] = _window_sum(
                d_net, cum_net, t64 - np.timedelta64(21, "D"), t64)
            out.at[idx, "grades_n_63d"] = _window_sum(
                d_n, cum_n, t64 - np.timedelta64(63, "D"), t64)
    except Exception:
        pass
    return out


def earnings_distance(trades: pd.DataFrame, date_col: str,
                      cap_days: int = 126) -> pd.DataFrame:
    """days_since_earn (last earnings < t) and days_to_earn (next >= t)."""
    out = pd.DataFrame(index=trades.index,
                       columns=["days_since_earn", "days_to_earn"], dtype=float)
    try:
        if "earnings" not in _CACHE:
            ec = pd.read_parquet(EARNINGS_PATH, columns=["ticker", "date"])
            ec["date"] = pd.to_datetime(ec["date"]).dt.normalize()
            ec["ticker"] = ec["ticker"].astype(str).str.upper()
            _CACHE["earnings"] = {
                t: np.sort(g["date"].values.astype("datetime64[ns]"))
                for t, g in ec.groupby("ticker")
            }
        cal = _CACHE["earnings"]
        for idx, row in trades.iterrows():
            dates = cal.get(str(row["Ticker"]).upper())
            t = pd.to_datetime(row[date_col])
            if dates is None or len(dates) == 0 or pd.isna(t):
                continue  # no earnings data -> NaN (commodity ETFs etc.)
            t64 = np.datetime64(t)
            pos = np.searchsorted(dates, t64, side="left")
            if pos > 0:
                since = (t64 - dates[pos - 1]) / np.timedelta64(1, "D")
                out.at[idx, "days_since_earn"] = min(float(since), cap_days)
            if pos < len(dates):
                to = (dates[pos] - t64) / np.timedelta64(1, "D")
                out.at[idx, "days_to_earn"] = min(float(to), cap_days)
    except Exception:
        pass
    return out


def add_ortho_features(feat: pd.DataFrame, trades: pd.DataFrame,
                       date_col: str) -> pd.DataFrame:
    """Append all orthogonal feature columns to an assembled feature frame."""
    dts = pd.to_datetime(trades[date_col]).dt.normalize()
    mkt = market_ortho_frame().reindex(dts.values)
    mkt.index = trades.index
    for c in ["pc_equity", "pc_equity_z63", "naaim_level", "naaim_z52"]:
        feat[c] = mkt[c] if c in mkt.columns else np.nan
    frag = fragility_frame().reindex(dts.values)
    frag.index = trades.index
    for c in config.FRAGILITY_FEATURES:
        feat[c] = frag[c] if c in frag.columns else np.nan
    feat = pd.concat([feat, grades_features(trades, date_col),
                      earnings_distance(trades, date_col)], axis=1)
    return feat
