"""Shared falsification machinery for the 2026-08-07 metals/energy candidates.

Read-only. Conventions used by every C5-C8 script:
  * signal is measured on close D; the trade ENTERS at close D+`lag` (lag=1 = the
    real MOC-tomorrow order) and exits `h` sessions later. fwd(D) is therefore
    P[D+lag+h]/P[D+lag] - 1.
  * pair return = r_long - r_short (1x each leg, gross 2x notional).
  * two controls: ALL-DAYS (whole overlap window) and LOCAL (every day within
    +/-126 td of some trigger, trigger days removed) -- the local one kills the
    "triggers just live in a good regime" explanation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd


def fwd_lag(s: pd.Series, h: int, lag: int = 1) -> pd.Series:
    """Forward return entering at close D+lag, exiting h sessions after that."""
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def pair_fwd(P: pd.DataFrame, long_t: str, short_t: str, h: int, lag: int = 1) -> pd.Series:
    return fwd_lag(P[long_t], h, lag) - fwd_lag(P[short_t], h, lag)


def local_control(all_dates: pd.DatetimeIndex, trig: pd.DatetimeIndex, win: int = 126) -> pd.DatetimeIndex:
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    keep = np.zeros(len(all_dates), dtype=bool)
    for d in trig:
        p = pos.get(d)
        if p is None:
            continue
        keep[max(0, p - win):min(len(all_dates), p + win + 1)] = True
    out = all_dates[keep]
    return out.difference(trig)


def report(fw: pd.Series, trig: pd.DatetimeIndex, label: str, h: int) -> list[dict]:
    """Conditional row + the two control rows."""
    valid = fw.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    rows = [summarize(fw.loc[t].values, f"{label} COND h{h}")]
    rows.append(summarize(fw.loc[valid].values, f"{label} ctrl ALL-DAYS"))
    loc = local_control(valid, t).intersection(valid)
    rows.append(summarize(fw.loc[loc].values, f"{label} ctrl LOCAL +/-126td"))
    return rows


def episodes(fw: pd.Series, trig: pd.DatetimeIndex, h: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    valid = fw.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    ep = declusters(t, h, valid)
    return ep, fw.loc[ep].values


def cluster_note(dates: pd.DatetimeIndex, vals: np.ndarray, k: int = 2) -> str:
    """How much of total return sits in the top-k episodes?"""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "n/a"
    tot = v.sum()
    order = np.argsort(-np.abs(v))[:k]
    top = v[order].sum()
    yrs = pd.DatetimeIndex(dates).year
    by_yr = pd.Series(v).groupby(yrs.values).sum().sort_values(ascending=False)
    share = (top / tot * 100) if tot != 0 else np.nan
    return (f"top{k} episodes {[str(pd.Timestamp(dates[i]).date()) for i in order]} "
            f"= {100*top:+.2f}pp of {100*tot:+.2f}pp total ({share:.0f}%); "
            f"best yrs {dict((k2, round(100*v2, 1)) for k2, v2 in by_yr.head(2).items())}")


def cpi_in_window(anchor: pd.DatetimeIndex, all_dates: pd.DatetimeIndex, h: int,
                  lag: int = 1, kinds=("cpi",)) -> np.ndarray:
    """True when a CPI (or given event) print lands inside the held window."""
    ev = load_events(list(kinds))["date"].values.astype("datetime64[ns]")
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    out = []
    for d in anchor:
        p = pos.get(d)
        if p is None or p + lag + h >= len(all_dates):
            out.append(False)
            continue
        lo, hi = all_dates[p + lag], all_dates[p + lag + h]
        out.append(bool(((ev > np.datetime64(lo)) & (ev <= np.datetime64(hi))).any()))
    return np.asarray(out)
