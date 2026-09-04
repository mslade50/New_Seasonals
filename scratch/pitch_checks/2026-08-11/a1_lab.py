"""Shared helpers for the a1_ adversarial checks (2026-08-11).

Everything here is anchor plumbing that all four C1 scripts need; the
statistics themselves come from pitch_lab.  Nothing is re-derived that
pitch_lab already owns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
)

# ProShares cut SVXY from -1x to -0.5x daily; announced 2018-02-27,
# effective with the 2018-02-28 session.
SVXY_LEV_BREAK = pd.Timestamp("2018-02-28")


def tdom_of(idx: pd.DatetimeIndex) -> pd.Series:
    """1-indexed trading day of month, matching 01_event_class_recon.py."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).cumcount() + 1


def anchor_dates(ev: pd.DataFrame, kind: str, offset: int,
                 all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates exactly `offset` sessions before an event of `kind`."""
    dates = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == kind, "date"].unique()))
    out = []
    for d in dates:
        loc = all_dates.searchsorted(d)
        if loc >= len(all_dates):
            continue
        j = loc - offset
        if 0 <= j < len(all_dates):
            out.append(all_dates[j])
    return pd.DatetimeIndex(sorted(set(out)))


def event_sessions(ev: pd.DataFrame, kind: str,
                   all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The event's own trading session (next session if it fell on a holiday)."""
    dates = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == kind, "date"].unique()))
    out = []
    for d in dates:
        loc = all_dates.searchsorted(d)
        if loc < len(all_dates):
            out.append(all_dates[loc])
    return pd.DatetimeIndex(sorted(set(out)))


def tdom_control(f: pd.Series, anchors: pd.DatetimeIndex,
                 tdom: pd.Series, all_dates: pd.DatetimeIndex,
                 pos: pd.Series) -> pd.Series:
    """Trading-day-of-month matched control on the ENTRY session (anchor+1),
    exactly the 01_event_class_recon.py construction."""
    fa = f.dropna()
    ent_tdom = tdom.reindex(all_dates)[pos[anchors].values + 1].values
    m = tdom.reindex(fa.index).isin(set(ent_tdom.tolist())) & ~fa.index.isin(anchors)
    return fa[m]


def rebase_half(close: pd.Series, break_date: pd.Timestamp = SVXY_LEV_BREAK
                ) -> pd.Series:
    """Return a synthetic CONSTANT -0.5x SVXY price series: halve every daily
    return before the leverage break, leave the -0.5x era untouched.  This is
    the only honest basis for pooling across 2018-02, and it is the same
    convention svxy_postevent_grid.py used (synthetic -0.5x legs from UVXY)."""
    r = close.pct_change()
    r_adj = r.where(close.index >= break_date, r * 0.5)
    return (1.0 + r_adj.fillna(0.0)).cumprod() * float(close.iloc[0])


def loyo(dates: pd.DatetimeIndex, vals: np.ndarray) -> pd.DataFrame:
    """Leave-one-year-out means (pp) and the floor."""
    d = pd.DatetimeIndex(dates)
    v = np.asarray(vals, float)
    rows = []
    for y in sorted(set(d.year)):
        m = d.year != y
        if m.sum() < 5:
            continue
        rows.append({"drop_year": y, "n": int(m.sum()),
                     "mean_pct": 100 * v[m].mean(),
                     "in_year_pct": 100 * v[~m].mean(),
                     "in_year_n": int((~m).sum())})
    return pd.DataFrame(rows)
