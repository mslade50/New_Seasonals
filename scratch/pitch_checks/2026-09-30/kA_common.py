"""Shared helpers for the kA_ checks on 2026-09-30 (calendar anchors ON the
month-end / quarter-end close; entry = Close[ME-0], so lag=0 from the anchor)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def own_series(t: str) -> pd.Series:
    return close_panel([t])[t].dropna()


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of each COMPLETE month on this calendar (the month of the
    last bar is dropped: Sep 2026's last bar is 09-29, not the ME)."""
    s = pd.Series(idx, index=idx)
    me = pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)
    last = idx[-1]
    return me[~((me.year == last.year) & (me.month == last.month))]


def fwd(c: pd.Series, anchors, h: int, start: int = 0) -> pd.Series:
    """C[p+start+h]/C[p+start]-1 on c's own calendar, indexed by anchor."""
    idx = c.index
    pos = pd.Series(range(len(idx)), index=idx)
    out = {}
    for d in pd.DatetimeIndex(anchors):
        p = pos.get(d)
        if p is None:
            continue
        a, b = p + start, p + start + h
        if a < 0 or b >= len(idx):
            continue
        out[d] = c.iloc[b] / c.iloc[a] - 1.0
    return pd.Series(out, dtype=float)


def all_windows(c: pd.Series, h: int) -> pd.Series:
    return (c.shift(-h) / c - 1.0).dropna()


def rec(v: pd.Series, label: str) -> dict:
    v = pd.Series(v).dropna()
    s = summarize(v.values, label)
    if s["n"]:
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v) - w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 100 * d, d / se


def eras(v: pd.Series, label: str, cuts=("2018-01-01",)) -> list[dict]:
    v = pd.Series(v).dropna()
    out = []
    edges = [pd.Timestamp("1900-01-01")] + [pd.Timestamp(c) for c in cuts] + [pd.Timestamp("2100-01-01")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v.index >= lo) & (v.index < hi)
        out.append(rec(v[m], f"{label} {lo.year if lo.year > 1900 else 'start'}..{hi.year - 1 if hi.year < 2100 else 'now'}"))
    return out


def asof_on(src: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Value of src at-or-before each date in idx (PIT forward fill)."""
    s = src.dropna()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def nfp_dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load_events(["nfp"])["date"])
