"""Shared helpers for the kC_ checks (c6/c7/c8), 2026-09-21. pitch_lab conventions:
lag=1 entry (MOC the session after the signal close), fractions in / percent out."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
BAR = pd.Timestamp("2026-09-18")


def nyse_panel(tickers: list[str], ffill: tuple[str, ...] = ()) -> pd.DataFrame:
    raw = close_panel(sorted(set(tickers) | {"SPY"}))
    idx = raw["SPY"].dropna().index
    px = raw.reindex(idx).copy()
    for t in ffill:
        # FX / DX trade some NYSE holidays and vice versa: take the last print
        full = raw[t].dropna()
        px[t] = full.reindex(idx.union(full.index)).ffill(limit=5).reindex(idx)
    return px.loc[:BAR]


def dret(s: pd.Series) -> pd.Series:
    return s / s.shift(1) - 1.0


def roll_beta(ry: pd.Series, rx: pd.Series, n: int = 252) -> pd.Series:
    cov = ry.rolling(n, min_periods=n // 2).cov(rx)
    var = rx.rolling(n, min_periods=n // 2).var()
    return cov / var


def resid_index(px: pd.DataFrame, leg: str, hedge: str = "SPY", n: int = 252):
    """Daily-rebalanced long leg / short beta*hedge index, beta known at t-1."""
    ry, rx = dret(px[leg]), dret(px[hedge])
    b = roll_beta(ry, rx, n).shift(1)
    rr = ry - b * rx
    idx = (1 + rr.fillna(0)).cumprod()
    idx[rr.isna() & (idx.index < rr.first_valid_index())] = np.nan
    return idx, rr, b


def to_sessions(dates, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map calendar dates onto the session index: exact, else the prior session."""
    out = []
    for d in pd.DatetimeIndex(dates):
        if d < idx[0] or d > idx[-1]:
            continue
        p = idx.searchsorted(d, side="right") - 1
        if p >= 0:
            out.append(idx[p])
    return pd.DatetimeIndex(sorted(set(out)))


def cellstats(px, mask, legs, h, label, gap=None, lag=1, cost_bps=None, base_mask=None):
    """Episode-level stats; record vs coin and vs the vehicle's own up-rate."""
    r = vehicle_ret(px, legs, h, lag)
    v = r.notna()
    m = mask.reindex(px.index, fill_value=False).values & v.values
    days = px.index[m]
    if len(days) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, gap or max(h, 1), px.index)
    vals = r.loc[epi].values
    base = r[v] if base_mask is None else r[v & base_mask.reindex(px.index, fill_value=False)]
    w = int((vals > 0).sum())
    p0 = float((base > 0).mean())
    out = summarize(vals, label)
    out["ctl_pct"] = 100 * base.mean()
    out["edge_pp"] = out["mean_pct"] - 100 * base.mean()
    out["rec"] = f"{w}-{len(vals)-w}"
    out["p_coin"] = round(sign_test(w, len(vals)), 4)
    out["p_base"] = round(sign_test(w, len(vals), p0), 4)
    if cost_bps:
        out["x_cost"] = round(100 * out["mean_pct"] / (cost_bps * len(legs)), 2)
    return out, epi, vals


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se
