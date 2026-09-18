"""Shared helpers for the kC_ checks (c6/c7/c8), 2026-09-18. pitch_lab conventions:
lag=1 entry (MOC the session after the signal close), fractions in / percent out."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
BAR = pd.Timestamp("2026-09-17")


def panel(tickers: list[str], cal_ticker: str, ffill: tuple[str, ...] = ()) -> pd.DataFrame:
    raw = close_panel(tickers)
    cal = raw[cal_ticker].dropna().index
    px = raw.reindex(cal).copy()
    for t in ffill:
        px[t] = px[t].ffill(limit=3)
    return px.loc[:BAR]


def dret(s: pd.Series) -> pd.Series:
    return s / s.shift(1) - 1.0


def cellstats(px, mask, legs, h, label, gap=5, lag=1, cost_bps=None):
    """Episode-level stats with record scored vs coin AND vs the vehicle's own
    unconditional up-rate at the same h, plus edge vs own all-days mean."""
    r = vehicle_ret(px, legs, h, lag)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    epi = declusters(days, gap, px.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    p0 = float((base > 0).mean())
    out = summarize(vals, label)
    out["n_days"] = len(days)
    out["ctl_pct"] = 100 * base.mean()
    out["edge_pp"] = out["mean_pct"] - 100 * base.mean()
    out["rec"] = f"{w}-{len(vals)-w}"
    out["p_coin"] = round(sign_test(w, len(vals)), 4)
    out["p_base"] = round(sign_test(w, len(vals), p0), 4)
    if cost_bps:
        out["x_cost"] = round(100 * out["mean_pct"] / (cost_bps * len(legs)), 2)
    return out, epi, vals


def signed_conc(dates, vals, label=""):
    v = np.asarray(vals, float)
    if len(v) < 3:
        return f"{label}: n<3"
    tot = v.sum()
    order = np.argsort(-v)
    top2 = v[order[:2]].sum()
    d2 = np.delete(v, order[:2])
    yrs = pd.DatetimeIndex(dates).year
    by = pd.Series(v, index=yrs).groupby(level=0).sum().sort_values(ascending=False)
    ysh = by.iloc[0] / tot * 100 if tot != 0 else np.nan
    return (f"{label}: total {100*tot:+.2f}pp N={len(v)}; signed top-2 "
            f"{[str(pd.Timestamp(dates[i]).date()) for i in order[:2]]} = {100*top2:+.2f}pp "
            f"({100*top2/tot if tot else np.nan:.0f}% of total); drop-best-2 mean "
            f"{100*d2.mean():+.3f}% (rec {int((d2>0).sum())}-{int((d2<=0).sum())}); best year "
            f"{by.index[0]} {100*by.iloc[0]:+.2f}pp ({ysh:.0f}% of total); median {100*np.median(v):+.3f}%")


def welch(ep, ctrl):
    ep = np.asarray(ep, float); ep = ep[~np.isnan(ep)]
    ctrl = np.asarray(ctrl, float); ctrl = ctrl[~np.isnan(ctrl)]
    if len(ep) < 2:
        return np.nan
    se = np.sqrt(ep.var(ddof=1) / len(ep) + ctrl.var(ddof=1) / len(ctrl))
    return (ep.mean() - ctrl.mean()) / se


def local_ctl_row(px, mask, legs, h, lag=1, gap=5):
    r = vehicle_ret(px, legs, h, lag)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    epi = declusters(days, gap, px.index)
    loc = local_control(px.index[v.values], days)
    ep = r.loc[epi].values
    lc = r.loc[loc].values
    return {"h": h, "lag": lag, "n_ep": len(ep), "cond_pct": 100 * np.nanmean(ep),
            "local_pct": 100 * np.nanmean(lc), "welch_t_local": welch(ep, lc),
            "own_all_pct": 100 * r[v].mean(), "welch_t_all": welch(ep, r[v].values)}


def midterm_mask(idx):
    return pd.Series(idx.year % 4 == 2, index=idx)


def pit_beta(y: pd.Series, x: pd.Series, win: int = 252) -> pd.Series:
    """Trailing-window OLS beta of y on x, measured through D-1 (shifted)."""
    cov = y.rolling(win, min_periods=win // 2).cov(x)
    var = x.rolling(win, min_periods=win // 2).var()
    return (cov / var).shift(1)
