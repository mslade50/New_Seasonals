"""Shared setup for the kA_ checks (c2/c3/c4), 2026-09-18.

Builds one SPY-calendar panel with SPY, ^VIX, ^VIX3M, SVXY, UVXY, IWM and a
spliced -0.5x short-vol series 'SVS': real SVXY from 2018-02-28 (the -0.5x
era), and 0.5 x the real pre-break SVXY daily return before it (pre-break
SVXY was -1x, so half its daily return is a daily-rebalanced -0.5x). SVS is
for ERA CONTEXT only; the tradeable numbers use real post-break SVXY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2018-02-28")
POST = pd.Timestamp("2018-03-01")


def build_panel() -> pd.DataFrame:
    raw = close_panel(["SPY", "^VIX", "^VIX3M", "SVXY", "UVXY", "IWM"])
    cal = raw["SPY"].dropna().index
    px = raw.reindex(cal).copy()
    px["^VIX"] = px["^VIX"].ffill(limit=2)
    px["^VIX3M"] = px["^VIX3M"].ffill(limit=2)
    sv = px["SVXY"]
    r = sv / sv.shift(1) - 1
    syn = pd.Series(np.where(r.index < BREAK, 0.5 * r, r), index=r.index)
    first = sv.first_valid_index()
    syn = syn.loc[first:].fillna(0.0)
    svs = (1 + syn).cumprod()
    svs.iloc[0] = 1.0
    px["SVS"] = svs.reindex(cal)
    return px


def next_event_distance(cal: pd.DatetimeIndex, kinds: list[str]) -> pd.Series:
    """Sessions from each date to the NEXT event of the given kinds, strictly
    after the date (1 = next session). 999 when none."""
    ev = pd.DatetimeIndex(load_events(kinds)["date"])
    pos, _ = anchor_positions(cal, ev, 0)
    fps = np.sort(np.unique(np.array(pos)))
    out = np.full(len(cal), 999)
    j = 0
    for i in range(len(cal)):
        while j < len(fps) and fps[j] <= i:
            j += 1
        if j < len(fps):
            out[i] = fps[j] - i
    return pd.Series(out, index=cal)


def prev_event_distance(cal: pd.DatetimeIndex, kinds: list[str]) -> pd.Series:
    """Sessions SINCE the most recent event on/before the date (0 = event day)."""
    ev = pd.DatetimeIndex(load_events(kinds)["date"])
    pos, _ = anchor_positions(cal, ev, 0)
    fps = np.sort(np.unique(np.array(pos)))
    out = np.full(len(cal), 999)
    j = -1
    for i in range(len(cal)):
        while j + 1 < len(fps) and fps[j + 1] <= i:
            j += 1
        if j >= 0:
            out[i] = i - fps[j]
    return pd.Series(out, index=cal)


def hedge_beta(px: pd.DataFrame, veh: str, h: int, lag: int, era_mask: pd.Series) -> float:
    rs = vehicle_ret(px, [(veh, 1.0)], h, lag)
    rspy = vehicle_ret(px, [("SPY", 1.0)], h, lag)
    ok = rs.notna() & rspy.notna() & era_mask
    return float(np.polyfit(rspy[ok].values, rs[ok].values, 1)[0])


def rec_row(v: np.ndarray, label: str, cost_bps: float | None = None) -> dict:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if len(v):
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v) - w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
        if cost_bps:
            s["x_cost"] = round(100 * 100 * v.mean() / cost_bps, 2)
    return s


def signed_concentration(dates, vals, k: int = 2) -> str:
    v = pd.Series(np.asarray(vals, float), index=pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return "n/a"
    tot = v.sum()
    top = v.sort_values(ascending=False).head(k)
    rest = v.drop(top.index)
    by_yr = v.groupby(v.index.year).sum().sort_values(ascending=False)
    yshare = by_yr.iloc[0] / tot * 100 if tot != 0 else np.nan
    return (f"signed top{k} {[str(d.date()) for d in top.index]} = "
            f"{100*top.sum():+.2f}pp of {100*tot:+.2f}pp total "
            f"({(top.sum()/tot*100 if tot else np.nan):.0f}%); drop-best-{k} mean "
            f"{100*rest.mean():+.3f}% on N={len(rest)}; best year {by_yr.index[0]} "
            f"{100*by_yr.iloc[0]:+.2f}pp ({yshare:.0f}% of total)")
