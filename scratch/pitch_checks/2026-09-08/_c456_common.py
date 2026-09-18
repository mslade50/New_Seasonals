"""Thin composition layer for the C4/C5/C6 ^SKEW-21d-rank checkers.

Nothing statistical is invented here. Masks, alignment and the dial join only;
every number comes out of pitch_lab (summarize / declusters / local_control /
sign_test / bootstrap_p_le0 / cluster_note / horizon_scan).

The one definitional fact this file pins, because the whole morning turns on
it: `pct_rank(^SKEW, n)` is the trailing-252 percentile of the n-day RETURN.
The LEVEL percentile is a different object and is built separately below.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402,F401
    ROOT, load_prices, close_panel, fwd_lag, fwd_ret, vehicle_ret, declusters,
    local_control, summarize, show, era_split, bootstrap_p_le0, sign_test,
    cluster_note, pct_rank, zscore, rolling_on_valid, horizon_scan, battery,
)

FRAG_PATH = ROOT / "data" / "rd2_fragility.parquet"


def align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Foreign-calendar series onto the vehicle calendar, no lookahead."""
    if s.dtype == bool:
        s = s.astype(float)
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def roll_max(s: pd.Series, n: int = 252) -> pd.Series:
    return rolling_on_valid(s, lambda x: x.rolling(n).max())


def level_pct_trailing(s: pd.Series, lookback: int = 252) -> pd.Series:
    """Percentile of the LEVEL against its own trailing window."""
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100)


def level_pct_expanding(s: pd.Series, minp: int = 252) -> pd.Series:
    """Percentile of the LEVEL against full history to date."""
    return rolling_on_valid(s, lambda x: x.expanding(minp).rank(pct=True) * 100)


def dial_series(idx: pd.DatetimeIndex) -> pd.Series:
    """The live sizing statistic: 10d MA of the 63d fragility column, PIT
    parquet, aligned onto the vehicle calendar. 2016+ only, by construction."""
    f = pd.read_parquet(FRAG_PATH)
    ma = f["63d"].rolling(10).mean()
    return align(ma, idx)


def cell(ret: pd.Series, trig: pd.DatetimeIndex, h: int, label: str,
         min_gap: int | None = None, show_dates: int = 0) -> dict | None:
    """One conditional cell against pitch_lab's three controls, EXCESS first."""
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    if len(t) == 0:
        print(f"\n--- {label} (h={h}): NO TRIGGERS.")
        return None
    epi = declusters(t, min_gap or h, valid)
    ep = ret.loc[epi].values
    loc = local_control(valid, t)
    span = valid[(valid >= t[0]) & (valid <= t[-1])]
    rows = [
        summarize(ret.loc[t].values, f"COND day-level (N={len(t)})"),
        summarize(ep, f"COND episodes (N={len(epi)})"),
        summarize(ret.loc[span].values, "CTRL-a own drift, same span"),
        summarize(ret.loc[valid].values, "CTRL-b all days, full history"),
        summarize(ret.loc[loc].values, "CTRL-c local +/-126td ex-trigger"),
    ]
    show(rows, f"{label}  (h={h}td, lag=1)")
    wins = int((ep > 0).sum())
    base_all = float(ret.loc[valid].mean())
    base_loc = float(ret.loc[loc].mean()) if len(loc) else np.nan
    base_span = float(ret.loc[span].mean())
    print(f"  record {wins}-{len(ep) - wins}   sign p={sign_test(wins, len(ep)):.4f}"
          f"   bootstrap P(mean<=0)={bootstrap_p_le0(ep):.3f}")
    print(f"  EXCESS vs CTRL-b all days = {100 * (ep.mean() - base_all):+.3f}pp"
          f" | vs CTRL-c local = {100 * (ep.mean() - base_loc):+.3f}pp"
          f" | vs CTRL-a span = {100 * (ep.mean() - base_span):+.3f}pp")
    print(f"  {cluster_note(epi, ep)}")
    if show_dates:
        print("  episodes: " + ", ".join(str(d.date()) for d in epi[:show_dates])
              + (" ..." if len(epi) > show_dates else ""))
    return {"label": label, "h": h, "epi": epi, "ep": ep, "n": len(epi),
            "n_days": len(t), "mean_pct": 100 * ep.mean(),
            "excess_all_pct": 100 * (ep.mean() - base_all),
            "excess_loc_pct": 100 * (ep.mean() - base_loc),
            "wins": wins, "sign_p": sign_test(wins, len(ep))}


def row(ret: pd.Series, trig, h: int, label: str, min_gap: int | None = None,
        extra: dict | None = None) -> dict:
    """One compact grid row: episodes, raw mean, EXCESS, record, sign p."""
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    out = {"label": label, "n_days": len(t)}
    if extra:
        out.update(extra)
    if len(t) == 0:
        out.update({"n": 0})
        return out
    epi = declusters(t, min_gap or h, valid)
    ep = ret.loc[epi].values
    base = float(ret.loc[valid].mean())
    w = int((ep > 0).sum())
    out.update({
        "n": len(epi),
        "mean_pct": round(100 * ep.mean(), 3),
        "excess_pct": round(100 * (ep.mean() - base), 3),
        "med_pct": round(100 * float(np.median(ep)), 3),
        "hit": round(100 * float((ep > 0).mean()), 1),
        "worst_pct": round(100 * ep.min(), 2),
        "rec": f"{w}-{len(epi) - w}",
        "sign_p": round(sign_test(w, len(epi)), 4),
    })
    return out


def jaccard(a: pd.Series, b: pd.Series) -> tuple[float, int, int, int]:
    av, bv = a.values.astype(bool), b.values.astype(bool)
    inter = int((av & bv).sum())
    union = int((av | bv).sum())
    return (inter / union if union else np.nan), inter, int(av.sum()), int(bv.sum())
