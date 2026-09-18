"""Thin composition layer over pitch_lab for the 2026-09-07 stage-B1 price-state survey.

Nothing statistical is invented here. Every number comes from pitch_lab's
summarize / declusters / local_control / sign_test / bootstrap_p_le0 /
cluster_note. This file exists so the six S1..S6 scripts print the SAME
control table instead of six hand-rolled ones.

Conventions inherited unchanged from pitch_lab:
  - returns are FRACTIONS in, PERCENT out of summarize()
  - entry is lag=1 (state prints on close D, order goes in on close D+1)
  - episodes are declustered at min_gap = h trading days (pitch_lab.battery's
    own default), so episode counts shrink as the horizon lengthens
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import (  # noqa: F401
    load_prices, close_panel, fwd_lag, fwd_ret, declusters, local_control,
    summarize, show, era_split, bootstrap_p_le0, sign_test, cluster_note,
    pct_rank, zscore, rolling_on_valid, vehicle_ret, horizon_scan,
)


def align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Put a series computed on its OWN calendar onto the vehicle calendar
    without lookahead (union -> ffill -> reindex). ^VIX/^SKEW/^TNX carry a few
    sessions SPY does not; a bare reindex would blank those rows."""
    if s.dtype == bool:
        s = s.astype(float)   # ffill on an object-dtype bool column is deprecated
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def sma(s: pd.Series, n: int) -> pd.Series:
    return rolling_on_valid(s, lambda x: x.rolling(n).mean())


def roll_max(s: pd.Series, n: int = 252) -> pd.Series:
    return rolling_on_valid(s, lambda x: x.rolling(n).max())


def roll_min(s: pd.Series, n: int = 252) -> pd.Series:
    return rolling_on_valid(s, lambda x: x.rolling(n).min())


def cell(ret: pd.Series, trig: pd.DatetimeIndex, h: int, label: str,
         show_dates: int = 30, min_gap: int | None = None) -> dict | None:
    """One conditional cell against pitch_lab's three controls.

    ret   : forward return series aligned to the SIGNAL date (fractions)
    trig  : candidate trigger dates
    """
    valid = ret.dropna().index
    t = pd.DatetimeIndex(trig).intersection(valid)
    if len(t) == 0:
        print(f"\n--- {label} (h={h}): NO TRIGGERS with a resolvable forward return.")
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
    print(f"  record {wins}-{len(ep) - wins}   sign p={sign_test(wins, len(ep)):.4f}"
          f"   bootstrap P(mean<=0)={bootstrap_p_le0(ep):.3f}")
    print(f"  edge vs CTRL-b all days = {100 * (ep.mean() - base_all):+.3f}pp"
          f"   |  edge vs CTRL-c local = {100 * (ep.mean() - base_loc):+.3f}pp")
    print(f"  {cluster_note(epi, ep)}")
    if show_dates:
        print("  episodes: " + ", ".join(str(d.date()) for d in epi[:show_dates])
              + (" ..." if len(epi) > show_dates else ""))
    return {"label": label, "h": h, "epi": epi, "ep": ep,
            "mean_pct": 100 * ep.mean(), "n": len(epi), "n_days": len(t),
            "edge_all_pct": 100 * (ep.mean() - base_all),
            "edge_loc_pct": 100 * (ep.mean() - base_loc),
            "wins": wins, "sign_p": sign_test(wins, len(ep))}


def hscan(retf, trig: pd.DatetimeIndex, label: str,
          hs=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Episode-level horizon sweep for a callable h -> forward-return series."""
    rows = []
    for h in hs:
        ret = retf(h)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(trig).intersection(valid)
        if len(t) == 0:
            rows.append({"h": h, "n": 0})
            continue
        epi = declusters(t, h, valid)
        ep = ret.loc[epi].values
        loc = local_control(valid, t)
        base = float(ret.loc[valid].mean())
        r = summarize(ep, f"h={h}")
        r["h"] = h
        r["n_days"] = len(t)
        r["ctl_all_pct"] = round(100 * base, 3)
        r["ctl_loc_pct"] = round(100 * float(ret.loc[loc].mean()), 3) if len(loc) else np.nan
        r["edge_all_pct"] = round(r["mean_pct"] - 100 * base, 3)
        w = int((ep > 0).sum())
        r["rec"] = f"{w}-{len(ep) - w}"
        r["sign_p"] = round(sign_test(w, len(ep)), 4)
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = [c for c in ["h", "n", "n_days", "mean_pct", "median_pct", "hit",
                        "t", "worst_pct", "ctl_all_pct", "ctl_loc_pct",
                        "edge_all_pct", "rec", "sign_p"] if c in df.columns]
    print(f"\n=== HORIZON SWEEP: {label} (episodes, decluster gap = h) ===")
    print(df[keep].round(3).to_string(index=False))
    return df


def state_line(name: str, value, claim=None, unit: str = "") -> None:
    tag = ""
    if claim is not None:
        try:
            tag = f"   [claimed {claim}{unit}  delta {float(value) - float(claim):+.2f}]"
        except (TypeError, ValueError):
            tag = f"   [claimed {claim}]"
    print(f"  {name:<46s} {value}{unit}{tag}")
