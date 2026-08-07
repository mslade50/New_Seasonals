"""Shared falsification battery for the 2026-08-07 pitch checks. Read-only.

Trade convention modeled (matches the candidates): signal measured on close of D,
entry MOC on D+1, exit MOC on D+1+h. So the tradeable return is
close[D+1+h]/close[D+1]-1, aligned to the signal date D. `lag=0` reproduces the
naive signal-close-to-close version for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import declusters, summarize, show, era_split, bootstrap_p_le0, load_events  # noqa

import numpy as np
import pandas as pd


def vehicle_ret(px: pd.DataFrame, legs: list[tuple[str, float]], h: int, lag: int = 1) -> pd.Series:
    """Equal-dollar leg combination; returns are aligned to the SIGNAL date."""
    out = None
    for tkr, w in legs:
        c = px[tkr]
        r = (c.shift(-(lag + h)) / c.shift(-lag)) - 1.0
        out = w * r if out is None else out + w * r
    return out


def cpi_in_window(px: pd.DataFrame, dates: pd.DatetimeIndex, h: int, lag: int = 1) -> np.ndarray:
    """True when a CPI print lands strictly after the entry close and on/before exit."""
    cpi = set(load_events(["cpi"])["date"])
    idx = px.index
    pos = pd.Series(range(len(idx)), index=idx)
    flags = []
    for d in dates:
        i = pos.get(d)
        a, b = i + lag, i + lag + h
        if b >= len(idx):
            flags.append(False)
            continue
        lo, hi = idx[a], idx[b]
        flags.append(any(lo < c <= hi for c in cpi))
    return np.array(flags, dtype=bool)


def battery(px, mask, legs, h, title, cost_bps, variants=None, lag=1, min_gap=None):
    """Full kill-battery. `variants` = {label: boolean mask} threshold sensitivities."""
    ret = vehicle_ret(px, legs, h, lag)
    naive = vehicle_ret(px, legs, h, lag=0)
    valid = ret.notna()
    sig_all = px.index[mask.reindex(px.index, fill_value=False).values & valid.values]
    if len(sig_all) == 0:
        print(f"\n##### {title}: NO TRIGGERS EVER. Dead. #####")
        return
    span = (sig_all[0], sig_all[-1])
    in_span = (px.index >= span[0]) & (px.index <= span[1]) & valid.values

    print(f"\n##### {title}  (h={h} td, entry lag={lag}) #####")
    print(f"legs={legs}  trigger span {span[0].date()} .. {span[1].date()}")

    # --- 1. pattern + two controls -------------------------------------------
    epi = declusters(sig_all, min_gap or h, px.index)
    rows = [summarize(ret.loc[sig_all].values, f"COND day-level (N={len(sig_all)})"),
            summarize(ret.loc[epi].values, f"COND episodes (N={len(epi)})"),
            summarize(ret[in_span].values, "CTRL-a own drift, same span"),
            summarize(ret[valid].values, "CTRL-b all days, full history"),
            summarize(naive.loc[sig_all].values, "COND day-level, lag=0 (no-lag)")]
    show(rows, "1. conditional vs controls")
    ep = ret.loc[epi].values
    ctrl = ret[in_span].values
    if len(ep) > 1:
        se = np.sqrt(ep.var(ddof=1) / len(ep) + ctrl.var(ddof=1) / len(ctrl))
        print(f"  episode-vs-control diff = {100*(ep.mean()-ctrl.mean()):+.3f}%  "
              f"welch t = {(ep.mean()-ctrl.mean())/se:+.2f}   "
              f"bootstrap P(mean<=0) = {bootstrap_p_le0(ep):.3f}")

    # --- 2/3. era stability on episodes --------------------------------------
    show(era_split(epi, ep), "2+3. episode era split (pre-2018 / 2018+)")
    show(era_split(sig_all, ret.loc[sig_all].values), "  day-level era split (for contrast)")
    print(f"  worst episode window: {100*ep.min():.2f}%  on "
          f"{epi[int(np.argmin(ep))].date()}   |  worst day-level: "
          f"{100*ret.loc[sig_all].min():.2f}%")
    print("  episode dates:", ", ".join(str(d.date()) for d in epi))

    # --- 4. threshold sensitivity --------------------------------------------
    if variants:
        vr = []
        for lbl, m in variants.items():
            s = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
            if len(s) == 0:
                vr.append({"label": lbl, "n": 0})
                continue
            e = declusters(s, min_gap or h, px.index)
            r = summarize(ret.loc[e].values, f"{lbl} (episodes)")
            r["n_days"] = len(s)
            vr.append(r)
        show(vr, "4. threshold sensitivity (episode level)")

    # --- 5. cost sanity -------------------------------------------------------
    n_legs = len(legs)
    rt = cost_bps * n_legs
    print(f"\n5. cost: {n_legs} leg(s) x ~{cost_bps} bps round trip = {rt} bps. "
          f"episode mean {100*ep.mean():.3f}% = {100*ep.mean()*100:.1f} bps -> "
          f"{(100*ep.mean()*100)/rt:.1f}x cost (need >=5x)")

    # --- 6. CPI in window -----------------------------------------------------
    fl = cpi_in_window(px, epi, h, lag)
    show([summarize(ep[fl], f"CPI IN window (N={fl.sum()})"),
          summarize(ep[~fl], f"CPI OUT of window (N={(~fl).sum()})")],
         "6. CPI-in-hold split (episodes)")
    fld = cpi_in_window(px, sig_all, h, lag)
    rd = ret.loc[sig_all].values
    show([summarize(rd[fld], f"CPI IN (day-level, N={fld.sum()})"),
          summarize(rd[~fld], f"CPI OUT (day-level, N={(~fld).sum()})")], "  day-level CPI split")
