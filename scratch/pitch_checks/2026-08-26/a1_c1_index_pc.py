"""a1 / C1: index put/call 10d-MA at a trailing-252d percentile <= 10 while
the EQUITY put/call sits mid-range (25-75). Vehicle SPY, direction from data.

Round 1. Order of operations:
  0. horizon scan LONG SPY at h=1..10, gated and bare, so the sign and the
     horizon come from the table rather than from the mechanism story.
  1. full battery at the best horizon.
  2. GATE ATTRIBUTION up front, because the equity-mid-range clause discards
     184 of 289 index<=10 days (a0) and that is exactly the shape the registry
     killed on 2026-08-25 ("a conditioning clause can be ANTI-selective").
  3. threshold ladder on the index percentile cut.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (battery, close_panel, declusters, horizon_scan,  # noqa
                       show, summarize, vehicle_ret)

PC = Path(__file__).resolve().parents[3] / "data" / "cboe_putcall.parquet"


def pit_pctile(s, ma=10, window=252):
    m = s.dropna().rolling(ma, min_periods=ma).mean()
    return m.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def main() -> None:
    pc = pd.read_parquet(PC)
    idxp = pit_pctile(pc["index"])
    eqp = pit_pctile(pc["equity"])

    px = close_panel(["SPY"])
    # restrict the price index to the era the signal can exist in
    px = px.loc[idxp.dropna().index.min():]
    idxp = idxp.reindex(px.index).ffill(limit=3)
    eqp = eqp.reindex(px.index).ffill(limit=3)

    m_gated = ((idxp <= 10) & (eqp >= 25) & (eqp <= 75)).fillna(False)
    m_bare = (idxp <= 10).fillna(False)
    m_comp = ((idxp <= 10) & ~((eqp >= 25) & (eqp <= 75))).fillna(False)

    print(f"trigger days: gated={int(m_gated.sum())} bare={int(m_bare.sum())} "
          f"complement={int(m_comp.sum())}   price span "
          f"{px.index.min().date()}..{px.index.max().date()}")

    print("\n### 0. HORIZON SCAN, long SPY, lag=1 (sign comes from here) ###")
    for lbl, m in [("GATED", m_gated), ("BARE", m_bare), ("COMPLEMENT", m_comp)]:
        d = px.index[m.values]
        show(horizon_scan(px, d, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10)),
             f"{lbl} long SPY  (n_days={len(d)})")

    # all-days baseline for reference
    rows = []
    for h in (1, 2, 3, 5, 7, 10):
        r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
        rows.append(summarize(r.dropna().values, f"all days h={h}"))
    show(rows, "0b. unconditional SPY over the SAME era (2020-10+)")

    print("\n### 2. GATE ATTRIBUTION (the registry's anti-selective test) ###")
    for h in (3, 5, 10):
        r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
        out = []
        for lbl, m in [("bare index<=10", m_bare), ("gated (eq 25-75)", m_gated),
                       ("complement (eq outside)", m_comp)]:
            d = px.index[m.values & r.notna().values]
            e = declusters(d, max(h, 10), px.index)
            out.append(summarize(r.loc[e].values, f"h={h} {lbl}"))
        show(out, f"gate attribution h={h}")

    print("\n### 3. index-percentile threshold ladder (episodes, h=5/10) ###")
    for h in (5, 10):
        r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
        out = []
        for cut in (2, 5, 10, 15, 20, 30, 50):
            m = (idxp <= cut).fillna(False)
            d = px.index[m.values & r.notna().values]
            e = declusters(d, max(h, 10), px.index)
            s = summarize(r.loc[e].values, f"h={h} index<={cut}")
            s["n_days"] = len(d)
            out.append(s)
        show(out, f"ladder h={h}")

    print("\n### 1. FULL BATTERY, gated mask, both plausible horizons ###")
    variants = {
        "idx<=5 & eq25-75": ((idxp <= 5) & (eqp >= 25) & (eqp <= 75)).fillna(False),
        "idx<=15 & eq25-75": ((idxp <= 15) & (eqp >= 25) & (eqp <= 75)).fillna(False),
        "idx<=10 & eq20-80": ((idxp <= 10) & (eqp >= 20) & (eqp <= 80)).fillna(False),
        "idx<=10 & eq30-70": ((idxp <= 10) & (eqp >= 30) & (eqp <= 70)).fillna(False),
        "idx<=10 bare": m_bare,
    }
    for h in (5, 10):
        battery(px, m_gated, [("SPY", 1.0)], h,
                f"C1 index P/C <=10 pctile + equity mid-range, LONG SPY",
                cost_bps=6.0, variants=variants, min_gap=10)


if __name__ == "__main__":
    main()
