"""Shared state probe for C4/C5/C6/C7: today's readings, history depth,
cluster depth of the live thrust, and vehicle drag on every instrument I am
about to credit an edge to.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["GDX", "GLD", "SLV", "XLE", "XOP", "USO", "EOG", "CVX", "VLO", "XME", "NEM", "SPY"]
px = load_prices(TK)
for t in TK:
    if t in px:
        s = px[t]["Close"]
        print(f"{t:5s} {s.index.min().date()} .. {s.index.max().date()}  n={len(s)}")

P = close_panel(TK)
ASOF = P.index[-1]
print(f"\nASOF = {ASOF.date()}")

rows = []
for t in TK:
    if t not in P.columns:
        continue
    s = P[t].dropna()
    hi52 = s.rolling(252).max()
    sma200 = s.rolling(200).mean()
    rows.append({
        "tkr": t,
        "ret5_pct": round(100 * s.pct_change(5).loc[ASOF], 2),
        "ret21_pct": round(100 * s.pct_change(21).loc[ASOF], 2),
        "ret63_pct": round(100 * s.pct_change(63).loc[ASOF], 2),
        "rank5": round(pct_rank(s, 5).loc[ASOF], 1),
        "rank63": round(pct_rank(s, 63).loc[ASOF], 1),
        "z10": round(zscore(s).loc[ASOF], 2),
        "d52wh_pct": round(100 * (s.loc[ASOF] / hi52.loc[ASOF] - 1), 2),
        "d200_pct": round(100 * (s.loc[ASOF] / sma200.loc[ASOF] - 1), 2),
        # vehicle drag: unconditional 10td drift, full history
        "drift10_pct": round(100 * fwd_lag(s, 10, 1).mean(), 3),
        "drift5_pct": round(100 * fwd_lag(s, 5, 1).mean(), 3),
        "drift2_pct": round(100 * fwd_lag(s, 2, 1).mean(), 3),
    })
show(rows, "today's state + unconditional drift (the control that is never zero)")

# ---- cluster depth of the live thrust ----
print("\n=== live cluster depth: how many consecutive sessions has the trigger been on? ===")


def depth(mask: pd.Series, asof) -> int:
    m = mask.fillna(False)
    idx = list(m.index)
    p = idx.index(asof)
    d = 0
    while p - d >= 0 and bool(m.iloc[p - d]):
        d += 1
    return d


def depth_hist(mask: pd.Series):
    """Distribution of run-lengths-so-far across all trigger days."""
    m = mask.fillna(False).values
    run, out = 0, []
    for v in m:
        run = run + 1 if v else 0
        if v:
            out.append(run)
    return np.array(out)


g, gl, sl = P["GDX"].dropna(), P["GLD"].dropna(), P["SLV"].dropna()
sp5 = (g.pct_change(5) - gl.reindex(g.index).pct_change(5)) * 100
m_c6 = sp5 >= 8.0
print(f"C6 (GDX-GLD 5d spread >= 8pp): today {sp5.loc[ASOF]:+.2f}pp, depth={depth(m_c6, ASOF)}; "
      f"hist run-length dist p50={np.median(depth_hist(m_c6)):.0f} p90={np.percentile(depth_hist(m_c6), 90):.0f} "
      f"max={depth_hist(m_c6).max()}")

r5s = sl.pct_change(5) * 100
hi52s = sl.rolling(252).max()
dds = 100 * (sl / hi52s - 1)
m_c7 = (r5s >= 8.0) & (dds <= -25.0)
print(f"C7 (SLV 5d >= 8% & >=25% below 52wh): today r5={r5s.loc[ASOF]:+.2f}%, dd={dds.loc[ASOF]:+.2f}%, "
      f"depth={depth(m_c7, ASOF)}; hist run-length p50={np.median(depth_hist(m_c7)):.0f} "
      f"p90={np.percentile(depth_hist(m_c7), 90):.0f} max={depth_hist(m_c7).max()}")

rk5g = pct_rank(g, 5)
m_c4 = rk5g >= 95
print(f"C4 (GDX rank5 >= 95): today {rk5g.loc[ASOF]:.1f}, depth={depth(m_c4, ASOF)}; "
      f"hist run-length p50={np.median(depth_hist(m_c4)):.0f} p90={np.percentile(depth_hist(m_c4), 90):.0f}")

xle = P["XLE"].dropna()
rk5x, rk21x = pct_rank(xle, 5), pct_rank(xle, 21)
m_c5 = (rk5x <= 15) & (rk21x >= 55)
print(f"C5 (XLE rank5<=15 & rank21>=55): today rk5={rk5x.loc[ASOF]:.1f} rk21={rk21x.loc[ASOF]:.1f}, "
      f"depth={depth(m_c5, ASOF)}; hist run-length p50={np.median(depth_hist(m_c5)):.0f} "
      f"p90={np.percentile(depth_hist(m_c5), 90):.0f}")

# ---- events sanity ----
ev = load_events(["cpi", "ppi"])
print(f"\nevents: cpi n={(ev.event=='cpi').sum()}, ppi n={(ev.event=='ppi').sum()}, "
      f"range {ev.date.min().date()}..{ev.date.max().date()}")
nxt = ev[ev.date > ASOF].head(4)
print(nxt.to_string(index=False))
