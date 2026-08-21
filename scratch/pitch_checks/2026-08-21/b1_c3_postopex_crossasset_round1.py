"""C3/C4/C5 round 1 — the post-opex window on NON-EQUITY classes.

ENTRY CONVENTION, stated once and used everywhere in this file:
  Today 2026-08-21 IS monthly opex. The tradeable order this morning is
  "enter MOC on the OPEX CLOSE". In the pitch's lag=1 grammar that means the
  SIGNAL date is the session BEFORE opex (2026-08-20, the freshest bar) and
  entry lands on the opex close.  So:
      anchor_A (LIVE FORM) = index position of opex - 1, lag=1
                           -> entry MOC on the opex close, exit +h sessions
      anchor_B (contrast)  = opex date itself,           lag=1
                           -> entry MOC on the session AFTER opex
  Form B is the event sleeve's V4 window shifted a day; form A is what can
  actually be pitched today. Both are reported.

The US-equity opex anchor is CLOSED in this repo (registry 2026-08-07 into,
2026-08-20 out of, on SPY and IWM). This file only walks non-equity /
non-US-equity classes: GLD SLV TLT IEF HYG LQD USO XLE UUP FXI.
SPY is carried as a reference column ONLY, never as a candidate.

Grid walked: 10 vehicles x 10 horizons = 100 cells. Priced as a grid at the
bottom, per the multiplicity rule.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "USO", "XLE", "UUP", "FXI"]
REF = "SPY"
HS = list(range(1, 11))
# round-trip cost in bps (2 legs of spread+slip), deliberately generous
COST = {"GLD": 2, "SLV": 4, "TLT": 2, "IEF": 3, "HYG": 4, "LQD": 4,
        "USO": 5, "XLE": 2, "UUP": 6, "FXI": 4, "SPY": 2}

px = close_panel(VEH + [REF])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

opex = load_events(["opex"])["date"]
opex = pd.DatetimeIndex([d for d in opex if d in pos.index])
print(f"opex anchors present in the price calendar: {len(opex)} "
      f"({opex[0].date()} .. {opex[-1].date()})")

# anchor A: the session BEFORE opex (entry MOC on the opex close at lag=1)
aA = pd.DatetimeIndex([idx[pos[d] - 1] for d in opex if pos[d] >= 1])
# anchor B: opex itself (entry MOC on the session after opex)
aB = opex


def cell(v, anchors, h, lag=1):
    """(mean_pct, excess_pct, n, wins, worst_pct, dates, vals) for one cell."""
    s = px[v].dropna()
    r = fwd_lag(s, h, lag)
    a = pd.DatetimeIndex(anchors).intersection(r.dropna().index)
    if len(a) == 0:
        return None
    span = (a[0], a[-1])
    base = r.dropna()
    base = base[(base.index >= span[0]) & (base.index <= span[1])]
    vals = r.loc[a].values
    return dict(v=v, h=h, n=len(a), mean_pct=100 * vals.mean(),
                med_pct=100 * float(np.median(vals)),
                ctrl_pct=100 * base.mean(),
                excess_pct=100 * (vals.mean() - base.mean()),
                wins=int((vals > 0).sum()),
                worst_pct=100 * vals.min(),
                dates=a, vals=vals)


for lbl, anch in [("A  entry MOC on the OPEX CLOSE (live form)", aA),
                  ("B  entry MOC on opex+1 (V4-style window)", aB)]:
    print(f"\n\n########## FORM {lbl} ##########")
    grid = []
    for v in VEH + [REF]:
        for h in HS:
            c = cell(v, anch, h)
            if c:
                grid.append(c)
    g = pd.DataFrame([{k: c[k] for k in
                       ("v", "h", "n", "mean_pct", "med_pct", "ctrl_pct",
                        "excess_pct", "wins", "worst_pct")} for c in grid])
    piv = g[g.v != REF].pivot(index="v", columns="h", values="excess_pct")
    print("\nEXCESS over own same-span all-days drift, pct (rows=vehicle, cols=h):")
    print(piv.round(3).to_string())
    pivm = g[g.v != REF].pivot(index="v", columns="h", values="mean_pct")
    print("\nRAW conditional mean, pct:")
    print(pivm.round(3).to_string())
    print("\nSPY reference row (excess):")
    print(g[g.v == REF].set_index("h")["excess_pct"].round(3).to_string())

    cand = g[g.v != REF].copy()
    sd = cand["excess_pct"].std(ddof=1)
    print(f"\nGRID PRICE: {len(cand)} cells walked, excess sd = {sd:.3f}pp, "
          f"{int((cand.excess_pct.abs() > 1.0).sum())} cells clear |1.0pp|")
    top = cand.reindex(cand.excess_pct.abs().sort_values(ascending=False).index).head(8)
    print("top 8 cells by |excess|:")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for _, row in top.head(5).iterrows():
        c = [x for x in grid if x["v"] == row.v and x["h"] == row.h][0]
        sp = sign_test(c["wins"], c["n"]) if c["mean_pct"] > 0 else \
            sign_test(c["n"] - c["wins"], c["n"])
        edge_bps = abs(c["excess_pct"]) * 100
        print(f"  {c['v']} h={c['h']}: {c['wins']}-{c['n']-c['wins']}, "
              f"sign p={sp:.4f}, median {c['med_pct']:+.3f}%, "
              f"worst {c['worst_pct']:.2f}%, edge {edge_bps:.1f}bps = "
              f"{edge_bps/COST[c['v']]:.1f}x a {COST[c['v']]}bp round trip "
              f"(need >=5x)")
        z = row.excess_pct / sd
        # Sidak over the 100-cell grid, two-sided per-cell normal approx
        from math import erfc
        p1 = erfc(abs(z) / np.sqrt(2))
        print(f"      grid z = {z:+.2f}, per-cell p = {p1:.3f}, "
              f"Sidak over {len(cand)} cells = {1-(1-p1)**len(cand):.3f}")

# ---------------------------------------------------------------- August only
print("\n\n########## AUGUST-ONLY subgrid, FORM A (live form) ##########")
augA = pd.DatetimeIndex([d for d in aA if idx[pos[d] + 1].month == 8])
print(f"August anchors: {len(augA)}")
rows = []
for v in VEH:
    for h in (3, 5, 10):
        c = cell(v, augA, h)
        if c:
            rows.append({"v": v, "h": h, "n": c["n"], "mean": c["mean_pct"],
                         "med": c["med_pct"], "ctrl_alldays": c["ctrl_pct"],
                         "excess": c["excess_pct"],
                         "rec": f"{c['wins']}-{c['n']-c['wins']}",
                         "worst": c["worst_pct"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# --------------------------------------------------------------- midterm split
print("\n\n########## MIDTERM (year%4==2) split, FORM A, all months ##########")
rows = []
for v in VEH:
    for h in (3, 5, 10):
        c = cell(v, aA, h)
        if not c:
            continue
        d = pd.DatetimeIndex(c["dates"])
        m = (d.year % 4 == 2)
        rows.append({"v": v, "h": h,
                     "midterm_n": int(m.sum()),
                     "midterm": 100 * c["vals"][m].mean(),
                     "nonmid": 100 * c["vals"][~m].mean(),
                     "all": c["mean_pct"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
