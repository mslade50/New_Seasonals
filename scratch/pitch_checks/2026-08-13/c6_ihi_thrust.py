"""C6 - IHI 21d rank >= 99 while still >= 10% below its 52w high.

Tests BOTH directions (fade and continuation) against:
 - IHI's OWN unconditional drift over the same span (the headline control)
 - XLV and SPY residuals at the measured beta
and answers the anti-rip-off question: is this the book's Overbot Fade family
with a 1x ticker, or the 52wh Breakout / Sector BO continuation family?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["IHI", "XLV", "SPY"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
ihi, xlv, spy = raw["IHI"]["Close"], raw["XLV"]["Close"], raw["SPY"]["Close"]

r21 = pct_rank(ihi, 21)
dd = ihi / ihi.rolling(252).max() - 1.0


def mask(r21_min=99, dd_max=-0.10):
    return ((r21 >= r21_min) & (dd <= dd_max)).reindex(px.index).fillna(False)


print("=== occurrence counts ===")
rows = []
for rq in (95, 97, 99, 100):
    for dm in (-0.05, -0.08, -0.10, -0.15, 0.0):
        m = mask(rq, dm)
        d = px.index[m.values]
        rows.append({"r21>=": rq, "dd<=": dm, "n_days": len(d),
                     "n_epi": len(declusters(d, 5, px.index)),
                     "first": str(d[0].date()) if len(d) else "-",
                     "last": str(d[-1].date()) if len(d) else "-"})
print(pd.DataFrame(rows).to_string(index=False))

# betas
al = pd.concat([ihi.pct_change(), xlv.pct_change(), spy.pct_change()], axis=1).dropna()
al.columns = ["ihi", "xlv", "spy"]
b_xlv = np.polyfit(al["xlv"], al["ihi"], 1)[0]
b_spy = np.polyfit(al["spy"], al["ihi"], 1)[0]
print(f"\nbeta IHI~XLV = {b_xlv:.3f} (corr {al['ihi'].corr(al['xlv']):.3f})   "
      f"beta IHI~SPY = {b_spy:.3f} (corr {al['ihi'].corr(al['spy']):.3f})")

m = mask()
trig = px.index[m.values]
print(f"\nAS SPECIFIED n_days={len(trig)} epi={len(declusters(trig, 5, px.index))}")
print("dates:", ", ".join(str(d.date()) for d in trig))

# ------------------------------------------------------------------ book overlap
print("\n\n=== BOOK OVERLAP: is this the Overbot Fade family with a 1x ticker? ===")
r2, r5, r10 = pct_rank(ihi, 2), pct_rank(ihi, 5), pct_rank(ihi, 10)
r126, r252 = pct_rank(ihi, 126), pct_rank(ihi, 252)
print(f"TODAY IHI: r2={r2.iloc[-1]:.1f} r5={r5.iloc[-1]:.1f} r10={r10.iloc[-1]:.1f} "
      f"r21={r21.iloc[-1]:.1f} r126={r126.iloc[-1]:.1f} r252={r252.iloc[-1]:.1f}")
fade_shape = ((r2 > 85) & (r5 > 85) & (r10 > 85) & (r21 > 85)
              & (r126 < 65) & (r252 < 65)).reindex(px.index).fillna(False)
print(f"3x-Overbot-Fade FILTER SHAPE on IHI: {int(fade_shape.sum())} days ever; "
      f"live today = {bool(fade_shape.iloc[-1])}")
ov = m & fade_shape
print(f"C6 trigger days that ALSO satisfy the fade shape: {int(ov.sum())} of {int(m.sum())} "
      f"({100*ov.sum()/max(1, m.sum()):.0f}%)")
print("  -> the 126/252 < 65 leg is what separates them; IHI r126/r252 on C6 days:")
print("    ", pd.DataFrame({"r126": r126.loc[trig].round(1),
                            "r252": r252.loc[trig].round(1)}).describe().loc[
    ["min", "50%", "max"]].round(1).to_string())
# 52wh Breakout / Sector BO need a NEW 52w HIGH; C6 requires >=10% below it
print(f"52wh Breakout / Sector BO both require a NEW 52w HIGH. C6 requires >=10% "
      f"below it -> mutually exclusive by construction. IHI dd today = "
      f"{100*dd.iloc[-1]:.2f}%")

# ------------------------------------------------------------------ batteries
variants = {"r21>=97": mask(97, -0.10), "r21>=100": mask(100, -0.10),
            "dd<=-5%": mask(99, -0.05), "dd<=-15%": mask(99, -0.15),
            "no dd gate": mask(99, 0.0)}
for h in (3, 5, 10):
    battery(px, m, [("IHI", -1.0)], h, "C6 SHORT IHI outright (FADE)", 8.0, variants)
    battery(px, m, [("IHI", 1.0)], h, "C6 LONG IHI outright (CONTINUATION)", 8.0, variants)
    battery(px, m, [("IHI", -1.0), ("XLV", b_xlv)], h,
            f"C6 SHORT IHI vs LONG XLV, beta-neutral (beta {b_xlv:.2f})", 10.0)
    battery(px, m, [("IHI", -1.0), ("SPY", b_spy)], h,
            f"C6 SHORT IHI vs LONG SPY, beta-neutral (beta {b_spy:.2f})", 9.0)

# ------------------------------------------------------------------ cuts
print("\n\n=== midterm / SPY-near-high / cluster depth, h=5 ===")
epi = declusters(trig, 5, px.index)
for legs, lbl in [([("IHI", -1.0)], "short IHI"), ([("IHI", 1.0)], "long IHI"),
                  ([("IHI", -1.0), ("XLV", b_xlv)], "short IHI vs XLV")]:
    r = vehicle_ret(px, legs, 5)
    mt = np.array([d.year % 4 == 2 for d in epi])
    spyhi = ((spy / spy.rolling(252).max() - 1) > -0.01).reindex(px.index).fillna(False)
    g = spyhi.loc[epi].values
    show([summarize(r.loc[epi[mt]].values, f"{lbl} midterm (N={int(mt.sum())})"),
          summarize(r.loc[epi[~mt]].values, f"{lbl} non-midterm"),
          summarize(r.loc[epi[g]].values, f"{lbl} SPY within 1% of high (N={int(g.sum())})"),
          summarize(r.loc[epi[~g]].values, f"{lbl} SPY not near high")])

print("\n=== cluster depth (today = 3rd consecutive trigger session) ===")
mv = m.values
depth = np.zeros(len(mv), int)
for i in range(len(mv)):
    depth[i] = depth[i - 1] + 1 if mv[i] and i > 0 else (1 if mv[i] else 0)
dser = pd.Series(depth, index=px.index)
for h in (3, 5, 10):
    r = vehicle_ret(px, [("IHI", -1.0)], h)
    rows = []
    for lo, hi in [(1, 1), (2, 3), (4, 99)]:
        sel = trig[(dser.loc[trig] >= lo) & (dser.loc[trig] <= hi)]
        rows.append(summarize(r.loc[sel].values,
                              f"h={h} short, depth {lo}-{hi} (N={len(sel)})"))
    show(rows)
print(f"  today's depth = {int(dser.iloc[-1])}; trigger-population median depth = "
      f"{float(dser.loc[trig].median()):.1f}")

print("\n=== year concentration, short h=5 episodes ===")
r = vehicle_ret(px, [("IHI", -1.0)], 5)
print(" ", (pd.Series(r.loc[epi].values, index=epi.year).groupby(level=0)
            .agg(["sum", "count"]).round(4).to_string()))
