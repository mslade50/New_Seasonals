"""C7 round 1: long gold on the yields-up / dollar-at-a-floor state.

Same trigger as C6. The kills to try:
  - GLD's own unconditional drift over the same span (CTRL-a) and the LOCAL
    +/-126td control (does the trigger just select gold bull markets?)
  - gate attribution BOTH ways (TNX alone, DX alone, and the joint minus each)
  - selection: does the cell require gold ALREADY strong, and what is the
    magnitude gradient at today's reading?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

T = ["DX-Y.NYB", "^TNX", "GLD", "GDX", "SLV", "SPY", "TLT"]
px = close_panel(T).dropna(subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx, gld = px["DX-Y.NYB"], px["^TNX"], px["GLD"]

rk_tnx, rk_dx = pct_rank(tnx, 21), pct_rank(dx, 21)
r21_tnx = tnx.pct_change(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)
tnx_only = ((r21_tnx > 0) & (rk_tnx >= 65)).fillna(False)
dx_only = (rk_dx <= 20).fillna(False)

print("GLD panel starts", px.index[0].date(), " trigger days", int(base.sum()))

variants = {
    "TNX>=55 / DX<=20": ((r21_tnx > 0) & (rk_tnx >= 55) & (rk_dx <= 20)).fillna(False),
    "TNX>=75 / DX<=20": ((r21_tnx > 0) & (rk_tnx >= 75) & (rk_dx <= 20)).fillna(False),
    "TNX>=65 / DX<=10": ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 10)).fillna(False),
    "TNX>=65 / DX<=30": ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 30)).fillna(False),
    "TNX gate ALONE": tnx_only,
    "DX gate ALONE": dx_only,
    "DX<=20 & TNX FALLING": ((r21_tnx < 0) & (rk_dx <= 20)).fillna(False),
    "TNX>=65 & DX rank>=50": ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx >= 50)).fillna(False),
}

for h in (3, 5):
    battery(px, base, [("GLD", 1.0)], h, f"C7 LONG GLD h={h}", 2.0,
            variants=variants, min_gap=21,
            event_kinds=("fomc_decision", "cpi", "nfp"))

battery(px, base, [("GDX", 1.0)], 5, "C7 LONG GDX h=5", 5.0,
        variants=variants, min_gap=21, event_kinds=("fomc_decision", "cpi", "nfp"))

# ---- gate attribution both ways, episode level, several horizons ----
print("\n\n=== C7 GATE ATTRIBUTION (episodes, min_gap=21) ===")
rows = []
cells = {
    "JOINT (TNX up+rank65, DX<=20)": base,
    "TNX gate alone": tnx_only,
    "TNX gate & DX rank>20 (joint removed)": (tnx_only & ~dx_only),
    "DX gate alone": dx_only,
    "DX gate & NOT TNX gate (joint removed)": (dx_only & ~tnx_only),
    "neither gate": (~tnx_only & ~dx_only),
}
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = ret.dropna().index
    for lbl, m in cells.items():
        t = px.index[m.reindex(px.index, fill_value=False).values].intersection(valid)
        epi = declusters(t, 21, valid)
        r = summarize(ret.loc[epi].values, f"h={h} {lbl}")
        rows.append(r)
show(rows, "gold forward by gate cell")

# ---- selection: is gold already strong on trigger days? ----
print("\n=== C7 SELECTION CHECK: gold's OWN state on trigger days ===")
rk_gld = pct_rank(gld, 21)
r21_gld = gld.pct_change(21)
trig = px.index[base.values]
trig = trig[rk_gld.reindex(trig).notna()]
print("GLD rank21 on trigger days: median %.1f, mean %.1f (all days median %.1f); "
      "today %.1f" % (rk_gld.loc[trig].median(), rk_gld.loc[trig].mean(),
                      rk_gld.median(), rk_gld.iloc[-1]))
print("GLD 21d ret on trigger days: median %+0.2f%%; today %+0.2f%%"
      % (100*r21_gld.loc[trig].median(), 100*r21_gld.iloc[-1]))

# magnitude gradient: forward by GLD's own prior 21d return bucket
ret5 = vehicle_ret(px, [("GLD", 1.0)], 5)
valid = ret5.dropna().index
epi = declusters(pd.DatetimeIndex(trig).intersection(valid), 21, valid)
pri = r21_gld.loc[epi]
rows = []
for lo, hi, lbl in [(-1, 0.0, "GLD 21d < 0"), (0.0, 0.04, "0-4%"),
                    (0.04, 0.08, "4-8%"), (0.08, 1, ">=8% (TODAY +8.42%)")]:
    m = (pri > lo) & (pri <= hi)
    rows.append(summarize(ret5.loc[epi[m.values]].values, lbl))
show(rows, "h=5 forward by gold's PRIOR 21d return (episodes)")

rows = []
for lo, hi, lbl in [(0, 50, "GLD rank21 <50"), (50, 75, "50-75"),
                    (75, 90, "75-90 (TODAY 77.0)"), (90, 101, ">=90")]:
    m = (rk_gld.loc[epi] > lo) & (rk_gld.loc[epi] <= hi)
    rows.append(summarize(ret5.loc[epi[m.values]].values, lbl))
show(rows, "h=5 forward by gold's PRIOR rank21 (episodes)")

# magnitude gradient on the TRIGGER's own dials
lvl = tnx - tnx.shift(21)
rows = []
for lo, hi, lbl in [(-9, 0.10, "TNX 21d chg <=0.10pt (TODAY +0.108)"),
                    (0.10, 0.25, "0.10-0.25pt"), (0.25, 9, ">0.25pt")]:
    m = (lvl.loc[epi] > lo) & (lvl.loc[epi] <= hi)
    rows.append(summarize(ret5.loc[epi[m.values]].values, lbl))
show(rows, "h=5 forward by YIELD-RISE magnitude (episodes)")

r21_dx = dx.pct_change(21)
rows = []
for lo, hi, lbl in [(-9, -0.04, "DX 21d <=-4%"), (-0.04, -0.02, "-4 to -2%"),
                    (-0.02, 9, ">-2% (TODAY -1.33%)")]:
    m = (r21_dx.loc[epi] > lo) & (r21_dx.loc[epi] <= hi)
    rows.append(summarize(ret5.loc[epi[m.values]].values, lbl))
show(rows, "h=5 forward by DOLLAR-FALL magnitude (episodes)")
