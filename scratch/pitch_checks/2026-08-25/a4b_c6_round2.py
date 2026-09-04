"""C6 round 2 - the live rung, dose response, era/dial regime, cost.

Round 1 established two things:
  - the dispersion condition pays at PIT >= 90/99 and NOTHING at PIT >= 80
    (edge +0.010pp h=3, -0.001pp h=5), and today's cross-section is PIT 84.5
  - on C1's own trigger days the generic xsec spread (+0.878% h=3) BEATS
    C1's long-XLK-only (+0.225%), and XLK is the bottom-1 of 9 on the median
    trigger day, so C1 has no sector content.

This round asks whether the version that ACTUALLY FIRES TODAY is a trade:
dose response around 84.5, cost at 4 legs, era/dial split, and the k-ladder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
px = close_panel(SECTORS + ["SPY"])
S = px[SECTORS].dropna(how="any")
R5 = S.pct_change(5)
DISP = (R5.max(axis=1) - R5.min(axis=1)) * 100.0
disp_pit = DISP.dropna().rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100,
                                            raw=True).reindex(S.index)
TODAY_PIT = disp_pit.iloc[-1]
print(f"today dispersion {DISP.iloc[-1]:.2f}pp, PIT252 {TODAY_PIT:.1f}")


def legs_ret(h: int, k: int, lag: int = 1):
    fwd = pd.DataFrame({c: S[c].shift(-(lag + h)) / S[c].shift(-lag) - 1.0
                        for c in S.columns})
    ordr = R5.rank(axis=1, method="first")
    n = R5.notna().sum(axis=1)
    L = fwd.where(ordr.le(k, axis=0) & R5.notna()).mean(axis=1)
    H = fwd.where(ordr.gt(n - k, axis=0) & R5.notna()).mean(axis=1)
    return L, H, L - H


# ------------------------------------------------------------ A. dose response
print("\n########## A. DOSE RESPONSE around today's PIT of 84.5 ##########")
for h in (3, 5):
    L, H, SP = legs_ret(h, 2)
    v = SP.dropna()
    unc = 100 * SP.loc[declusters(v.index, h, v.index)].mean()
    rows = []
    for lo, hi in [(0, 100), (70, 100), (80, 100), (82, 88), (84, 100), (85, 100),
                   (90, 100), (95, 100), (99, 100)]:
        m = ((disp_pit >= lo) & (disp_pit <= hi)).fillna(False)
        e = declusters(S.index[m.values].intersection(v.index), h, v.index)
        s = summarize(SP.loc[e].values, f"PIT in [{lo},{hi}]")
        s["edge_vs_uncond"] = round(s.get("mean_pct", np.nan) - unc, 3)
        s["fires_today"] = bool(lo <= TODAY_PIT <= hi)
        rows.append(s)
    show(rows, f"h={h}  bottom2-top2   (unconditional {unc:+.3f}%)")

# ------------------------------------------------------------ B. cost
print("\n########## B. COST at 4 legs ##########")
print("  sector ETF round trip ~2-3 bps each way -> ~5 bps per leg round trip;")
print("  bottom-2/top-2 = 4 legs = ~20 bps of the notional the spread is quoted on.")
for h in (3, 5):
    L, H, SP = legs_ret(h, 2)
    v = SP.dropna()
    for lo in (80, 84, 90, 99):
        m = (disp_pit >= lo).fillna(False)
        e = declusters(S.index[m.values].intersection(v.index), h, v.index)
        mu = 100 * SP.loc[e].mean()
        print(f"  h={h} PIT>={lo:>2}: {mu:+.3f}% = {mu*100:6.1f} bps -> "
              f"{mu*100/20:4.1f}x cost   N_epi={len(e)}"
              f"{'   <-- fires today' if lo <= TODAY_PIT else ''}")

# ------------------------------------------------------------ C. k ladder
print("\n########## C. k-LADDER (definition neighbour) ##########")
rows = []
for k in (1, 2, 3, 4):
    for h in (3, 5):
        L, H, SP = legs_ret(h, k)
        v = SP.dropna()
        m = (disp_pit >= 80).fillna(False)
        e = declusters(S.index[m.values].intersection(v.index), h, v.index)
        s = summarize(SP.loc[e].values, f"k={k} h={h} PIT>=80")
        s["cost_x"] = round(100 * s.get("mean_pct", np.nan) / (2 * k * 5), 2)
        rows.append(s)
show(rows, "at the rung that fires today")

# ------------------------------------------------------------ D. era / dial
print("\n########## D. ERA + FRAGILITY-DIAL REGIME (PIT>=80 cell, h=3) ##########")
L, H, SP = legs_ret(3, 2)
v = SP.dropna()
m = (disp_pit >= 80).fillna(False)
e = declusters(S.index[m.values].intersection(v.index), 3, v.index)
vals = SP.loc[e].values
show(era_split(e, vals), "era split")
mid = np.array([d.year % 4 == 2 for d in e])
show([summarize(vals[mid], f"midterm (N={int(mid.sum())})"),
      summarize(vals[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm split")

frag = pd.read_parquet("data/rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
dial = frag["63d"].rolling(10).mean()
dv = dial.reindex(e)
have = dv.notna().values
print(f"\n  dial coverage: {int(have.sum())} of {len(e)} episodes have a reading "
      f"(pre-2016 has no dial; 2016..2026-07-02 is a RECOMPUTE vintage)")
if have.sum():
    print(f"  median trigger dial {dv[have].median():.1f}, max {dv[have].max():.1f}  "
          f"(today 89.5)")
    for lo, hi in [(0, 50), (50, 80), (80, 200)]:
        sel = have & (dv.values >= lo) & (dv.values < hi)
        if sel.sum():
            print(f"    dial [{lo},{hi}): N={int(sel.sum())} mean "
                  f"{100*vals[sel].mean():+.3f}% hit {100*(vals[sel]>0).mean():.0f}%")
        else:
            print(f"    dial [{lo},{hi}): N=0")

# ------------------------------------------------------------ E. today's basket
print("\n########## E. WHAT THE TRADE ACTUALLY IS TODAY ##########")
print(f"  long  bottom-2: {list(R5.iloc[-1].nsmallest(2).index)}  "
      f"({', '.join(f'{t} {R5.iloc[-1][t]*100:+.2f}%' for t in R5.iloc[-1].nsmallest(2).index)})")
print(f"  short top-2:    {list(R5.iloc[-1].nlargest(2).index)}  "
      f"({', '.join(f'{t} {R5.iloc[-1][t]*100:+.2f}%' for t in R5.iloc[-1].nlargest(2).index)})")
print("  i.e. long XLK+XLI / short XLV+XLP - the SAME four names as C1, C9 and C10.")
