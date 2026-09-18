"""D1c -- the live decision trace, and the identity of the dial>=80 anchors.

Three things the verdict quotes and must be able to point at:
 1. the anchor arithmetic: 2026-09-09 IS the k=-2 session of CPI 2026-09-11,
    entry MOC 2026-09-10, exit at the 2026-09-11 print close;
 2. the three dial>=80 clear-calendar anchors that read -2.788%;
 3. proof that the entry-definition and SPY-calendar gate series never
    disagree on band membership at any historical anchor (so the 9.921 vs
    9.127 split is cosmetic, not a second cell).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, load_events, rolling_on_valid,
                       anchor_positions, summarize)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
RELEVER = pd.Timestamp("2018-02-28")
px = close_panel(["SVXY", "^VIX", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]


def relpct(v):
    rng = (rolling_on_valid(v, lambda x: x.rolling(21).max())
           - rolling_on_valid(v, lambda x: x.rolling(21).min()))
    return rolling_on_valid(rng / rolling_on_valid(v, lambda x: x.rolling(21).mean()),
                            lambda x: x.rolling(252).rank(pct=True) * 100)


REL = relpct(vix)
REL_SPY = relpct(px["^VIX"].reindex(cal))

print("1. ANCHOR ARITHMETIC")
print("   last 6 NYSE sessions:", [str(d.date()) for d in cal[-6:]])
print("   CPI print 2026-09-11. k=-1 = 2026-09-10 (today, PPI session), "
      "k=-2 = 2026-09-09.")
print(f"   gate is read at the k=-2 CLOSE: rel-range pctile "
      f"{float(REL.dropna().iloc[-1]):.3f} on 2026-09-09 -> inside (5,15].")
print("   lag=1 entry -> MOC at the 2026-09-10 close; h=1 exit -> the "
      "2026-09-11 CPI close. No lookahead: the gate close is already known.")

print("\n2. BAND-MEMBERSHIP AGREEMENT BETWEEN THE TWO LIVE CONVENTIONS")
a = ((REL > 5) & (REL <= 15)).reindex(cal, fill_value=False)
b = ((REL_SPY > 5) & (REL_SPY <= 15)).reindex(cal, fill_value=False)
both = REL.reindex(cal).notna() & REL_SPY.reindex(cal).notna()
dis = cal[both.values & (a.values != b.values)]
print(f"   sessions where both series are defined: {int(both.sum())}")
print(f"   sessions where in-band membership DISAGREES: {len(dis)}")
if len(dis):
    print("   " + ", ".join(f"{d.date()} {float(REL[d]):.2f}/{float(REL_SPY[d]):.2f}"
                            for d in dis[-15:]))
print(f"   max |entry-def minus SPY-cal| over the last 252 sessions: "
      f"{float((REL.reindex(cal) - REL_SPY.reindex(cal)).abs().tail(252).max()):.3f} pts")

print("\n3. THE DIAL>=80 CLEAR-CALENDAR ANCHORS")
KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALLP = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
posn = pd.Series(range(len(cal)), index=cal)
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], -2)
    for i, ap in enumerate(p):
        d0 = kept[i]
        nxt = ALLP[ALLP > d0]
        rw = 99 if len(nxt) == 0 else int(posn.get(nxt[0], 0) - posn.get(d0, 0))
        rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
F = pd.DataFrame(rows).set_index("anchor").sort_index()
F = F[~F.index.duplicated(keep="first")].assign(
    runway_td=F.groupby(level=0)["runway_td"].min())
F["rel"] = REL.reindex(F.index).values
F["s1"] = fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(F.index).values
F["p1"] = fwd_lag(px["SPY"].dropna(), 1, lag=1).reindex(F.index).values
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
F["dial"] = frag["63d"].rolling(10).mean().reindex(F.index).values
CL = F[(F["runway_td"] >= 3) & F["s1"].notna() & (F.index >= RELEVER)]
hi = CL[CL["dial"] >= 80]
print(hi.assign(s1=(100 * hi["s1"]).round(2), p1=(100 * hi["p1"]).round(2),
                rel=hi["rel"].round(2), dial=hi["dial"].round(1)).to_string())
print(f"   mean {100*hi['s1'].mean():+.3f}%  record "
      f"{int((hi['s1']>0).sum())}-{int((hi['s1']<0).sum())}")
print(f"   for contrast, dial [60,80): n={int(((CL['dial']>=60)&(CL['dial']<80)).sum())} "
      f"{100*CL.loc[(CL['dial']>=60)&(CL['dial']<80),'s1'].mean():+.3f}%")
print("\n   and the same anchors' dial band, all-era, for the 31 ARMED ones:")
A = F[(F["runway_td"] >= 3) & F["s1"].notna() & (F["rel"] > 5) & (F["rel"] <= 15)]
print(f"   armed anchors with a dial reading: {int(A['dial'].notna().sum())} of {len(A)}; "
      f"max dial {A['dial'].max():.1f}; live 87.66")
print(f"   what a dial <= 68 arm would have required today: a fall of "
      f"{87.66 - A['dial'].max():.2f} points in ma10(63d).")
