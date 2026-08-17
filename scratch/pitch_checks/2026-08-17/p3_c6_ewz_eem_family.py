"""C6 confirmation - EWZ washout (z10 -1.68, the tape's lowest) against EEM.

This is NOT a fresh round-1/round-2 development. The registry closed this as a
FAMILY on 2026-08-13:

  "A single market breaking inside an intact thrust, FXI form. The third
   country tested on this shape after EWZ (twice) ... the pair as pitched is
   wrong-signed (residual against EEM -0.277%) ... the EEM-positive gate also
   puts SPY below its 200d on 0.0% of trigger days against a 19.7% base rate
   ... Treat 'one market decouples from a risk-on thrust' as a dead FAMILY."

plus "The Brazil five-day washout, long form" (top-2 episodes +60.6pp of a
+85.8pp total; tightening rank5 flips the sign).

The only thing that could reopen it is a MATERIALLY different trigger. Today's
claim to novelty is the z10 depth (-1.68, lowest on the 218-name tape) rather
than a rank cut. So this script asks exactly one question: does the z10
construction, at today's depth, change the sign of the family result? If not,
it is the same corpse and the kill cites a live number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H_MAIN = 5
pan = close_panel(["EWZ", "EEM", "SPY"]).dropna()
IDX = pan.index
ewz, eem, spy = pan["EWZ"], pan["EEM"], pan["SPY"]
print(f"panel {IDX[0].date()} .. {IDX[-1].date()}  N={len(IDX)}")

z10 = zscore(ewz, 10)
eem5 = eem.pct_change(5)
eem63r = pct_rank(eem, 63)
ewz63r = pct_rank(ewz, 63)
spy200 = spy.rolling(200).mean()
below = spy < spy200
base_bear = float(below.mean())

print(f"TODAY: EWZ z10={z10.iloc[-1]:+.2f}  EEM 5d={100*eem5.iloc[-1]:+.2f}%  "
      f"EEM 63d rank={eem63r.iloc[-1]:.1f}  EWZ 63d rank={ewz63r.iloc[-1]:.1f}  "
      f"SPY below 200d={bool(below.iloc[-1])}")
print("NOTE the premise: EEM's own 63d rank is LOWER than EWZ's today "
      f"({eem63r.iloc[-1]:.1f} vs {ewz63r.iloc[-1]:.1f}); 'the parent is firm' is a "
      "5-day statement only.")
print(f"base rate SPY below 200d = {100*base_bear:.1f}%")

MASK = ((z10 <= -1.5) & (eem5 > 0)).fillna(False)
sig = IDX[MASK.values]
print(f"\ntrigger (EWZ z10<=-1.5 & EEM 5d>0): N={len(sig)} days, "
      f"episodes(10td)={len(declusters(sig, 10, IDX))}, "
      f"bear-tape {100*float(below.loc[sig].mean()):.1f}% vs base {100*base_bear:.1f}%")

variants = {
    "z10<=-1.25 & EEM+": ((z10 <= -1.25) & (eem5 > 0)).fillna(False),
    "z10<=-1.5 & EEM+ (BASE)": MASK,
    "z10<=-1.75 & EEM+ (today -1.76)": ((z10 <= -1.75) & (eem5 > 0)).fillna(False),
    "z10<=-2.0 & EEM+": ((z10 <= -2.0) & (eem5 > 0)).fillna(False),
    "z10<=-1.5, NO EEM gate": (z10 <= -1.5).fillna(False),
    "z10<=-1.5 & EEM 5d<0 (mirror)": ((z10 <= -1.5) & (eem5 < 0)).fillna(False),
    "z10<=-1.5 & EEM+ & EEM 63d rank>=50": ((z10 <= -1.5) & (eem5 > 0) & (eem63r >= 50)).fillna(False),
}

battery(pan, MASK, [("EWZ", 1.0)], H_MAIN, "C6 LONG EWZ outright", cost_bps=5.0,
        variants=variants, min_gap=10, event_kinds=("opex", "vix_expiry"))
battery(pan, MASK, [("EWZ", 1.0), ("EEM", -1.0)], H_MAIN,
        "C6 LONG EWZ / SHORT EEM (equal $)", cost_bps=5.0, min_gap=10,
        event_kinds=("opex", "vix_expiry"))

# ------------------------------------------------------------------ THE BETA TRAP
print("\n" + "=" * 92)
print("THE MANDATORY BETA-NEUTRAL RESIDUAL")
print("=" * 92)
epi = declusters(sig, 10, IDX)
for h in (3, 5, 10):
    rw = fwd_lag(ewz, h, 1)
    rm = fwd_lag(eem, h, 1)
    ok = rw.notna() & rm.notna()
    b = float(np.polyfit(rm[ok], rw[ok], 1)[0])
    show([summarize(rw.reindex(epi).dropna().values, f"h={h} EWZ leg"),
          summarize(rm.reindex(epi).dropna().values, f"h={h} EEM leg"),
          summarize((rw - rm).reindex(epi).dropna().values, f"h={h} equal-$ spread"),
          summarize((rw - b * rm).reindex(epi).dropna().values,
                    f"h={h} BETA-NEUTRAL resid (beta={b:.2f})"),
          summarize((rw - b * rm)[ok].values, f"h={h} resid all days (control)")],
         f"h={h}, episodes N={len(epi)}")

# ------------------------------------------------------------------ z10 depth grid
print("\n" + "=" * 92)
print("DEPTH GRID - does today's depth change the sign? (h=5, episodes, beta-neutral)")
print("=" * 92)
rw = fwd_lag(ewz, H_MAIN, 1)
rm = fwd_lag(eem, H_MAIN, 1)
ok = rw.notna() & rm.notna()
b = float(np.polyfit(rm[ok], rw[ok], 1)[0])
rows = []
for cut in (-1.0, -1.25, -1.5, -1.68, -1.75, -2.0, -2.25):
    for gate, glbl in ((eem5 > 0, "EEM 5d>0"), (pd.Series(True, index=IDX), "no gate")):
        m = ((z10 <= cut) & gate).fillna(False)
        s = IDX[m.values]
        if len(s) < 3:
            rows.append({"z10_cut": cut, "gate": glbl, "n_days": len(s), "n_epi": 0})
            continue
        e = declusters(s, 10, IDX)
        out = rw.reindex(e).dropna()
        sp = (rw - rm).reindex(e).dropna()
        rz = (rw - b * rm).reindex(e).dropna()
        rows.append({"z10_cut": cut, "gate": glbl, "n_days": len(s), "n_epi": len(out),
                     "outright_pct": round(100 * out.mean(), 3),
                     "spread_pct": round(100 * sp.mean(), 3),
                     "resid_pct": round(100 * rz.mean(), 3),
                     "hit_out": round(100 * (out > 0).mean(), 1),
                     "bear_frac": round(100 * float(below.loc[s].mean()), 1)})
show(rows, f"h={H_MAIN}, beta fixed at the full-sample {b:.2f}")

print("\n" + "=" * 92)
print("CONCENTRATION of the outright cell (the 08-13 Brazil kill's objection)")
print("=" * 92)
v = rw.reindex(epi).dropna()
print(cluster_note(v.index, v.values))
print("episodes:", ", ".join(f"{d.date()}:{100*x:+.1f}" for d, x in v.items()))
