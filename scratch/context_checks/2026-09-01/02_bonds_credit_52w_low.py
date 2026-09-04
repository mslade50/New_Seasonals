"""IEF and LQD both close at a trailing-252 low on the same session, while SPY
sits within 3% of its own trailing-252 high.

Tonight: IEF dist_52w_low 0.00%, LQD 0.00%, TLT 0.64%, HYG 2.92%, SPY -2.07%
from its 52-week high. The engine fired P2/P2b on IEF and LQD separately with
n=8 to 12 own-forward cells that say nothing. The joint state is the cell.

Convention: context brief, so forward returns are lag=0 close-to-close from the
anchor close. h=1 is the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["IEF", "LQD", "TLT", "HYG", "SPY", "^TNX"]
px = close_panel(TKRS).dropna(subset=["IEF", "LQD", "SPY"])
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}, {len(px)} sessions")

W = 252
low = {t: rolling_on_valid(px[t], lambda x: x.rolling(W, min_periods=200).min())
       for t in TKRS}
high = {t: rolling_on_valid(px[t], lambda x: x.rolling(W, min_periods=200).max())
        for t in TKRS}

at_low = lambda t: px[t] <= low[t] * (1 + 1e-9)
near_high = lambda t, tol: px[t] >= high[t] * (1 - tol)

both_bonds = at_low("IEF") & at_low("LQD")
joint = both_bonds & near_high("SPY", 0.03)

print(f"\nIEF at a 252d low:            {int(at_low('IEF').sum())} sessions")
print(f"LQD at a 252d low:            {int(at_low('LQD').sum())} sessions")
print(f"BOTH on the same session:     {int(both_bonds.sum())} sessions")
print(f"  ... and SPY within 3% of its 252d high: {int(joint.sum())} sessions")

all_dates = px.index
trig_raw = all_dates[joint.fillna(False).values]
print(f"\nraw joint dates ({len(trig_raw)}):")
for d in trig_raw:
    print(f"  {d.date()}  SPY {100*(px['SPY'][d]/high['SPY'][d]-1):+.2f}% from high, "
          f"TLT {100*(px['TLT'][d]/low['TLT'][d]-1):+.2f}% over its low")

# episodes: 21td declustering, so one credit-stress leg counts once
epi = declusters(trig_raw, 21, all_dates)
print(f"\nepisodes after 21td declustering: {len(epi)}")
print("  " + ", ".join(str(d.date()) for d in epi))

print("\n=== forward returns from the anchor close, lag=0 ===")
for subj in ["SPY", "LQD", "IEF", "TLT", "HYG"]:
    rows = []
    for h in (1, 5, 10, 21):
        r = fwd_ret(px[subj], h)
        v = r.loc[epi].dropna()
        s = summarize(v.values, f"h={h}")
        if s["n"]:
            base = r.dropna()
            s["ctl_all_pct"] = round(100 * base.mean(), 3)
            s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
            wins = int((v.values > 0).sum())
            s["record"] = f"{wins}-{s['n']-wins}"
            s["sign_p"] = round(sign_test(wins, s["n"]), 4)
        rows.append(s)
    show(rows, f"{subj} after the joint state")

# is the SPY-near-high condition doing work, or is it the bond low alone?
print("\n=== control: BOTH bonds at a low, WITHOUT the SPY-near-high filter ===")
epi_all = declusters(all_dates[both_bonds.fillna(False).values], 21, all_dates)
for subj in ["SPY", "LQD"]:
    rows = []
    for h in (1, 5, 21):
        r = fwd_ret(px[subj], h)
        v = r.loc[epi_all].dropna()
        s = summarize(v.values, f"h={h}")
        wins = int((v.values > 0).sum())
        s["record"] = f"{wins}-{s['n']-wins}"
        s["sign_p"] = round(sign_test(wins, s["n"]), 4)
        rows.append(s)
    show(rows, f"{subj}, bond-low only ({len(epi_all)} episodes)")

# local control: the +/-126td neighbourhood of the joint dates, joint days removed
print("\n=== local control (+/-126td neighbourhood, trigger days removed) ===")
ctl = local_control(all_dates, trig_raw, 126)
for subj in ["SPY", "LQD"]:
    for h in (1, 5, 21):
        r = fwd_ret(px[subj], h)
        c = r.loc[r.index.intersection(ctl)].dropna()
        e = r.loc[epi].dropna()
        print(f"  {subj} h={h}: episodes {100*e.mean():+.3f}% (n {len(e)}) vs "
              f"local {100*c.mean():+.3f}% (n {len(c)}) vs "
              f"all {100*r.dropna().mean():+.3f}%")

print("\n=== era split, SPY h=5 and h=21 ===")
for h in (5, 21):
    r = fwd_ret(px["SPY"], h).loc[epi].dropna()
    show(era_split(r.index, r.values), f"SPY h={h}")

print("\n=== concentration, SPY h=21 ===")
r21 = fwd_ret(px["SPY"], 21).loc[epi].dropna()
print("  " + cluster_note(r21.index, r21.values, k=2))
