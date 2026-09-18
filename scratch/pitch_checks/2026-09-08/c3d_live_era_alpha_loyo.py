"""C3 round 2c -- the live-era arm number.

Two things left to price before C3 is closed:
  1. the -0.5x leverage era on its own, beta-adjusted with a beta fitted OFF
     the trigger set, for the PITCHED entry (print1-2) and the POST entry
     (on print 2). The live era is the only era that trades.
  2. leave-one-year-out floors, so the "12 of 15 years positive" claim is
     priced rather than asserted.
  3. the beta leg's own era fragility: SPY on the pair anchors, full history
     vs the SVXY era, at h=5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

px = close_panel(["SVXY", "SPY"])
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)
KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(
    [load_events([k])["date"] for k in KINDS]).unique()))
ppi = load_events(["ppi"])["date"]
p_all, _ = anchor_positions(cal, ppi, 0)
print_pos = {int(pos.get(d, -1)) for d in ALL_PRINTS}
pair_pos = [p for p in p_all if (p + 1) in print_pos]

sv = px["SVXY"].dropna()
spy = px["SPY"].dropna()
CUT = pd.Timestamp("2018-02-28")


def alpha_on(k, h, since=None):
    rs, rp = fwd_lag(sv, h, lag=1), fwd_lag(spy, h, lag=1)
    both = pd.concat([rs, rp], axis=1).dropna()
    both.columns = ["svxy", "spy"]
    if since is not None:
        both = both[both.index >= since]
    a = pd.DatetimeIndex(cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]])
    trig = both.index.intersection(a)
    off = both.index.difference(a)
    if len(trig) < 5 or len(off) < 100:
        return None
    b, c = np.polyfit(both.loc[off, "spy"], both.loc[off, "svxy"], 1)
    res = (both["svxy"] - (b * both["spy"] + c)).loc[trig].values
    raw = both.loc[trig, "svxy"].values
    return {"n": len(trig), "beta_off": round(b, 2),
            "raw_pct": round(100 * raw.mean(), 3),
            "alpha_pct": round(100 * res.mean(), 3),
            "alpha_hit": round(100 * (res > 0).mean(), 1),
            "alpha_t": round(res.mean() / (res.std(ddof=1) / np.sqrt(len(res))), 2),
            "alpha_sign_p": round(sign_test(int((res > 0).sum()), len(res)), 4)}


print("=" * 100)
print("1. BETA-ADJUSTED ALPHA (beta fitted off the trigger set)")
print("=" * 100)
rows = []
for lbl, k, h, since in (("PITCHED print1-2 h=5, FULL", -3, 5, None),
                         ("PITCHED print1-2 h=5, LIVE -0.5x era", -3, 5, CUT),
                         ("POST on print2 h=5, FULL", 0, 5, None),
                         ("POST on print2 h=5, LIVE -0.5x era", 0, 5, CUT),
                         ("POST on print2 h=3, LIVE -0.5x era", 0, 3, CUT)):
    r = alpha_on(k, h, since)
    if r:
        r["label"] = lbl
        rows.append(r)
show(rows, "SVXY alpha over SPY")

print("\n" + "=" * 100)
print("2. LEAVE-ONE-YEAR-OUT floors (raw, h=5)")
print("=" * 100)
for lbl, k, since in (("PITCHED print1-2, FULL", -3, None),
                      ("PITCHED print1-2, LIVE era", -3, CUT),
                      ("POST on print2, LIVE era", 0, CUT)):
    r5 = fwd_lag(sv, 5, lag=1)
    a = pd.DatetimeIndex(cal[[p + k for p in pair_pos if 0 <= p + k < len(cal)]])
    s = r5.reindex(a).dropna()
    if since is not None:
        s = s[s.index >= since]
    yrs = sorted(set(s.index.year))
    floors = {y: 100 * s[s.index.year != y].mean() for y in yrs}
    worst = min(floors, key=floors.get)
    print(f"  {lbl:28s} n={len(s):3d} full {100*s.mean():+.3f}%  "
          f"LOYO min {floors[worst]:+.3f}% (drop {worst})  "
          f"neg years {[y for y in yrs if s[s.index.year==y].mean()<0]}")

print("\n" + "=" * 100)
print("3. THE BETA LEG'S OWN ERA FRAGILITY -- SPY on the pair anchors, h=5")
print("=" * 100)
rp = fwd_lag(spy, 5, lag=1)
a = pd.DatetimeIndex(cal[[p - 3 for p in pair_pos if p - 3 >= 0]])
for lbl, sub in (("full history 2000+", rp.reindex(a).dropna()),
                 ("SVXY era 2011-10+", rp.reindex(a).dropna().loc["2011-10-01":]),
                 ("live -0.5x era 2018-03+", rp.reindex(a).dropna().loc[CUT:])):
    base = rp.dropna() if "full" in lbl else rp.dropna().loc[sub.index[0]:]
    print(f"  SPY {lbl:24s} n={len(sub):3d} {100*sub.mean():+.3f}% vs same-span "
          f"drift {100*base.mean():+.3f}% -> edge {100*(sub.mean()-base.mean()):+.3f}pp")
