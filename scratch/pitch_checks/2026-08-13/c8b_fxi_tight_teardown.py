"""c8b - teardown of C8's ONE positive slice.

The wide C8 cell is negative everywhere, but the tight variant
(FXI r5<=20 & r21>=70 & EEM 5d>0) shows long-FXI-outright 6 episodes,
100% hit, +0.917% at h=3, t=5.33. Small N is not a kill, so this slice gets
the full small-N treatment: episode dates, era distribution, what the EEM
leg did on the same windows (is it China or is it EM beta), the residual, the
sign test against FXI's OWN drifting hit rate rather than a coin, and horizon
stability.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["FXI", "EEM", "SPY"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
fxi, eem, spy = raw["FXI"]["Close"], raw["EEM"]["Close"], raw["SPY"]["Close"]
al = pd.concat([fxi.pct_change(), eem.pct_change(), spy.pct_change()], axis=1).dropna()
al.columns = ["fxi", "eem", "spy"]
b_eem = np.polyfit(al["eem"], al["fxi"], 1)[0]

m = ((pct_rank(fxi, 5) <= 20) & (pct_rank(fxi, 21) >= 70)
     & (eem.pct_change(5) > 0)).reindex(px.index).fillna(False)
trig = px.index[m.values]
epi = declusters(trig, 5, px.index)
print(f"trigger days = {[str(d.date()) for d in trig]}")
print(f"episodes     = {[str(d.date()) for d in epi]}")

print("\n=== per-episode detail, h=3 ===")
rows = []
for d in epi:
    rows.append({
        "date": str(d.date()),
        "yr": d.year,
        "midterm": d.year % 4 == 2,
        "fxi_5d": round(100 * fxi.pct_change(5).loc[d], 2),
        "fxi_r5": round(pct_rank(fxi, 5).loc[d], 1),
        "fxi_r21": round(pct_rank(fxi, 21).loc[d], 1),
        "fxi_h3": round(100 * fwd_lag(fxi, 3).loc[d], 2),
        "eem_h3": round(100 * fwd_lag(eem, 3).loc[d], 2),
        "spy_h3": round(100 * fwd_lag(spy, 3).loc[d], 2),
        "resid_h3": round(100 * vehicle_ret(px, [("FXI", 1.0), ("EEM", -b_eem)], 3).loc[d], 2),
        "fxi_h5": round(100 * fwd_lag(fxi, 5).loc[d], 2),
        "fxi_h10": round(100 * fwd_lag(fxi, 10).loc[d], 2),
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== is it China or EM beta? ===")
for h in (1, 2, 3, 5, 10):
    f = fwd_lag(fxi, h).loc[epi].values
    e = fwd_lag(eem, h).loc[epi].values
    r = vehicle_ret(px, [("FXI", 1.0), ("EEM", -b_eem)], h).loc[epi].values
    base_hit = (fwd_lag(fxi, h).dropna() > 0).mean()
    w = int((f > 0).sum())
    n = int(np.isfinite(f).sum())
    print(f"  h={h:2d}: FXI {100*np.nanmean(f):+.3f}%  EEM {100*np.nanmean(e):+.3f}%  "
          f"resid vs EEM {100*np.nanmean(r):+.3f}%   record {w}-{n-w}  "
          f"sign p vs coin {sign_test(w, n):.4f}  vs FXI own hit {base_hit:.3f} -> "
          f"{sign_test(w, n, base_hit):.4f}")

print("\n=== FXI unconditional drift on the same span, for reference ===")
for h in (3, 5):
    r = fwd_lag(fxi, h)
    print(f"  h={h}: all days {100*r.mean():+.3f}% hit {100*(r>0).mean():.1f}%")

print("\n=== era: how much of the record is the 2006-07 China bubble ===")
pre08 = np.array([d.year <= 2007 for d in epi])
r3 = fwd_lag(fxi, 3).loc[epi].values
show([summarize(r3[pre08], f"<=2007 (N={int(pre08.sum())})"),
      summarize(r3[~pre08], f">=2008 (N={int((~pre08).sum())})")])
print("  drop-best:", end=" ")
o = np.argsort(-r3)
print(f"full {100*r3.mean():+.3f}% -> drop1 {100*np.delete(r3, o[0]).mean():+.3f}% "
      f"-> drop2 {100*np.delete(r3, o[:2]).mean():+.3f}%")

print("\n=== definition fragility: nudge each threshold, outright h=3 episodes ===")
rows = []
for r5 in (15, 20, 25, 30):
    for r21 in (60, 70, 80):
        mm = ((pct_rank(fxi, 5) <= r5) & (pct_rank(fxi, 21) >= r21)
              & (eem.pct_change(5) > 0)).reindex(px.index).fillna(False)
        t = px.index[mm.values]
        e = declusters(t, 5, px.index)
        v = fwd_lag(fxi, 3).reindex(px.index).loc[e].dropna()
        if len(v):
            rows.append({"r5<=": r5, "r21>=": r21, "n_epi": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "hit": round(100 * (v > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
