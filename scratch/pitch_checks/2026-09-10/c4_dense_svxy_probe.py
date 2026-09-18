"""c4 — honesty probe on the ONE cell that came out of C3's gate attribution
with a sign p of 0.000: LONG SVXY on a DENSE calendar ALONE, no compression
condition (h=6: +1.540%, N=126 episodes, t 2.10; h=10: +2.606%, t 3.00).

This is NOT a candidate. It was not pre-specified, it fell out of a grid this
checker walked, and reporting it as a survivor would be manufacturing one.
The probe exists so the finding is characterised truthfully rather than
dangled: is it stable in the tradeable -0.5x era, does it outrank its own
placebos, and is it distinguishable from SVXY's unconditional carry?

Note the reason it is interesting at all: SPY's identical dense cell is FLAT
(+0.078% full 26y t 0.44; +0.003% in the -0.5x era on 58 episodes), so any
SVXY lift is a vol-term-structure effect, not equity beta. Which is also the
reason it deserves suspicion rather than a pitch slot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

EVENT_KINDS = ["nfp", "cpi", "ppi", "fomc_decision", "opex",
               "quad_witching", "vix_expiry"]
WIN, THRESH = 6, 4
RELEVER = pd.Timestamp("2018-02-28")

px = close_panel(["SPY", "SVXY", "^VIX", "^VIX3M"])
spy = px["SPY"].dropna()
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events(EVENT_KINDS)
locs = idx.searchsorted(pd.DatetimeIndex(ev["date"]))
cnt = np.zeros(len(idx))
for L in locs[locs < len(idx)]:
    cnt[L] += 1
cs = np.concatenate([[0.0], np.cumsum(cnt)])
dens = pd.Series([np.nan if p + WIN >= len(idx) else cs[p + WIN + 1] - cs[p + 1]
                  for p in range(len(idx))], index=idx)
dense = (dens >= THRESH).fillna(False)

H = 6
ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
valid = ret.dropna().index
t_all = pd.DatetimeIndex(idx[dense.values]).intersection(valid)

print("=" * 80)
print("1. ERA STABILITY — LONG SVXY on DENSE alone (the re-lever is Feb 2018)")
rows = []
for lbl, lo, hi in [("-1.0x era 2011-10..2018-02", None, RELEVER),
                    ("-0.5x era 2018-02+ (THE TRADEABLE ONE)", RELEVER, None),
                    ("full (DO NOT POOL, shown for contrast)", None, None)]:
    v5 = valid
    if lo is not None:
        v5 = v5[v5 >= lo]
    if hi is not None:
        v5 = v5[v5 < hi]
    t = t_all.intersection(v5)
    e = declusters(t, max(H, WIN), v5)
    v = ret.loc[e].values
    s = summarize(v, f"DENSE {lbl}")
    s["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
    base = ret.loc[v5]
    rows.append({"label": f"  ctrl all days, {lbl}", "n": len(base),
                 "mean_pct": round(100 * base.mean(), 3),
                 "hit": round(100 * (base > 0).mean(), 1)})
show(rows, "   era split, never pooled")

print("\n2. PLACEBO LADDER k=-5..+5, -0.5x era only")
v5 = valid[valid >= RELEVER]
rows = []
for k in range(-5, 6):
    sh = []
    for d in t_all.intersection(v5):
        p = pos.get(d)
        q = p + k
        if 0 <= q < len(idx):
            sh.append(idx[q])
    t = pd.DatetimeIndex(sorted(set(sh))).intersection(v5)
    e = declusters(t, max(H, WIN), v5)
    rows.append(summarize(ret.loc[e].values,
                          f"k={k:+d}" + ("  <-- TRUE" if k == 0 else "")))
show(rows, "   placebo ladder (-0.5x era)")
tm = [r for r in rows if "TRUE" in r["label"]][0]["mean_pct"]
print(f"   TRUE ranks {1 + sum(1 for r in rows if r.get('mean_pct', -9e9) > tm)}/11")

print("\n3. CONCENTRATION + the 2020 problem, -0.5x era")
t = t_all.intersection(v5)
e = declusters(t, max(H, WIN), v5)
v = ret.loc[e].values
print(f"   N={len(e)}  mean {100*v.mean():+.3f}%  hit {100*(v>0).mean():.1f}%  "
      f"worst {100*v.min():+.2f}% on {e[int(np.argmin(v))].date()}")
print(f"   {cluster_note(e, v)}")
yrs = pd.DatetimeIndex(e).year
by_y = pd.Series(v).groupby(yrs.values).sum()
print(f"   per-year: {dict((int(y), round(100*r,2)) for y, r in by_y.items())}")
o = np.argsort(-v)
print(f"   drop-best-2 {100*np.delete(v, o[:2]).mean():+.3f}%  "
      f"drop-best-year ({by_y.idxmax()}) "
      f"{100*v[yrs != by_y.idxmax()].mean():+.3f}%")

print("\n4. IS IT JUST CARRY? term structure at the anchor vs the control")
vx, v3 = px["^VIX"].reindex(idx).ffill(), px["^VIX3M"].reindex(idx).ffill()
ts = (vx / v3)
for lbl, dates in [("DENSE anchors", e), ("all -0.5x days", v5)]:
    x = ts.reindex(pd.DatetimeIndex(dates)).dropna()
    print(f"   {lbl:18s} N={len(x):5d}  mean VIX/VIX3M {x.mean():.4f}  "
          f"median {x.median():.4f}  share in contango(<1) "
          f"{100*(x<1).mean():.1f}%")
print(f"   today's VIX/VIX3M = {ts.dropna().iloc[-1]:.4f}")
print("\n   If the DENSE anchors simply sit in deeper contango than average,")
print("   the cell is carry selection and the event calendar is a proxy.")
