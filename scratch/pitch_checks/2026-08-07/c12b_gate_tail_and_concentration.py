"""C12 follow-up: the gated cell is the only thing in the four candidates that
showed a pulse, so stress it.

Three questions:
 (a) Do the 36 gated pre-expiry windows concentrate in a few calm years?
 (b) Does the VIX rank5<=25 gate actually PROTECT, or does the -14% worst case
     inside the gated pre-expiry sample just reflect 36 monthly draws? Measure
     the worst 8td SVXY window over ALL gate-satisfying days.
 (c) Where does the 7.5% V4 calendar overlap actually come from?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SVXY", "^VIX"])
sv = px["SVXY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
cal = sv.index
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
vxp = [d for d in ev.loc[ev.event == "vix_expiry", "date"] if d <= cal[-1]]
opx = [d for d in ev.loc[ev.event == "opex", "date"] if d <= cal[-1]]
vr5 = pct_rank(vix, 5).reindex(cal)
K = 8

W = []
for E in vxp:
    prior = cal[cal < E]
    if len(prior) == 0:
        continue
    xi = pos[prior[-1]]
    if xi - K >= 0:
        W.append((cal[xi - K], cal[xi]))
ent = pd.DatetimeIndex([a for a, _ in W])
V = np.array([sv.loc[b] / sv.loc[a] - 1.0 for a, b in W])
g = vr5.reindex(ent).to_numpy() <= 25

# --- (a) concentration -----------------------------------------------------
gd, gv = ent[g], V[g]
print("=== (a) the 36 gated windows, by year ===")
print("  ", dict(pd.Series(1, index=gd).groupby(gd.year).sum()))
srt = pd.Series(gv, index=gd).sort_values()
print(f"  worst 5: {[(str(d.date()), round(100*x,2)) for d, x in srt.head(5).items()]}")
print(f"  best  5: {[(str(d.date()), round(100*x,2)) for d, x in srt.tail(5).items()]}")
print(f"  mean {100*gv.mean():+.3f}%  t={gv.mean()/(gv.std(ddof=1)/np.sqrt(len(gv))):.2f}")
# leave-one-YEAR-out
loyo = []
for y in sorted(set(gd.year)):
    m = gd.year != y
    v = gv[m]
    loyo.append({"drop_year": y, "n": int(m.sum()), "mean_pct": 100 * v.mean(),
                 "t": v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))})
show(loyo, "leave-one-year-out on the gated cell")
i = int(np.argmax(gv))
print(f"  drop the single best window ({gd[i].date()}, {100*gv[i]:+.2f}%): "
      f"mean {100*gv.mean():+.3f}% -> {100*np.delete(gv,i).mean():+.3f}%")

# modern-era only
mm = gd >= pd.Timestamp("2018-06-01")
v = gv[mm]
print(f"\n  2018-06+ (-0.5x SVXY) gated cell: N={len(v)} mean={100*v.mean():+.3f}% "
      f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} "
      f"worst={100*v.min():.2f}% P(mean<=0)={bootstrap_p_le0(v):.3f}")
print(f"  years present: {sorted(set(gd[mm].year))}")

# --- (b) does the gate protect? -------------------------------------------
print("\n=== (b) does VIX rank5<=25 protect? worst 8td over ALL gated days ===")
f8 = fwd_ret(sv, K)
ok = (vr5 <= 25).to_numpy() & f8.notna().to_numpy()
gall = f8.to_numpy()[ok]
gdates = cal[ok]
print(f"  N gate-days (overlapping) = {len(gall)}")
print(f"  mean {100*gall.mean():+.3f}%  worst {100*gall.min():.2f}% on "
      f"{gdates[int(np.argmin(gall))].date()}")
print(f"  1st/5th pctile: {100*np.percentile(gall,1):.2f}% / "
      f"{100*np.percentile(gall,5):.2f}%   P(< -10%) = "
      f"{100*(gall<-0.10).mean():.1f}%")
for lo, nm in [("2018-06-01", "2018-06+ (-0.5x)")]:
    m2 = gdates >= pd.Timestamp(lo)
    print(f"  {nm}: N={m2.sum()} mean={100*gall[m2].mean():+.3f}% "
          f"worst={100*gall[m2].min():.2f}% on "
          f"{gdates[m2][int(np.argmin(gall[m2]))].date()}  "
          f"P(< -10%)={100*(gall[m2]<-0.10).mean():.1f}%")
# what was the VIX rank on the eve of the Feb-2018 blowup?
for d in ["2018-01-24", "2018-01-26", "2018-01-29", "2018-01-31"]:
    d = pd.Timestamp(d)
    if d in vr5.index:
        print(f"    VIX rank5 on {d.date()} = {vr5.loc[d]:.1f}  "
              f"-> fwd 8td SVXY = {100*f8.get(d, np.nan):.1f}%  "
              f"(gate would have ALLOWED it: {vr5.loc[d] <= 25})")

# --- (c) where is the V4 overlap? -----------------------------------------
print("\n=== (c) V4 calendar overlap forensics ===")
v4days = set()
for O in opx:
    if O.month == 9 or O not in pos.index:
        continue
    p = pos[O]
    if p + 3 < len(cal):
        v4days |= set(cal[p + 1: p + 4])
predays = set()
for a, b in W:
    predays |= set(cal[pos[a] + 1: pos[b] + 1])
ovl = sorted(predays & v4days)
print(f"  overlap {len(ovl)} of {len(predays)} pre-expiry held days "
      f"({100*len(ovl)/len(predays):.1f}%)")
print(f"  overlap months: {sorted(set((d.year, d.month) for d in ovl))[:12]} ...")
opx_t = [o for o in opx if o in pos.index]
gapexp = [int(pos[b] - pos[[o for o in opx_t if o < b][-1]]) for a, b in W
          if any(o < b for o in opx_t)]
print(f"  td from the PREVIOUS opex to the pre-expiry exit: "
      f"min={min(gapexp)} med={int(np.median(gapexp))} max={max(gapexp)}")
print("  (overlap arises in the months where VIX expiry lands early enough that"
      " the 8td run-in reaches back into the prior month's V4 window)")
