"""C11 round 1 -- dollar pullback INSIDE an uptrend, in DX futures.

Pre-specified trigger, from today's state: DX-Y.NYB z10 <= -1.5 (sharp short
-term washout) AND rank63 >= 60 AND price within 4% of its 252d high (the
pullback is inside an uptrend, not a downtrend). Vehicle DX futures, cost
~1.5 bp round trip. Entry lag=1 MOC-tomorrow. h=5 pre-specified to match the
horizon Friday's DX idea used, so the two are comparable; h-scan labelled.

REPETITION: 2026-08-07 pitched LONG DX futures, MOC entry, 5 td horizon
(fingerprint e409803df080ad9e) and that trade is STILL OPEN (exit 2026-08-14).
This script also measures the overlap of the two triggers so the "materially
different idea" claim is checked rather than asserted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["DX-Y.NYB", "UUP"])
dx_all = px["DX-Y.NYB"].dropna()
px = pd.DataFrame({"DX": dx_all})
idx = px.index
dx = px["DX"]

z10 = zscore(dx, 10)
r21 = pct_rank(dx, 21)
r63 = pct_rank(dx, 63)
hi = dx.rolling(252).max()
dist = dx / hi - 1.0
sma200 = dx.rolling(200).mean()

print(f"DX span {idx[0].date()} .. {idx[-1].date()} n={len(idx)}")
print(f"today: z10={z10.iloc[-1]:+.2f} rank21={r21.iloc[-1]:.1f} "
      f"rank63={r63.iloc[-1]:.1f} dist52wh={100*dist.iloc[-1]:+.2f}% "
      f"px/sma200={100*(dx.iloc[-1]/sma200.iloc[-1]-1):+.2f}%")

Z, R63, D = -1.5, 60.0, -0.04
m = ((z10 <= Z) & (r63 >= R63) & (dist >= D)).fillna(False)
m_z = (z10 <= Z).fillna(False)                      # no uptrend gate
m_up = ((r63 >= R63) & (dist >= D)).fillna(False)   # no washout gate
print(f"\ntriggers: full {int(m.sum())} | z10 alone {int(m_z.sum())} | "
      f"uptrend alone {int(m_up.sum())}")
run = 0
for v in m.values[::-1]:
    if v:
        run += 1
    else:
        break
print(f"CLUSTER DEPTH TODAY: {run} consecutive trigger sessions")

variants = {
    "z10<=-1.25": ((z10 <= -1.25) & (r63 >= R63) & (dist >= D)).fillna(False),
    "z10<=-1.75": ((z10 <= -1.75) & (r63 >= R63) & (dist >= D)).fillna(False),
    "z10<=-2.0": ((z10 <= -2.0) & (r63 >= R63) & (dist >= D)).fillna(False),
    "rank63>=50": ((z10 <= Z) & (r63 >= 50) & (dist >= D)).fillna(False),
    "rank63>=70": ((z10 <= Z) & (r63 >= 70) & (dist >= D)).fillna(False),
    "dist>=-2.5%": ((z10 <= Z) & (r63 >= R63) & (dist >= -0.025)).fillna(False),
    "dist>=-6%": ((z10 <= Z) & (r63 >= R63) & (dist >= -0.06)).fillna(False),
    "GATE ATTR: z10 alone": m_z,
    "GATE ATTR: uptrend alone": m_up,
    "rank21<=20 instead of z10": ((r21 <= 20) & (r63 >= R63) & (dist >= D)).fillna(False),
}

for h in (5, 10):
    battery(px, m, [("DX", 1.0)], h=h,
            title=f"LONG DX | z10<=-1.5 inside rank63>=60 within 4% of 52wh, h={h}",
            cost_bps=1.5, lag=1, min_gap=h, event_kinds=("cpi",),
            variants=variants if h == 5 else None)

print("\n" + "=" * 78)
print("SCAN (multiplicity applies): horizon 1..21")
print("=" * 78)
show(horizon_scan(px, idx[m.values], [("DX", 1.0)],
                  hs=(1, 2, 3, 5, 7, 10, 15, 21)), "DX pullback-in-uptrend")

print("\n" + "=" * 78)
print("MIDTERM SPLIT + year histogram (h=5 episodes)")
print("=" * 78)
h = 5
r = vehicle_ret(px, [("DX", 1.0)], h, 1)
val = r.notna()
e = declusters(idx[m.values & val.values], h, idx)
v = r.loc[e].values
yr = pd.DatetimeIndex(e).year
mid = yr % 4 == 2
base = r[val]
bmid = base.index.year % 4 == 2
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})"),
      summarize(base[bmid].values, "CTRL all days midterm"),
      summarize(base[~bmid].values, "CTRL all days non-midterm")], "midterm")
print("\nyear histogram (pp):")
print(pd.Series(100 * v, index=pd.DatetimeIndex(e)).groupby(yr).agg(
    ["count", "sum", "mean"]).round(3).to_string())
print("\n" + cluster_note(pd.DatetimeIndex(e), v, k=3))

print("\n" + "=" * 78)
print("COST: 1.5 bp futures round trip vs the edge, and the UUP alternative")
print("=" * 78)
for hh in (3, 5, 10):
    rr = vehicle_ret(px, [("DX", 1.0)], hh, 1)
    ee = declusters(idx[m.values & rr.notna().values], hh, idx)
    ed = 100 * 100 * rr.loc[ee].mean()
    print(f"h={hh:2d} episode edge {ed:+.1f} bps -> {ed/1.5:.1f}x the 1.5 bp "
          f"futures round trip, {ed/6.0:.1f}x the 6 bp UUP round trip "
          f"(registry: UUP needs >6 bps and gets ~6)")

print("\n" + "=" * 78)
print("REPETITION AUDIT vs 2026-08-07 (fingerprint e409803df080ad9e)")
print("  Friday: LONG DX FUT, MOC entry, 5 td, anchored on the August NFP")
print("  Today:  LONG DX FUT, MOC entry, 5 td, anchored on a price state")
print("=" * 78)
ev = load_events(["nfp"])["date"]
nfp_pos = []
for x in ev:
    p = int(idx.searchsorted(x, side="left"))
    if p < len(idx):
        nfp_pos.append(p)
nfp_mask = pd.Series(False, index=idx)
nfp_mask.iloc[nfp_pos] = True
aug_nfp = nfp_mask & (idx.month == 8)
print(f"August NFP anchors: {int(aug_nfp.sum())}; midterm-year ones: "
      f"{int((aug_nfp & (idx.year % 4 == 2)).sum())}")
# how often does the price-state trigger coincide with (or sit inside) a
# post-NFP week?  If it usually does, the "different anchor" claim is thin.
pos = pd.Series(range(len(idx)), index=idx)
trig_pos = pos[m].values
nfp_arr = np.array(nfp_pos)
near = np.array([bool((np.abs(nfp_arr - p) <= 5).any()) for p in trig_pos])
print(f"price-state triggers within 5 td of ANY NFP: {int(near.sum())} of "
      f"{len(trig_pos)} ({100*near.mean():.1f}%)  "
      f"[unconditional share of days within 5td of an NFP = "
      f"{100*np.mean([bool((np.abs(nfp_arr-p)<=5).any()) for p in range(len(idx))]):.1f}%]")
print(f"\nIS TODAY (2026-08-07 signal) WITHIN 5 TD OF THE AUG NFP? "
      f"{bool((np.abs(nfp_arr - (len(idx)-1)) <= 5).any())}")
print("Friday's open trade exits 2026-08-14 MOC. A 5 td trade entered "
      "2026-08-11 MOC exits 2026-08-18 MOC -> 3 of 5 sessions overlap.")
