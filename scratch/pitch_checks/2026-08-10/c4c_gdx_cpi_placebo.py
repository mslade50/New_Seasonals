"""C4 round 2b -- the placebo battery that decides it.

Round 2 showed the CPI-2 anchor does ALL the work (+0.962% vs +0.131% for the
same thrust NOT on a CPI-2 anchor).  But the anchor placebo showed EVERY
scheduled-event -2 anchor is positive (ppi +0.609, nfp +0.638, opex +0.573),
which is the 2026-08-07 VIX-expiry lesson: "mid-month position plus noise".

So: (1) how much do those anchor sets OVERLAP with CPI-2, (2) is it month
position, (3) is it day of week, (4) definition neighbours on the thrust
threshold, (5) today's cluster depth.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG, H = 1, 2
P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
g = P["GDX"]
idx = g.index
rk5 = pct_rank(g, 5)
fw = fwd_lag(g, H, LAG)
ok = fw.notna()
VALID = idx[ok.values]
ASOF = idx[-1]


def anchor_k(kind: str, k: int = 2) -> pd.DatetimeIndex:
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


CPI2 = anchor_k("cpi")
THR = (rk5 >= 80.0).fillna(False)
TRIG = pd.DatetimeIndex(CPI2).intersection(idx[THR.values & ok.values])

# ------------------------------------------------------------- 1. overlap
print("### 1. do the 'placebo' anchors overlap CPI-2? ###")
for kind in ("ppi", "nfp", "opex", "fomc_decision"):
    a = anchor_k(kind)
    inter = pd.DatetimeIndex(a).intersection(CPI2)
    print(f"  {kind}-2: n={len(a):4d}  overlap with CPI-2 = {len(inter):4d} "
          f"({100*len(inter)/max(len(a),1):.0f}% of {kind}-2 anchors)")
# and: the *thrust* subsets
print("\n  thrust-conditioned overlap (the actual trigger sets):")
for kind in ("ppi", "nfp", "opex", "fomc_decision"):
    a = pd.DatetimeIndex(anchor_k(kind)).intersection(idx[THR.values & ok.values])
    inter = a.intersection(TRIG)
    print(f"  {kind}-2 x thrust: n={len(a):3d}  overlap with CPI cell = {len(inter):3d} "
          f"({100*len(inter)/max(len(a),1):.0f}%)")

# strictly disjoint placebo: event -2 anchors that are NOT CPI-2 anchors
print("\n  DISJOINT placebo (event-2 anchors with CPI-2 removed), thrust ON:")
rows = []
for kind in ("ppi", "nfp", "opex", "fomc_decision"):
    a = pd.DatetimeIndex(anchor_k(kind)).intersection(idx[THR.values & ok.values])
    a = a.difference(CPI2)
    if len(a) < 4:
        rows.append({"label": f"{kind}-2 ex-CPI", "n": len(a)})
        continue
    e = declusters(a, H, VALID)
    v = fw.loc[e].values
    r = summarize(v, f"{kind}-2 ex-CPI x thrust")
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
e = declusters(TRIG, H, VALID)
vC = fw.loc[e].values
rC = summarize(vC, "CPI-2 x thrust (the cell)")
rC["sign_p"] = round(sign_test(int((vC > 0).sum()), len(vC)), 4)
rows.append(rC)
show(rows, "disjoint anchor placebo")

# ------------------------------------------------------- 2. month position
print("\n### 2. MONTH-POSITION placebo (the VIX-expiry lesson) ###")
dom_bd = pd.Series([0] * len(idx), index=idx)
cur, last_m = 0, None
for i, d in enumerate(idx):
    if last_m != (d.year, d.month):
        cur, last_m = 0, (d.year, d.month)
    cur += 1
    dom_bd.iloc[i] = cur
tp = dom_bd.loc[TRIG]
print(f"  CPI-2 x thrust trigger business-day-of-month: min={tp.min()} "
      f"p25={tp.quantile(.25):.0f} median={tp.median():.0f} p75={tp.quantile(.75):.0f} max={tp.max()}")
lo, hi = int(tp.quantile(.10)), int(tp.quantile(.90))
mpos = ((dom_bd >= lo) & (dom_bd <= hi))
matched = idx[THR.values & ok.values & mpos.values]
matched = pd.DatetimeIndex(matched).difference(CPI2)
em = declusters(matched, H, VALID)
vm = fw.loc[em].values
show([rC,
      summarize(vm, f"MATCHED month-pos bd {lo}-{hi}, thrust, NOT CPI-2"),
      summarize(fw.loc[declusters(idx[THR.values & ok.values], H, VALID)].values,
                "thrust, any day")],
     "month-position-matched control")

# ------------------------------------------------------- 3. day of week
print("\n### 3. day-of-week placebo ###")
dows = pd.DatetimeIndex(TRIG).dayofweek
print("  trigger dow counts:", dict(pd.Series(dows).value_counts().sort_index()))
rows = []
for dw in sorted(set(dows)):
    a = idx[THR.values & ok.values]
    a = pd.DatetimeIndex([d for d in a if d.dayofweek == dw]).difference(CPI2)
    ee = declusters(a, H, VALID)
    rows.append(summarize(fw.loc[ee].values, f"thrust, dow={dw}, NOT CPI-2"))
show(rows, "same thrust on each weekday, CPI-2 removed")

# ------------------------------------------------ 4. definition neighbours
print("\n### 4. definition neighbours ###")
rows = []
for thr in (60.0, 70.0, 80.0, 85.0, 90.0, 95.0):
    m = (rk5 >= thr).fillna(False)
    t = pd.DatetimeIndex(CPI2).intersection(idx[m.values & ok.values])
    ee = declusters(t, H, VALID)
    v = fw.loc[ee].values
    r = summarize(v, f"CPI-2 x rank5>={thr:.0f}")
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    r["boot_p"] = round(bootstrap_p_le0(v), 3)
    rows.append(r)
show(rows, "thrust threshold sweep (k=2, h=2)")

# an absolute-return version of the thrust, not a rank
rows = []
r5 = g.pct_change(5) * 100
for thr in (2.0, 4.0, 6.0, 8.0):
    m = (r5 >= thr).fillna(False)
    t = pd.DatetimeIndex(CPI2).intersection(idx[m.values & ok.values])
    ee = declusters(t, H, VALID)
    v = fw.loc[ee].values
    r = summarize(v, f"CPI-2 x 5d ret >= {thr:.0f}%")
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "same idea, absolute 5d return instead of a rank (today = +21.3%)")

# k neighbours: is k=2 special or does the whole pre-print window work?
rows = []
for k in (1, 2, 3):
    a = anchor_k("cpi", k)
    t = pd.DatetimeIndex(a).intersection(idx[THR.values & ok.values])
    ee = declusters(t, H, VALID)
    v = fw.loc[ee].values
    r = summarize(v, f"k={k} (entry CPI-{k-1} close, exit CPI+{H-k+1})")
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "anchor-k neighbours at h=2")

# --------------------------------------------- 5. today's cluster depth
print("\n### 5. today's cluster depth on the ACTUAL trigger ###")
m = THR.fillna(False)
p = list(idx).index(ASOF)
d = 0
while p - d >= 0 and bool(m.iloc[p - d]):
    d += 1
print(f"  GDX rank5>=80 consecutive sessions incl. today: {d}")
runs, run = [], 0
for v in m.values:
    run = run + 1 if v else 0
    if v:
        runs.append(run)
runs = np.array(runs)
print(f"  historical run-length-so-far on trigger days: p25={np.percentile(runs,25):.0f} "
      f"p50={np.median(runs):.0f} p75={np.percentile(runs,75):.0f} max={runs.max()}")
dep = pd.Series(runs, index=idx[m.values])
tdep = dep.reindex(pd.DatetimeIndex(TRIG)).values
print(f"  cluster depth AT the 49 CPI-2 triggers: p50={np.median(tdep):.0f} "
      f"mean={tdep.mean():.1f}; today = {d}")
show([summarize(fw.loc[pd.DatetimeIndex(TRIG)[tdep <= 2]].values, "trigger depth<=2"),
      summarize(fw.loc[pd.DatetimeIndex(TRIG)[tdep > 2]].values, "trigger depth>2")],
     "does depth matter? (day-level, CPI-2 triggers are already ~1/month)")

# how extreme is today's 5d move vs the trigger population?
r5t = r5.loc[pd.DatetimeIndex(TRIG)].values
print(f"\n  today's GDX 5d return = {r5.loc[ASOF]:+.2f}%; trigger population "
      f"p50={np.median(r5t):+.2f}% p90={np.percentile(r5t,90):+.2f}% max={r5t.max():+.2f}% "
      f"-> today is the {100*(r5t < r5.loc[ASOF]).mean():.0f}th pctile of the cell")
