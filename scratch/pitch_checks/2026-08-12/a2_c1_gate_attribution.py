"""C1 round 2 -- GATE ATTRIBUTION. Round 1 failed to kill on the CPI-eve
conditioner (the conditioned cell is BETTER, +0.269pp tdom-excess vs +0.113pp
parent). So attack the gate itself.

The alternative explanation the whole idea stands or falls on: the live cell is
"the session AFTER a CPI print". Is the PPI gate doing ANY work, or is the trade
simply CPI+1 drift in TLT wearing a PPI label? Run it WITHOUT the PPI gate.

Also: offset ladder -5..+3 (registry: a plateau is month position, a spike is an
event); apples-to-apples era match (the CPI-on-eve subset is 52/55 post-2018, so
comparing it to a 25-year parent is a rigged comparison); CPI-day sign
conditioner; declustered concentration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "LQD", "SPY"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
base_hit = float((d1[~np.isnan(d1)] > 0).mean())

ev = load_events()


def sess_pos(dates):
    out = []
    for x in dates:
        p = int(idx.searchsorted(x, "left"))
        if 0 <= p < N:
            out.append(p)
    return sorted(set(out))


ppi_all = set(sess_pos(ev[ev.event == "ppi"]["date"]))
cpi_all = set(sess_pos(ev[ev.event == "cpi"]["date"]))
fomc_all = set(sess_pos(ev[ev.event == "fomc_decision"]["date"]))


def rep(v, lbl):
    if len(v) == 0:
        return {"cell": lbl, "N": 0}
    w = int((v > 0).sum())
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    return {"cell": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 4),
            "hit": round(100 * w / len(v), 1),
            "t": round(v.mean() / (sd / np.sqrt(len(v))), 2) if sd else np.nan,
            "signp_vs_base": round(sign_test(w, len(v), base_hit), 4)}


print("=" * 104)
print("A. *** THE GATE ATTRIBUTION ***  The session AFTER a CPI print, TLT.")
print("   Split by whether a PPI ALSO printed that session (the C1 gate).")
print("=" * 104)
rows_all, rows_18 = [], []
cpi_p1 = [p + 1 for p in sorted(cpi_all) if 1 <= p + 1 < N and not np.isnan(d1[p + 1])]
v_all = np.array([d1[q] for q in cpi_p1])
yr_all = np.array([idx[q].year for q in cpi_p1])
has_ppi = np.array([q in ppi_all for q in cpi_p1])
print(pd.DataFrame([
    rep(v_all, "CPI+1 session, ALL (no PPI gate)"),
    rep(v_all[has_ppi], "CPI+1 AND a PPI prints  <-- C1's live cell"),
    rep(v_all[~has_ppi], "CPI+1 and NO PPI  <-- the gate's counterfactual"),
]).to_string(index=False))
m18 = yr_all >= 2018
print("\n  Era-matched (2018+, where 52/55 of the live cell lives):")
print(pd.DataFrame([
    rep(v_all[m18], "CPI+1 2018+, ALL"),
    rep(v_all[m18 & has_ppi], "CPI+1 2018+ AND PPI (live cell)"),
    rep(v_all[m18 & ~has_ppi], "CPI+1 2018+ NO PPI"),
]).to_string(index=False))
a, b = v_all[m18 & has_ppi], v_all[m18 & ~has_ppi]
if len(a) > 1 and len(b) > 1:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"\n  GATE LIFT (2018+) = {100*(a.mean()-b.mean()):+.4f}pp  welch t "
          f"{(a.mean()-b.mean())/se:+.2f}")
print("  -> if the no-PPI arm is comparable, the PPI gate is a filter that")
print("     does not filter and the trade is CPI+1 drift.")

print("\n" + "=" * 104)
print("B. And the mirror: the PPI print session WITHOUT the CPI-eve, era-matched")
print("=" * 104)
ppi_l = [p for p in sorted(ppi_all) if 1 <= p < N and not np.isnan(d1[p])]
v_p = np.array([d1[p] for p in ppi_l])
yr_p = np.array([idx[p].year for p in ppi_l])
ce = np.array([(p - 1) in cpi_all for p in ppi_l])
p18 = yr_p >= 2018
print(pd.DataFrame([
    rep(v_p[p18], "PPI print 2018+, all"),
    rep(v_p[p18 & ce], "PPI print 2018+, CPI on eve (live)"),
    rep(v_p[p18 & ~ce], "PPI print 2018+, no CPI on eve"),
]).to_string(index=False))
print("\n  Pooled 2018+ 'CPI printed yesterday' cell regardless of PPI:")
print(pd.DataFrame([rep(v_all[m18], "CPI+1 2018+ (any)")]).to_string(index=False))

print("\n" + "=" * 104)
print("C. OFFSET LADDER around the PPI print, h=1, entry at close p+k")
print("   (registry: a spike that decays either side is an event; a plateau is")
print("    month position wearing an event label)")
print("=" * 104)
rows = []
for k in range(-5, 4):
    ok = [p for p in ppi_l if 6 <= p + k < N and not np.isnan(d1[p + k])]
    v = np.array([d1[p + k] for p in ok])
    y = np.array([idx[p].year for p in ok])
    ceo = np.array([(p - 1) in cpi_all for p in ok])
    rows.append({"k(session rel. print)": k, "N": len(v),
                 "all_pct": round(100 * v.mean(), 4),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "y2018p_pct": round(100 * v[y >= 2018].mean(), 4),
                 "cpi_eve_pct": round(100 * v[ceo].mean(), 4) if ceo.sum() else np.nan,
                 "cpi_eve_N": int(ceo.sum())})
print(pd.DataFrame(rows).to_string(index=False))
print(f"  ctrl: TLT unconditional daily {100*np.nanmean(d1):+.4f}%")

print("\n" + "=" * 104)
print("D. CPI-DAY SIGN CONDITIONER (is the live cell a reversal of the CPI move?)")
print("=" * 104)
live = [p for p in ppi_l if (p - 1) in cpi_all and p >= 2]
v = np.array([d1[p] for p in live])
prev = np.array([d1[p - 1] for p in live])
y = np.array([idx[p].year for p in live])
print(pd.DataFrame([
    rep(v[prev < 0], "CPI day was DOWN for TLT -> print day"),
    rep(v[prev >= 0], "CPI day was UP for TLT -> print day"),
]).to_string(index=False))
print(f"  corr(CPI-day move, print-day move) = {np.corrcoef(prev, v)[0,1]:+.3f}")
print(f"  TLT's CPI-day (2026-08-12) move is the live input; today's is unknown")
print("  at write time, so a sign conditioner is NOT executable at tonight's MOC")
print("  unless it is read off the tape at 15:59.")

print("\n" + "=" * 104)
print("E. CONCENTRATION of the live cell (55 obs, no declustering needed at h=1")
print("   since PPI prints are ~21td apart -- but check the YEAR concentration)")
print("=" * 104)
ld = pd.DatetimeIndex([idx[p] for p in live])
print("  " + cluster_note(ld, v, k=3))
byyr = pd.Series(v).groupby(y).sum() * 100
print("\n  per-year TOTAL pp contribution:")
print("  " + byyr.round(2).to_string().replace("\n", "\n  "))
tot = v.sum()
print(f"\n  drop the single best year: {100*(tot - byyr.max()/100):+.3f}pp total "
      f"over {len(v)-int((byyr.idxmax()==y).sum())} obs -> mean "
      f"{100*(tot - byyr.max()/100)/(len(v)-int((byyr.idxmax()==y).sum())):+.4f}%")
srt = np.sort(v)
print(f"  drop best 2 obs: mean {100*srt[:-2].mean():+.4f}%  "
      f"drop worst 2: {100*srt[2:].mean():+.4f}%")

print("\n" + "=" * 104)
print("F. FOMC contamination + midterm inside the live cell")
print("=" * 104)
fo = np.array([bool({p, p - 1, p + 1} & fomc_all) for p in live])
mid = (y % 4) == 2
print(pd.DataFrame([
    rep(v[~fo], "live cell, no FOMC within +/-1"),
    rep(v[fo], "live cell, FOMC adjacent"),
    rep(v[mid], "live cell, MIDTERM years (today)"),
    rep(v[~mid], "live cell, non-midterm"),
    rep(v[mid & (y >= 2018)], "live cell, midterm AND 2018+"),
]).to_string(index=False))
print(f"\n  midterm live-cell dates: "
      f"{', '.join(str(d.date()) for d in ld[mid])}")

print("\n" + "=" * 104)
print("G. AUGUST inside the live cell + the parent's month-shape problem")
print("=" * 104)
mo = ld.month.values
print(pd.DataFrame([rep(v[mo == 8], "live cell, AUGUST prints (today)"),
                    rep(v[mo != 8], "live cell, other months")]).to_string(index=False))
mo_p = pd.DatetimeIndex([idx[p] for p in ppi_l]).month.values
print(pd.DataFrame([rep(v_p[mo_p == 8], "PARENT, AUGUST prints"),
                    rep(v_p[mo_p != 8], "PARENT, other months")]).to_string(index=False))
print(f"\n  August live-cell dates: {', '.join(str(d.date()) for d in ld[mo == 8])}")
