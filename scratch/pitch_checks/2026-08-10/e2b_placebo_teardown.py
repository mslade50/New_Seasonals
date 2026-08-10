"""E2b -- teardown of the three cells round 1 left standing, and the
correlation the hard constraint demands.

Round 1 result, stated before this script runs:
  * The FOMC PLACEBO INVERTED.  The largest overnight premium in the entire
    study is SPY on FOMC day: +13.06 bps over SPY's own unconditional
    overnight, +13.51 bps tdom-matched, hit 64.2%, sign p 0.0000, 13.5x cost.
    FOMC prints at 14:00 ET.  Its overnight segment contains NO print.  The
    stated mechanism ("the print resolves in the opening auction") predicts
    this cell should be the NULL.  It is the biggest number in the table.
  * Dispersion agrees and against the premise: the 08:30 prints raise
    overnight sd by 0-17% (SPY/CPI 1.07x) while the 14:00 print raises
    INTRADAY sd by 10-48% (GLD 1.48x, GDX 1.38x).  The decomposition works;
    the 08:30 events simply do not concentrate their variance in the
    overnight segment for these instruments.  Only TLT-on-NFP does
    (sd 1.49x) and its mean is NEGATIVE (-9.6 bps excess, hit 41.5%).

Three things left to settle, all of them kill-relevant:
  A. nfp x SPY overnight is the only 08:30 cell clearing 5x cost (+5.32 bps
     tdom-matched on a 1 bp round trip).  NFP is a Friday.  Is it the Friday
     overnight?  And has it decayed?
  B. cpi x GDX overnight (+17.5 bps) is the best 08:30 cell by size.  The
     HARD CONSTRAINT says that if the best E2 result is GDX around this CPI,
     it is the morning's survivor in another wrapper.  Quantify.
  C. fomc x SPY overnight -- is it already book property?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

raw = load_prices(["SPY", "GDX"])
ev = load_events()


def segs(t):
    g = raw[t]
    o, c = g["Open"].values.astype(float), g["Close"].values.astype(float)
    n = len(c)
    on = np.full(n, np.nan)
    on[1:] = o[1:] / c[:-1] - 1.0
    return g.index, o, c, on


sidx, so, sc, son = segs("SPY")
gidx, go, gc, gon = segs("GDX")


def pos(idx, kind):
    out = []
    for x in ev[ev.event == kind]["date"]:
        p = int(idx.searchsorted(x, "left"))
        if 1 <= p < len(idx):
            out.append(p)
    return np.array(sorted(set(out)))


print("=" * 100)
print("A. nfp x SPY OVERNIGHT -- is it the FRIDAY overnight, and has it "
      "been arbitraged away?")
print("=" * 100)
p = pos(sidx, "nfp")
v = son[p]
ok = ~np.isnan(v)
p, v = p[ok], v[ok]
dow = np.array([sidx[q].weekday() for q in p])
print(f"  NFP session weekday distribution: "
      f"{dict(pd.Series(dow).value_counts().sort_index())}  (4 = Friday)")
alldow = np.array([d.weekday() for d in sidx])
fri = (alldow == 4) & ~np.isnan(son)
fri_ex = fri.copy()
fri_ex[p] = False
base_all = np.nanmean(son)
base_fri = son[fri_ex].mean()
w = int((v > 0).sum())
print(f"  NFP overnight      N={len(v)}  {1e4*v.mean():+.2f} bps  hit "
      f"{100*w/len(v):.1f}%  sign p {sign_test(w, len(v)):.4f}")
print(f"  CTRL all overnights          {1e4*base_all:+.2f} bps  "
      f"-> excess {1e4*(v.mean()-base_all):+.2f} bps")
print(f"  CTRL FRIDAY overnights only  {1e4*base_fri:+.2f} bps (N="
      f"{int(fri_ex.sum())}) -> DOW-MATCHED excess "
      f"{1e4*(v.mean()-base_fri):+.2f} bps = "
      f"{1e4*(v.mean()-base_fri)/1.0:.1f}x a 1 bp SPY round trip")
# first-Friday control: Fridays in the first week of the month, ex-NFP
tdom_s = pd.Series(sidx.year * 100 + sidx.month, index=sidx)
tdom_s = tdom_s.groupby(tdom_s.values).cumcount().values + 1
fw = fri_ex & (tdom_s <= 6)
print(f"  CTRL first-week FRIDAYS ex-NFP {1e4*son[fw].mean():+.2f} bps "
      f"(N={int(fw.sum())}) -> excess {1e4*(v.mean()-son[fw].mean()):+.2f} bps")
yr = np.array([sidx[q].year for q in p])
print("\n  decay profile (excess over the FRIDAY control), 5-year blocks:")
for lo in range(2000, 2026, 5):
    m = (yr >= lo) & (yr < lo + 5)
    if m.sum() < 5:
        continue
    ww = int((v[m] > 0).sum())
    print(f"    {lo}-{lo+4}: N={int(m.sum()):3d}  {1e4*v[m].mean():+7.2f} bps  "
          f"excess {1e4*(v[m].mean()-base_fri):+7.2f}  hit "
          f"{100*ww/int(m.sum()):5.1f}%")
print(f"\n  concentration: {cluster_note(sidx[p], v)}")
mid = (yr % 4) == 2
print(f"  MIDTERM {1e4*v[mid].mean():+.2f} bps (N={int(mid.sum())}) excess "
      f"{1e4*(v[mid].mean()-base_fri):+.2f}  |  non-midterm "
      f"{1e4*v[~mid].mean():+.2f} (N={int((~mid).sum())})")
print(f"  NEXT NFP: 2026-09-04 -- not today's trade under any form.")

print("\n" + "=" * 100)
print("B. cpi x GDX OVERNIGHT vs THE MORNING'S SURVIVOR (long GDX MOC "
      "2026-08-10 -> MOC 2026-08-17)")
print("=" * 100)
pg = pos(gidx, "cpi")
rows = []
for K in (2, 3):
    pair = []
    for q in pg:
        a = q - K                       # survivor entry close
        if a <= 0 or a + 5 >= len(gidx) or np.isnan(gon[q]):
            continue
        surv = gc[a + 5] / gc[a] - 1.0  # long GDX h=5 from the anchor close
        pair.append((gon[q], surv, gidx[q]))
    if not pair:
        continue
    on_v = np.array([x[0] for x in pair])
    sv = np.array([x[1] for x in pair])
    r = np.corrcoef(on_v, sv)[0, 1]
    # is the overnight INSIDE the survivor's holding window?
    inside = (K >= 1) and (K <= 5)
    rows.append({"survivor_anchor_K": K, "N": len(pair),
                 "overnight_bps": round(1e4 * on_v.mean(), 1),
                 "survivor_h5_pct": round(100 * sv.mean(), 3),
                 "corr(overnight, survivor)": round(r, 3),
                 "overnight_inside_survivor_window": inside,
                 "overnight_share_of_survivor": round(
                     on_v.mean() / sv.mean(), 3) if sv.mean() != 0 else np.nan})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  The CPI print is 2 sessions after the survivor's entry close, so "
      "the CPI-day")
print("  overnight is not merely correlated with the survivor -- it is a "
      "STRICT SUBSET of")
print("  the survivor's own exposure: same ticker, same direction, inside "
      "the same hold.")
print("  It cannot be pitched as a second, uncorrelated idea.")

print("\n" + "=" * 100)
print("C. fomc x SPY OVERNIGHT -- already book property?")
print("=" * 100)
pf = pos(sidx, "fomc_decision")
v = son[pf]
ok = ~np.isnan(v)
pf, v = pf[ok], v[ok]
yr = np.array([sidx[q].year for q in pf])
w = int((v > 0).sum())
print(f"  FOMC-day overnight (MOC decision-1 -> MOO decision) N={len(v)} "
      f"{1e4*v.mean():+.2f} bps, hit {100*w/len(v):.1f}%, sign p "
      f"{sign_test(w, len(v)):.4f}")
print("  The event sleeve's LIVE T1 FOMC_DRIFT is: long SPY 25% NAV, MOC "
      "4 sessions before")
print("  the decision -> MOO the decision-day open.  That trade CONTAINS "
      "this overnight.")
rows = []
for K in (1, 2, 3, 4):
    vals = []
    for q in pf:
        a = q - K
        if a <= 0:
            continue
        vals.append(so[q] / sc[a] - 1.0)   # MOC at decision-K -> MOO decision
    vals = np.array(vals)
    ww = int((vals > 0).sum())
    rows.append({"entry_sessions_before": K, "N": len(vals),
                 "MOC->MOO_bps": round(1e4 * vals.mean(), 1),
                 "hit": round(100 * ww / len(vals), 1),
                 "sign_p": round(sign_test(ww, len(vals)), 4)})
print(pd.DataFrame(rows).to_string(index=False))
mid = (yr % 4) == 2
print(f"\n  MIDTERM years {1e4*v[mid].mean():+.2f} bps (N={int(mid.sum())})  |"
      f"  non-midterm {1e4*v[~mid].mean():+.2f} bps (N={int((~mid).sum())})")
print("  2026 is a midterm year, which is exactly why the event sleeve's T2 "
      "INVERTS T1 to a")
print("  SHORT in midterm years.  Next FOMC decision is not today in any "
      "case.")
print(f"  concentration: {cluster_note(sidx[pf], v)}")
