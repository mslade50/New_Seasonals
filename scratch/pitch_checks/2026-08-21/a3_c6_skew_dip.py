"""C6 round 1: long SPY on a ^SKEW top-pole spike inside a five-day dip.

Watchlist (2026-08-12): pct_rank(^SKEW,5) >= 95, NO vix condition, h=5,
+0.372% excess over 185 episodes, 65.4% hit, sign p 0.026. Arming legs:
(i) SPY > 1% below its 52w high -- LIVE today at -1.96%, stated N=112,
+0.633% excess, sign p 0.014; (ii) non-midterm year -- DEAD, 2026 is midterm,
live intersection stated N=20 at -0.174%.

The entry's own caveat is the first test: the dip bucket (SPY 5d <= -1%)
carries +1.491% excess and plain dip-buying with no skew condition already
pays +0.219% over N=511. So do the gate attribution on the SKEW leg with the
dip held FIXED, then decide whether the midterm leg is a conditioner or a
20-observation slice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5

px = close_panel(["SPY", "^SKEW", "^VIX"]).loc[:ASOF]
px = px.dropna(subset=["SPY", "^SKEW"])
idx = px.index
sk5 = pct_rank(px["^SKEW"], 5)
sk63 = pct_rank(px["^SKEW"], 63)
spy5r = _valid_pct_change(px["SPY"], 5)
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
spy_dd = px["SPY"] / spy_hi - 1.0
vix21 = pct_rank(px["^VIX"], 21)
midterm = pd.Series([d.year % 4 == 2 for d in idx], index=idx)

print("=" * 100)
print("C6-0  LIVE STATE", ASOF.date())
print("=" * 100)
print(f"  ^SKEW {px['^SKEW'].loc[ASOF]:.2f}  5d rank {sk5.loc[ASOF]:.1f}  63d rank {sk63.loc[ASOF]:.1f}")
print(f"  SPY 5d ret {100*spy5r.loc[ASOF]:.2f}%   dist 52wh {100*spy_dd.loc[ASOF]:.2f}%")
print(f"  VIX {px['^VIX'].loc[ASOF]:.2f}  21d rank {vix21.loc[ASOF]:.1f}   midterm={bool(midterm.loc[ASOF])}")

leg = fwd_lag(px["SPY"], H, 1)
base = leg.dropna()
skew = (sk5 >= 95).fillna(False)
dip = (spy5r <= -0.01).fillna(False)
below = (spy_dd < -0.01).fillna(False)


def ep(m, gap=5):
    return declusters(idx[(m & leg.notna()).values], gap, idx)


def row(lbl, m, gap=5):
    e = ep(m, gap)
    if len(e) < 3:
        return {"label": lbl, "n": len(e)}
    v = leg.loc[e].values
    r = summarize(v, lbl)
    r["excess_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    r["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    return r


print("\n" + "=" * 100)
print("C6-1  REPRODUCE the watchlist numbers (h=5, min_gap 5)")
print("=" * 100)
show([row("skew rank5>=95 ALONE", skew),
      row("skew>=95 & SPY >1% below 52wh", skew & below),
      row("skew>=95 & below & NON-midterm", skew & below & ~midterm),
      row("skew>=95 & below & MIDTERM (live)", skew & below & midterm)],
     "watchlist reproduction")
print(f"  SPY all-days h=5 drift = {100*base.mean():+.3f}%  (N={len(base)})")

print("\n" + "=" * 100)
print("C6-2  GATE ATTRIBUTION -- hold the DIP fixed, add the skew leg")
print("=" * 100)
show([row("DIP alone: SPY 5d <= -1%", dip),
      row("DIP + skew rank5>=95", dip & skew),
      row("DIP + skew rank5<95 (complement)", dip & ~skew),
      row("SKEW alone", skew),
      row("neither (all days)", pd.Series(True, index=idx))],
     "dip-vs-skew attribution, h=5 episodes")
a = leg.loc[ep(dip & skew)].values
b = leg.loc[ep(dip)].values
c = leg.loc[ep(dip & ~skew)].values
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  skew increment over DIP-ALONE:    {100*(a.mean()-b.mean()):+.3f}pp  welch t {(a.mean()-b.mean())/se:+.2f}")
se2 = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
print(f"  skew increment over DIP-NO-SKEW:  {100*(a.mean()-c.mean()):+.3f}pp  welch t {(a.mean()-c.mean())/se2:+.2f}")

print("\n" + "=" * 100)
print("C6-3  TODAY'S FULL STATE: dip + skew + midterm")
print("=" * 100)
show([row("dip & skew, ALL years", dip & skew),
      row("dip & skew, NON-midterm", dip & skew & ~midterm),
      row("dip & skew, MIDTERM  <- TODAY", dip & skew & midterm),
      row("dip only, MIDTERM", dip & midterm),
      row("dip only, NON-midterm", dip & ~midterm)],
     "midterm split")

print("\n" + "=" * 100)
print("C6-4  IS THE MIDTERM LEG A CONDITIONER OR A SLICE?  same split on the")
print("      parent (skew alone) and on plain dip-buying")
print("=" * 100)
for lbl, m in (("skew alone", skew), ("dip alone", dip), ("all days", pd.Series(True, index=idx))):
    e1, e2 = ep(m & ~midterm), ep(m & midterm)
    v1, v2 = leg.loc[e1].values, leg.loc[e2].values
    se3 = np.sqrt(v1.var(ddof=1) / len(v1) + v2.var(ddof=1) / len(v2))
    print(f"  {lbl:<12} non-mid N={len(v1):<4} {100*v1.mean():+7.3f}%   "
          f"midterm N={len(v2):<4} {100*v2.mean():+7.3f}%   diff {100*(v2.mean()-v1.mean()):+7.3f}pp "
          f"welch t {(v2.mean()-v1.mean())/se3:+.2f}")

print("\n" + "=" * 100)
print("C6-5  BATTERY on today's live cell (dip + skew, all years)")
print("=" * 100)
variants = {}
for s in (90, 93, 95, 97):
    variants[f"skew rank5>={s} & dip"] = ((sk5 >= s) & dip).fillna(False)
for dp in (-0.005, -0.01, -0.02, -0.03):
    variants[f"skew>=95 & SPY5d<={100*dp:.1f}%"] = (skew & (spy5r <= dp)).fillna(False)
battery(px, dip & skew, [("SPY", 1.0)], H,
        "C6  long SPY | skew rank5>=95 AND SPY 5d <= -1%", cost_bps=2.0,
        variants=variants, min_gap=5, event_kinds=("cpi", "ppi", "nfp"))

print("\n" + "=" * 100)
print("C6-6  HORIZON SCAN on dip+skew")
print("=" * 100)
show(horizon_scan(px, ep(dip & skew), [("SPY", 1.0)], hs=(1, 2, 3, 5, 10), min_gap=5),
     "dip + skew episodes")
show(horizon_scan(px, ep(dip), [("SPY", 1.0)], hs=(1, 2, 3, 5, 10), min_gap=5),
     "dip alone (the control)")
