"""C5 round 2c -- the deciding test.

Round 2b left the duration-neutral curve position (long IEF / short 0.52 TLT,
which profits when the long end UNDERPERFORMS the belly on a duration-adjusted
basis, i.e. a bear-STEEPENER despite the label I gave it in 2b) looking real:
rotation P(grid max|t| >= 3.41) = 0.018, gate adds +0.179pp over the rising-
regime it lives in, era stable, midterm better, concentration -3%.

Two things can still kill it and both are substantive:
  (A) COST.  Best rung is h=10 at +0.221% = 22.1 bps against 6 bps round trip
      = 3.68x, under the 5x bar.  Walk the whole ladder and check whether ANY
      whole-variant clears it.
  (B) SUPPORT.  Today sits at the 1st percentile of the trigger's own 21d
      yield-change distribution (+0.035pt vs a trigger median +0.329pt) and
      the 87.6th percentile of the FULL-history yield level against a trigger
      median of 44.3.  If the edge is CONDITIONAL on the thrust or on the
      level, today's state buys none of it -- and the candidate's whole claim
      was that today is NOT a thrust.
This script conditions the cell on both and reads the fitted value at today's
numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

raw = close_panel(["^TNX", "TLT", "IEF"])
px = raw.dropna(how="any")
idx = px.index
tnx = px["^TNX"]
off_hi = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
LEVEL = off_hi >= -0.0025
chg21 = tnx - tnx.shift(21)
rank21 = pct_rank(tnx, 21, 252)
tnx_full = raw["^TNX"].dropna()
lvl_full = pd.Series([100.0 * (tnx_full.iloc[:i + 1] <= tnx_full.iloc[i]).mean()
                      for i in range(len(tnx_full))], index=tnx_full.index).reindex(idx)
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
POS = [("IEF", 1.0), ("TLT", -1.0 / BETA)]

TODAY_CHG21, TODAY_RANK21, TODAY_LVL = chg21.iloc[-1], rank21.iloc[-1], lvl_full.iloc[-1]
print("today: chg21 %+.3f pt | rank21 %.1f | full-history level pctile %.1f"
      % (TODAY_CHG21, TODAY_RANK21, TODAY_LVL))


def epis(h, mask=LEVEL):
    ret = vehicle_ret(px, POS, h, 1)
    sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool)
              & ret.notna().values]
    e = declusters(sig, max(h, 10), idx)
    return ret, e, ret.loc[e].values


# ======================================================= A. THE COST LADDER
print("\n" + "=" * 110)
print("A. COST.  2 legs x ~3 bps = 6 bps round trip.  Bar is 5x = 30 bps.")
print("=" * 110)
rows = []
for h in range(1, 11):
    ret, e, v = epis(h)
    bps = 100 * 100 * v.mean()
    rows.append({"h": h, "n_epi": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "bps": round(bps, 1), "x_cost": round(bps / 6.0, 2),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                 "clears_5x": bps / 6.0 >= 5.0})
print(pd.DataFrame(rows).to_string(index=False))
print("  -> best rung h=%d at %.2fx. The 5x bar needs %.1f bps; the ladder tops out at "
      "%.1f bps." % (max(rows, key=lambda r: r["x_cost"])["h"],
                     max(r["x_cost"] for r in rows), 30.0,
                     max(r["bps"] for r in rows)))
print("  A single-leg variant is not available: IEF alone at h=10 is the cell round 1")
print("  measured at +0.281%% (28.1 bps / 3 bps = 9.4x) -- but that is the C10 vehicle,")
print("  and its edge over its OWN control is only +0.143pp = 14.3 bps = 4.8x. Checked:")
ret_i = vehicle_ret(px, [("IEF", 1.0)], 10, 1)
sig = idx[LEVEL.values & ret_i.notna().values]
e = declusters(sig, 10, idx)
vi = ret_i.loc[e].values
print("     IEF-alone h=10 episodes %+.3f%%  ctrl-b all days %+.3f%%  EDGE %+.3f%% "
      "= %.1f bps = %.2fx cost" % (100 * vi.mean(), 100 * ret_i.dropna().mean(),
                                   100 * (vi.mean() - ret_i.dropna().mean()),
                                   100 * 100 * (vi.mean() - ret_i.dropna().mean()),
                                   100 * 100 * (vi.mean() - ret_i.dropna().mean()) / 3.0))

# =============================================== B. SUPPORT: THRUST CONDITIONING
print("\n" + "=" * 110)
print("B1. IS THE EDGE CONDITIONAL ON THE THRUST?  Split the trigger episodes by")
print("    the 21-session yield change.  Today is +0.035pt = the 1st percentile.")
print("=" * 110)
for h in (2, 5, 10):
    ret, e, v = epis(h)
    c = chg21.reindex(e).values
    rows = []
    for lo, hi in [(-9, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 9)]:
        m = (c > lo) & (c <= hi)
        r = summarize(v[m], "h=%d chg21 in (%.2f,%.2f]" % (h, lo, hi))
        rows.append(r)
    show(rows, f"B1.{h} thrust buckets (today's bucket is the FIRST, <=+0.15pt)")
    # linear read at today's value
    ok = ~np.isnan(c)
    if ok.sum() > 5:
        b, a = np.polyfit(c[ok], v[ok], 1)
        print("   OLS  fwd = %+.4f + %+.4f * chg21   -> fitted at today's %+.3f pt = "
              "%+.3f%% (%.1f bps, %.2fx cost)"
              % (100 * a, 100 * b, TODAY_CHG21, 100 * (a + b * TODAY_CHG21),
                 100 * 100 * (a + b * TODAY_CHG21),
                 100 * 100 * (a + b * TODAY_CHG21) / 6.0))
        print("   corr(chg21, fwd) = %+.3f over N=%d episodes"
              % (np.corrcoef(c[ok], v[ok])[0, 1], ok.sum()))

print("\n" + "=" * 110)
print("B2. AND ON THE 21d RETURN RANK?  Today is 49.2; trigger median is 86.5.")
print("=" * 110)
for h in (2, 5, 10):
    ret, e, v = epis(h)
    rk = rank21.reindex(e).values
    rows = []
    for lo, hi in [(0, 60), (60, 80), (80, 92), (92, 101)]:
        m = (rk >= lo) & (rk < hi)
        rows.append(summarize(v[m], "h=%d rank21 [%d,%d)" % (h, lo, hi)))
    show(rows, f"B2.{h} return-rank buckets (today's bucket is the FIRST, <60)")

print("\n" + "=" * 110)
print("B3. AND ON THE ABSOLUTE YIELD LEVEL?  Today's full-history level pctile is")
print("    87.6; the trigger population's median is 44.3.  (the ^SKEW-kill test)")
print("=" * 110)
for h in (2, 10):
    ret, e, v = epis(h)
    lv = lvl_full.reindex(e).values
    rows = []
    for lo, hi in [(0, 30), (30, 50), (50, 70), (70, 101)]:
        m = (lv >= lo) & (lv < hi)
        rows.append(summarize(v[m], "h=%d full-hist level pctile [%d,%d)" % (h, lo, hi)))
    show(rows, f"B3.{h} level buckets (today's bucket is the LAST, >=70)")

# ============================================ B4. the joint live-analogue cell
print("\n" + "=" * 110)
print("B4. THE ACTUAL LIVE ANALOGUE: trigger days that were ALSO a slow grind")
print("    (chg21 <= +0.15pt) -- the candidate's own framing.")
print("=" * 110)
SLOW = LEVEL & (chg21 <= 0.15)
print("  population: %d days" % int(SLOW.sum()))
print("  dates:", ", ".join(str(d.date()) for d in idx[SLOW.values]))
for h in (1, 2, 3, 5, 10):
    ret, e, v = epis(h, SLOW)
    if len(v) == 0:
        print("  h=%d: N=0" % h); continue
    w = int((v > 0).sum())
    print("  h=%2d  N_epi=%2d  mean %+.3f%%  hit %.0f%%  sign p %.4f  = %.1f bps = %.2fx"
          % (h, len(v), 100 * v.mean(), 100 * (v > 0).mean(), sign_test(w, len(v)),
             100 * 100 * v.mean(), 100 * 100 * v.mean() / 6.0))
