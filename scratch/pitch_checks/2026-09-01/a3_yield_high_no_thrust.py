"""A3 -- a trailing-252 yield MAXIMUM reached with NO thrust.

The claim is that this is a never-measured object because every prior yield-high
cell in this repo carried a rate thrust.  So the question is not "does the
no-thrust cell pay" but "is the no-thrust cell a DIFFERENT ANIMAL from the
with-thrust cell".  If the two halves are indistinguishable the object does not
exist and the candidate is a re-slice of A1's support.

Order of operations:
  1. COUNT FIRST, under BOTH percentile conventions (2026-08-31 trap):
     inclusive-self rolling max vs exclusive-self w[:-1] <= w[-1].
  2. Thrust ladder: 21d yield change <= 5 / 10 / 20 bp, and the complement.
  3. Measure TLT / IEF / FLAT / SPY, against three controls.
  4. The DIFFERENCE test: no-thrust vs with-thrust, Welch, on episodes.
  5. Standing traps: lag profile, decluster ladder, era, concentration by value.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ASOF = pd.Timestamp("2026-08-31")

px = close_panel(["^TNX", "TLT", "IEF", "SPY"]).dropna(how="any")
px = px[px.index <= ASOF]
idx = px.index
tnx = px["^TNX"]

hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL_INCL = off_hi >= -0.0025                       # convention 1: within 0.25% of max
# convention 2: strict exclusive-self running max (today > every one of the prior 251)
prev_max = rolling_on_valid(tnx.shift(1), lambda x: x.rolling(251).max())
LEVEL_EXCL = tnx > prev_max

chg21 = (tnx - tnx.shift(21)) * 100.0
chg63 = (tnx - tnx.shift(63)) * 100.0

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
VEH = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT, "SPY": [("SPY", 1.0)]}

POS = {dd: i for i, dd in enumerate(idx)}


def fast_decluster(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


print("=" * 110)
print("1. COUNT FIRST, both percentile conventions")
print("=" * 110)
print("  TODAY: off-high %+.5f%% | 21d chg %+.1f bp | 63d chg %+.1f bp"
      % (100 * off_hi.iloc[-1], chg21.iloc[-1], chg63.iloc[-1]))
print("  today fires LEVEL_INCL (within 0.25%% of max): %s" % bool(LEVEL_INCL.iloc[-1]))
print("  today fires LEVEL_EXCL (strictly above all 251 priors): %s" % bool(LEVEL_EXCL.iloc[-1]))
for lab, L in (("INCL (off-high >= -0.25%)", LEVEL_INCL), ("EXCL (new running max)", LEVEL_EXCL)):
    nd = int(L.fillna(False).sum())
    print("\n  %-28s day count %d" % (lab, nd))
    for thr in (5, 10, 20, 30, 1e9):
        m = L.fillna(False) & (chg21 <= thr)
        days = idx[m.reindex(idx, fill_value=False).values]
        for gap in (10, 21):
            ep = fast_decluster(days, gap)
            print("     21d chg <= %6s bp : %4d days | episodes@gap%d %3d"
                  % ("inf" if thr > 1e8 else int(thr), len(days), gap, len(ep)))
            if gap == 10 and len(ep):
                print("        episode dates:", ", ".join(str(x.date()) for x in ep[:20])
                      + (" ..." if len(ep) > 20 else ""))
            if gap == 21:
                break

print("\n" + "=" * 110)
print("2/3/4. NO-THRUST vs WITH-THRUST -- is it a different animal?")
print("   (INCL convention, h=5 and h=10, entry lag=1 MOC)")
print("=" * 110)
for thr in (5, 10, 20):
    print("\n  ############ thrust cut at 21d chg <= %d bp ############" % thr)
    NO = LEVEL_INCL.fillna(False) & (chg21 <= thr)
    YES = LEVEL_INCL.fillna(False) & (chg21 > thr)
    for h in (5, 10):
        for vk, legs in VEH.items():
            ret = vehicle_ret(px, legs, h, 1)
            valid = ret.notna()
            out = []
            for lab, M in (("NO-thrust", NO), ("WITH-thrust", YES)):
                sig = idx[M.reindex(idx, fill_value=False).values & valid.values]
                ep = fast_decluster(sig, max(h, 10))
                if len(ep) == 0:
                    out.append((lab, 0, np.nan, np.nan, None))
                    continue
                v = ret.loc[ep].values
                out.append((lab, len(v), 100 * v.mean(), 100 * (v > 0).mean(), v))
            a, b = out[0][4], out[1][4]
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                diff = wt = np.nan
            else:
                se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                diff, wt = 100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se
            base = 100 * float(ret.dropna().mean())
            sp = ("  h=%2d %-5s | NO N=%2d %+.3f%% hit %.0f%% | WITH N=%2d %+.3f%% hit %.0f%%"
                  " | diff %+.3fpp welch t %+.2f | all-days %+.3f%%"
                  % (h, vk, out[0][1], out[0][2], out[0][3],
                     out[1][1], out[1][2], out[1][3], diff, wt, base))
            if a is not None and len(a) >= 2:
                w = int((a > 0).sum())
                sp += " | NO sign p %.4f" % sign_test(w, len(a))
            print(sp)

print("\n" + "=" * 110)
print("5. STANDING TRAPS on the best-looking no-thrust cell (cut 20 bp, both h)")
print("=" * 110)
NO = LEVEL_INCL.fillna(False) & (chg21 <= 20)
for vk, legs in VEH.items():
    for h in (5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        sig = idx[NO.reindex(idx, fill_value=False).values & valid.values]
        if len(sig) < 5:
            continue
        print("\n  --- %s h=%d ---" % (vk, h))
        for lag in (0, 1, 2):
            r2 = vehicle_ret(px, legs, h, lag)
            s2 = idx[NO.reindex(idx, fill_value=False).values & r2.notna().values]
            ep = fast_decluster(s2, max(h, 10))
            v = r2.loc[ep].values
            print("    lag=%d N=%2d mean %+.4f%% t %+.2f hit %.0f%%"
                  % (lag, len(v), 100 * v.mean(),
                     v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 100 * (v > 0).mean()))
        for gap in (5, 10, 21, 42):
            ep = fast_decluster(sig, max(h, gap))
            v = ret.loc[ep].values
            print("    gap=%2d N=%2d mean %+.4f%% t %+.2f" % (gap, len(v), 100 * v.mean(),
                  v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan))
        ep = fast_decluster(sig, max(h, 10))
        v = ret.loc[ep].values
        show(era_split(ep, v), "    era")
        print("    year histogram:",
              dict(pd.Series(pd.DatetimeIndex(ep).year).value_counts().sort_index()))
        srt = np.sort(v)
        print("    concentration BY VALUE: mean %+.4f%% | drop top-2 %+.4f%% | drop worst-2 %+.4f%%"
              % (100 * v.mean(), 100 * srt[:-2].mean(), 100 * srt[2:].mean()))
        loc = local_control(idx[valid.values], pd.DatetimeIndex(sig))
        print("    CTRL local +/-126td %+.4f%% | all days %+.4f%%"
              % (100 * ret.loc[loc].mean(), 100 * ret[valid].mean()))
        print("    cost: %.1f bps -> %.2fx at 3 bps/leg x %d legs"
              % (100 * 100 * v.mean(), abs(100 * 100 * v.mean()) / (3.0 * len(legs)), len(legs)))
