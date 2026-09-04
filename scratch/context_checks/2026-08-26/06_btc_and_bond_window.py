"""Two leftovers.

(1) BTC z10 >= 2. The only price-lane cell tonight that is not a futures roll,
a revised bar or a corrupt print: spot crypto, n=296, era-stable, BH pass, and
the direction is continuation rather than the reversal the setup suggests.
Today's z10 is 3.13, so check whether the deeper tail behaves like the cell.

(2) The bond leg of month end, anchored on TODAY rather than on tomorrow, so
the horizon actually covers the rest of August. Drill 04 measured the right
effect at the wrong offset.

Convention: lag=0 close-to-close from the anchor close, h=1 is the next
session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, summarize, sign_test, cluster_note

# ---------------------------------------------------------------- part 1: BTC
px = close_panel(["BTC-USD", "ETH-USD"]).sort_index()

def z10(s):
    r10 = s.pct_change(10)
    vol = s.pct_change().rolling(21).std() * np.sqrt(10)
    return r10 / vol

def fwd_from(s, dates, h):
    p = s.index.searchsorted(dates)
    ok = (p + h < len(s)) & (p < len(s))
    p = p[ok]
    return s.index[p], (s.values[p + h] / s.values[p]) - 1.0

print("=" * 78)
print("1. BTC-USD, z10 >= 2 (today's reading 3.13)")
print("=" * 78)
s = px["BTC-USD"].dropna()
z = z10(s).dropna()
print(f"  history {s.index.min().date()} .. {s.index.max().date()}  n={len(s)}")
print(f"  today's z10: {z.iloc[-1]:.2f}   (bar {z.index[-1].date()})")

for lo, hi, lbl in [(2.0, 99, "z10 >= 2.0"),
                    (2.0, 3.0, "z10 2.0-3.0"),
                    (3.0, 99, "z10 >= 3.0  <-- today")]:
    trig = z[(z >= lo) & (z < hi)].index
    d, v = fwd_from(s, trig, 1)
    if len(v) < 4:
        continue
    sm = summarize(v, "")
    up = int((v > 0).sum())
    print(f"  {lbl:22s} n={len(v):4d}  h1 mean={sm['mean_pct']:+6.3f}%  "
          f"{up}-{len(v)-up} up  t={sm['t']:+5.2f}  signp={sign_test(up, len(v)):.4f}")

trig = z[z >= 3.0].index
print("\n  deep-tail (z10>=3) horizon path:")
for h in (1, 2, 3, 5, 10, 21):
    d, v = fwd_from(s, trig, h)
    sm = summarize(v, "")
    up = int((v > 0).sum())
    print(f"    h={h:<3d} n={len(v):4d}  mean={sm['mean_pct']:+7.3f}%  "
          f"{up}-{len(v)-up} up  t={sm['t']:+5.2f}")

d, v = fwd_from(s, trig, 1)
print(f"\n  concentration: {cluster_note(d, v, k=2)}")
pre = d < pd.Timestamp("2018-01-01")
for lbl, m in [("pre-2018", pre), ("2018+", ~pre)]:
    if m.sum() < 4:
        continue
    sm = summarize(v[m], "")
    up = int((v[m] > 0).sum())
    print(f"  {lbl:9s} n={m.sum():4d}  mean={sm['mean_pct']:+6.3f}%  "
          f"{up}-{int(m.sum())-up} up  t={sm['t']:+5.2f}")
sa = s.pct_change().shift(-1).dropna()
print(f"  all-days control  n={len(sa)}  mean={sa.mean()*100:+6.3f}%  "
      f"hit={(sa > 0).mean():.1%}")

# declustered: first z10>=3 after a 10-session gap
gaps = trig.to_series().diff().dt.days.fillna(999)
fresh = trig[(gaps >= 10).values]
d, v = fwd_from(s, fresh, 1)
sm = summarize(v, "")
up = int((v > 0).sum())
print(f"  DECLUSTERED (fresh trigger, 10d gap) n={len(v):3d}  "
      f"mean={sm['mean_pct']:+6.3f}%  {up}-{len(v)-up} up  t={sm['t']:+5.2f}  "
      f"signp={sign_test(up, len(v)):.4f}")

# ------------------------------------------------- part 2: bond month-end window
print("\n" + "=" * 78)
print("2. Bonds from TODAY's slot (4th-to-last August session) to month end")
print("=" * 78)
bp = close_panel(["TLT", "IEF", "SPY"]).sort_index()
ad = bp.index
pos_of = pd.Series(np.arange(len(ad)), index=ad)
last_pos = pd.Series(pos_of.values, index=ad).groupby(ad.to_period("M")).transform("max")
dist = pd.Series(last_pos.values - pos_of.values, index=ad)
CUR = ad.max().to_period("M")
A = ad[((dist == 3).values) & (ad.month == 8)]
A = A[A.to_period("M") != CUR]
print(f"  anchor n={len(A)} years, {A.min().date()} .. {A.max().date()}")
print("  h=1 is tomorrow, h=3 is the August close\n")
for sym in ["TLT", "IEF", "SPY"]:
    ser = bp[sym].dropna()
    print(f"  --- {sym} ---")
    for h in (1, 2, 3):
        d, v = fwd_from(ser, A, h)
        sm = summarize(v, "")
        up = int((v > 0).sum())
        lbl = {1: "tomorrow", 2: "thru Fri", 3: "thru Aug 31"}[h]
        print(f"    h={h} ({lbl:11s}) n={len(v):3d}  mean={sm['mean_pct']:+6.3f}%  "
              f"{up}-{len(v)-up} up  t={sm['t']:+5.2f}  "
              f"signp={sign_test(up, len(v)):.4f}")
