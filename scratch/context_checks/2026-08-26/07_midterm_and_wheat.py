"""Last checks before writing.

(a) The midterm qualifier on the bond leg, since it guts the equity leg and
    2026 is a midterm year. If bonds are also midterm-dead the brief says so.
(b) Wheat: drill 01 showed +9.15% was mostly real trading (+6.25% intraday,
    closing on the high) unlike corn's mostly-gap move. Is a 52-week high on a
    2-ATR up session worth anything for wheat?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, load_prices, summarize, sign_test, cluster_note

# ------------------------------------------------------------------- (a)
bp = close_panel(["TLT", "IEF", "SPY", "IWM"]).sort_index()
ad = bp.index
pos_of = pd.Series(np.arange(len(ad)), index=ad)
last_pos = pd.Series(pos_of.values, index=ad).groupby(ad.to_period("M")).transform("max")
dist = pd.Series(last_pos.values - pos_of.values, index=ad)
CUR = ad.max().to_period("M")
A = ad[((dist == 3).values) & (ad.month == 8)]
A = A[A.to_period("M") != CUR]


def fwd(ser, dates, h):
    p = ser.index.searchsorted(dates)
    ok = (p + h < len(ser)) & (p < len(ser))
    p = p[ok]
    return ser.index[p], (ser.values[p + h] / ser.values[p]) - 1.0


print("=" * 76)
print("(a) midterm vs non-midterm, anchored on today's slot")
print("=" * 76)
for sym in ["SPY", "IWM", "TLT", "IEF"]:
    ser = bp[sym].dropna()
    print(f"\n--- {sym} ---")
    for h, lbl in [(1, "tomorrow"), (2, "thru Fri"), (3, "thru Aug 31")]:
        d, v = fwd(ser, A, h)
        mid = (d.year % 4 == 2)
        for mlbl, m in [("midterm", mid), ("non-mid", ~mid)]:
            if m.sum() < 3:
                continue
            s = summarize(v[m], "")
            up = int((v[m] > 0).sum())
            print(f"  h={h} ({lbl:11s}) {mlbl:8s} n={int(m.sum()):3d}  "
                  f"mean={s['mean_pct']:+6.3f}%  {up}-{int(m.sum())-up} up  "
                  f"signp={sign_test(up, int(m.sum())):.4f}")

# ------------------------------------------------------------------- (b)
print("\n" + "=" * 76)
print("(b) Wheat: 52-week high on a >=2 ATR up session")
print("=" * 76)
d = load_prices(["ZW=F"])["ZW=F"].sort_index()
c, h_, l_ = d["Close"], d["High"], d["Low"]
pc = c.shift(1)
tr = pd.concat([h_ - l_, (h_ - pc).abs(), (l_ - pc).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()          # Wilder-14
chg = c - pc
is_52wh = c >= c.rolling(252).max()
big_up = chg >= 2 * atr.shift(1)
trig = c.index[(is_52wh & big_up).fillna(False).values]
print(f"  history {c.index.min().date()} .. {c.index.max().date()}")
print(f"  today: 52w high={bool(is_52wh.iloc[-1])}  "
      f"session move in ATR={(chg.iloc[-1] / atr.shift(1).iloc[-1]):.2f}")
print(f"  prior episodes: {len(trig)}")
for hh in (1, 3, 5, 10, 21):
    p = c.index.searchsorted(trig)
    ok = p + hh < len(c)
    v = (c.values[p[ok] + hh] / c.values[p[ok]]) - 1.0
    if len(v) < 4:
        continue
    s = summarize(v, "")
    up = int((v > 0).sum())
    print(f"    h={hh:<3d} n={len(v):3d}  mean={s['mean_pct']:+6.3f}%  "
          f"{up}-{len(v)-up} up  t={s['t']:+5.2f}  signp={sign_test(up, len(v)):.4f}")
p = c.index.searchsorted(trig)
ok = p + 1 < len(c)
v1 = (c.values[p[ok] + 1] / c.values[p[ok]]) - 1.0
print(f"  concentration: {cluster_note(c.index[p[ok]], v1, k=2)}")
sa = c.pct_change().shift(-1).dropna()
print(f"  all-days control n={len(sa)}  mean={sa.mean()*100:+6.3f}%  "
      f"hit={(sa > 0).mean():.1%}")
print(f"  trigger dates: {[str(x.date()) for x in trig][-12:]}")
