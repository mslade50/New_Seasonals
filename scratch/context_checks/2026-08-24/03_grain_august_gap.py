"""How singular is a >3% August overnight gap in corn and wheat, and does today
count as one? Also: verify the CT=F bar integrity flagged by drill 01.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test, fwd_ret, declusters  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["ZC=F", "ZW=F", "ZS=F", "CT=F"])

print("=== CT=F bar integrity, last 5 sessions ===")
d = px["CT=F"].loc["2026-08-18":, ["Open", "High", "Low", "Close"]]
d = d.assign(close_gt_high=d["Close"] > d["High"], open_zero=d["Open"] <= 0)
print(d.round(2).to_string())
bad = px["CT=F"][(px["CT=F"]["Close"] > px["CT=F"]["High"]) |
                 (px["CT=F"]["Open"] <= 0)]
print(f"corrupt CT=F bars in full history: {len(bad)}  dates: {[str(x.date()) for x in bad.index[-6:]]}")

print()
for t in ["ZC=F", "ZW=F", "ZS=F"]:
    df = px[t]
    gap = (df["Open"] / df["Close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan).dropna()
    aug = gap[gap.index.month == 8]
    big_aug = aug[aug > 0.03]
    print(f"{t}: August sessions {len(aug)}, overnight gap > +3%: {len(big_aug)} "
          f"-> {[f'{x.date()} {v*100:.2f}%' for x, v in big_aug.items()]}")
    allbig = gap[gap > 0.03]
    print(f"   full-year gap > +3%: n={len(allbig)}, by month "
          f"{dict(pd.Series(allbig.index.month).value_counts().sort_index())}")

print()
print("=== corn: sessions closing at a 252d high with a 5d return of 10%+ ===")
df = px["ZC=F"]
c = df["Close"]
hi252 = c.rolling(252).max()
r5 = c.pct_change(5)
mask = (c >= hi252 * 0.9999) & (r5 >= 0.10)
mask = mask & (df.index <= ASOF)
dts = df.index[mask]
dts_dc = declusters(dts, 10)
print(f"raw n={len(dts)}  declustered@10td n={len(dts_dc)}")
print("dates:", [str(x.date()) for x in dts_dc])
for h in (1, 5, 10, 21):
    f = fwd_ret(c, h)
    v = f.reindex(dts_dc).dropna().values
    s = summarize(v, f"h{h}")
    up = int((v > 0).sum())
    print(f"  h{h:<3} n={s['n']:<3} mean {s['mean']:>7.2f}%  med {s['median']:>7.2f}%  "
          f"hit {s['hit']:>5.1f}%  ({up}-{len(v)-up} up)  sign p {sign_test(up, len(v)):.4f}")

# control: all corn days
print()
for h in (1, 5, 10, 21):
    f = fwd_ret(c, h).dropna()
    s = summarize(f.values, "")
    print(f"  control all corn days h{h:<3} n={s['n']:<5} mean {s['mean']:>7.2f}%  hit {s['hit']:.1f}%")
