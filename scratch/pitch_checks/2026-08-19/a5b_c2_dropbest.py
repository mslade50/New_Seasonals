"""C2 subclass: drop-best-episode, ex-2026, and the exact turn-on number.

a5 showed the top 2 episodes carry 96% of the h=3 subclass total and both are
2026 prints.  This quantifies what is left without them and states the number
that would put the cell on the watchlist as a live trade rather than a
one-episode artifact.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

TK = ["XLV", "XLK", "SPY"]
raw = load_prices(TK)
px = close_panel(TK)
r1 = px.pct_change()
vk = (r1["XLV"] - r1["XLK"]).reindex(px.index)
hi252 = px["SPY"].rolling(252).max()
dist = px["SPY"] / hi252 - 1.0
spy_atrp = (pd.Series(wilder_atr(raw["SPY"]["High"], raw["SPY"]["Low"],
                                 raw["SPY"]["Close"], 14),
                      index=raw["SPY"].index).reindex(px.index) / px["SPY"])
SUB = ((vk >= 0.030) & (dist > -0.03) & (spy_atrp < 0.012)).fillna(False)

for h in (1, 2, 3, 5, 7, 10):
    ret = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
    valid = ret.dropna().index
    e = declusters(px.index[SUB.values].intersection(valid), h, valid)
    v = ret.loc[e].values
    o = np.argsort(-v)
    d1 = np.delete(v, o[0])
    d2 = np.delete(v, o[:2])
    m26 = np.array([d.year == 2026 for d in e])
    base = ret.loc[valid].mean()
    print(f"\nh={h:2d}  N={len(v):2d}")
    print(f"  full          mean {100*v.mean():+.3f}%  rec "
          f"{int((v>0).sum())}-{int((v<=0).sum())}  "
          f"vs pair base {100*base:+.3f}%")
    print(f"  drop best 1   mean {100*d1.mean():+.3f}%  (dropped "
          f"{100*v[o[0]]:+.2f}% on {e[o[0]].date()})")
    print(f"  drop best 2   mean {100*d2.mean():+.3f}%")
    if (~m26).sum():
        vn = v[~m26]
        print(f"  ex-2026       mean {100*vn.mean():+.3f}%  N={len(vn)}  rec "
              f"{int((vn>0).sum())}-{int((vn<=0).sum())}  sign p="
              f"{sign_test(int((vn>0).sum()), len(vn)):.4f}")
    if m26.sum():
        v26 = v[m26]
        print(f"  2026 only     mean {100*v26.mean():+.3f}%  N={len(v26)}")
    print(f"  cost check (4 bps): full {100*100*v.mean()/4:.1f}x  "
          f"dropbest {100*100*d1.mean()/4:.1f}x  drop2 {100*100*d2.mean()/4:.1f}x")

print("\n\n########## TURN-ON ARITHMETIC (h=3, the best-looking horizon) ##########")
h = 3
ret = vehicle_ret(px, [("XLK", 1.0), ("XLV", -1.0)], h)
valid = ret.dropna().index
e = declusters(px.index[SUB.values].intersection(valid), h, valid)
v = ret.loc[e].values
o = np.argsort(-v)
d1 = np.delete(v, o[0])
print(f"today N={len(v)}, record {int((v>0).sum())}-{int((v<=0).sum())}, "
      f"drop-best mean {100*d1.mean():+.3f}%")
print("cell fires ~7x/yr in the current regime (7 subclass episodes in 2026 alone).")
for k in (3, 4, 5, 6, 8):
    for w in range(k + 1):
        newv = np.concatenate([v, np.full(w, 0.008), np.full(k - w, -0.008)])
        oo = np.argsort(-newv)
        db = np.delete(newv, oo[0])
        rec_w = int((newv > 0).sum())
        if 100 * db.mean() >= 0.50 and rec_w / len(newv) >= 0.62:
            print(f"  +{k} new episodes, {w} winners -> N={len(newv)} "
                  f"rec {rec_w}-{len(newv)-rec_w} drop-best "
                  f"{100*db.mean():+.3f}%  <-- first combination that clears")
            break
    else:
        continue
    break
