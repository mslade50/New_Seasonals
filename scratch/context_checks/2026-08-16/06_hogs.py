"""Lean hogs printed -14.38% on Friday. Where does that rank, and what followed?

Three of tonight's price triggers are the same event: `P6:two_atr_day` down (n=115, h1 -0.47%,
t -2.77), `P5:rank5_extreme` bottom, `P5b:rank21_extreme` bottom, plus a 200d cross down. The
engine's 2-ATR cell pools every 2-ATR session; a -14% session is not that animal, so the drill
is to rank the move itself and look only at the comparable tail.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, summarize, sign_test, wilder_atr, declusters, cluster_note  # noqa

px = load_prices(["HE=F", "LE=F"])
h = px["HE=F"]
c = h["Close"].dropna()
r = c.pct_change()
print(f"HE=F {c.index[0].date()} .. {c.index[-1].date()}  n {len(c)}")
print(f"Friday: {100*r.iloc[-1]:+.2f}%  close {c.iloc[-1]:.2f}")

worst = r.nsmallest(8)
print("\nthe eight worst sessions in the series")
for d, x in worst.items():
    print(f"  {d.date()}  {100*x:+.2f}%")
print(f"rank of Friday among {len(r.dropna())} sessions: "
      f"{int((r < r.iloc[-1]).sum()) + 1} from the bottom "
      f"({100*(r < r.iloc[-1]).mean():.2f}th percentile)")

atr = wilder_atr(h["High"], h["Low"], h["Close"], 14)
atr_s = pd.Series(atr, index=h.index)
move_atr = (c - c.shift(1)) / atr_s.shift(1)
print(f"Friday in ATR units: {move_atr.iloc[-1]:.2f} ATR "
      f"(prior-bar Wilder-14 ATR {atr_s.iloc[-2]:.3f})")

# the comparable tail only: sessions at or beyond 4 ATR down, declustered
for thresh in (2, 3, 4, 5):
    trig = declusters(move_atr.index[(move_atr <= -thresh).fillna(False)], 5, c.index)
    print(f"\n<= -{thresh} ATR, declustered: n {len(trig)}")
    for hz in (1, 5, 10, 21):
        v = fwd_ret(c, hz).reindex(trig).dropna()
        if len(v) < 4:
            continue
        s = summarize(v.values, "")
        up = int((v > 0).sum())
        print(f"  h{hz:<3d} n {s['n']:3d}  {up}-{s['n']-up}  mean {s['mean_pct']:+.2f}%  "
              f"med {s['median_pct']:+.2f}%  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")

base = fwd_ret(c, 5).dropna()
print(f"\ncontrol, every session h5: n {len(base)} mean {100*base.mean():+.2f}% "
      f"med {100*base.median():+.2f}% up {100*(base>0).mean():.1f}%")

# limit moves: does the exchange limit make the next session mechanical
big = declusters(r.index[(r <= -0.08).fillna(False)], 5, c.index)
print(f"\nsessions at or below -8%, declustered: n {len(big)}")
for hz in (1, 2, 5, 10, 21):
    v = fwd_ret(c, hz).reindex(big).dropna()
    s = summarize(v.values, "")
    up = int((v > 0).sum())
    print(f"  h{hz:<3d} n {s['n']:3d}  {up}-{s['n']-up}  mean {s['mean_pct']:+.2f}%  "
          f"med {s['median_pct']:+.2f}%  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")
v = fwd_ret(c, 5).reindex(big).dropna()
print("  h5 cluster:", cluster_note(v.index, v.values, k=2))
print("  dates:", ", ".join(f"{d.date()}({100*r[d]:.0f}%)" for d in big))

# is cattle doing the same thing
lc = px["LE=F"]["Close"].dropna()
print(f"\nLE=F Friday {100*lc.pct_change().iloc[-1]:+.2f}%, "
      f"5d {100*(lc.iloc[-1]/lc.iloc[-6]-1):+.2f}%; HE=F 5d {100*(c.iloc[-1]/c.iloc[-6]-1):+.2f}%")
