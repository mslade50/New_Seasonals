"""Bitcoin's 21d return in the top 5% of its year, while still deep below the
52-week high and negative on the year.

The engine cell (n=304, +0.734% h1, t 2.88, BH pass, era-stable) pools every
top-5% 21d reading, most of which happen at or near highs. Tonight is the
other kind: +26.8% over 21 sessions, still 17.7% under the 52-week high, with
the trailing year at -8.6%. Does the continuation survive that split?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, summarize, sign_test, era_split,  # noqa
                       cluster_note, pct_rank, declusters, local_control)

px = close_panel(["BTC-USD", "^GSPC"])
btc = px["BTC-USD"].dropna()
idx = btc.index
r21 = btc.pct_change(21)
rank21 = pct_rank(btc, 21, 252)
hi52 = btc.rolling(252).max()
dist = btc / hi52 - 1.0
yr = btc.pct_change(252)

print(f"BTC 2026-09-04: {btc.iloc[-1]:,.0f}  21d {100*r21.iloc[-1]:+.1f}%  "
      f"rank21 {rank21.iloc[-1]:.1f}  dist52wh {100*dist.iloc[-1]:+.1f}%  "
      f"252d {100*yr.iloc[-1]:+.1f}%")


def line(label, dates, s, h):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:44} h{h:<3} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum()); st = summarize(v, label)
    print(f"  {label:44} h{h:<3} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


base = rank21 >= 95
deep = base & (dist <= -0.10)
near = base & (dist > -0.10)
down_yr = base & (dist <= -0.10) & (yr < 0)

for nm, m in [("21d rank >= 95 (the engine cell)", base),
              ("  ... and >10% below the 52w high", deep),
              ("  ... and within 10% of the 52w high", near),
              ("  ... deep AND the year negative (tonight)", down_yr)]:
    trig = pd.DatetimeIndex([d for d in m.index[m.fillna(False)] if d < idx[-1]])
    dec = declusters(trig, 10, idx)
    print(f"\n{nm}: {len(trig)} sessions, {len(dec)} declustered")
    if len(dec) >= 3:
        print(f"  years {sorted(set(d.year for d in dec))}")
    for h in (1, 5, 10, 21):
        line(nm.strip(), dec, btc, h)
    if nm.startswith("  ... deep AND") and len(dec) >= 5:
        r = fwd_ret(btc, 21).reindex(pd.DatetimeIndex([d for d in dec if d in btc.index])).dropna()
        print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}%"
                         for e in era_split(r.index, r.values)])
        print("  concentration:", cluster_note(r.index, r.values, 2))
        print(f"  episodes: {[f'{str(d.date())}:{100*v:+.1f}%' for d, v in zip(r.index, r.values)]}")

print("\n=== controls ===")
for h in (1, 5, 21):
    r = fwd_ret(btc, h).dropna(); up = int((r.values > 0).sum())
    print(f"  {'all BTC sessions':44} h{h:<3} n={len(r):5d} "
          f"mean={100*r.values.mean():+7.3f}% med={100*np.median(r.values):+7.3f}% "
          f"hit={100*up/len(r):5.1f}%")
