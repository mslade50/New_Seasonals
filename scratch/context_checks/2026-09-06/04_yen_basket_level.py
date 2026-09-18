"""The yen, framed on LEVEL rather than on the shock.

The last two briefs both published the reflex-bounce cell (four crosses at
2 ATR on 09-02, USDJPY's own 2 ATR day on 09-03) and the bounce did not show
either time. A third telling is banned. The new fact is the level: USDJPY's
63-day return rank closed at 0.4, effectively the weakest dollar against the
yen in a quarter, with six crosses in the bottom 5% of their year at once.

Cell tested here: USDJPY 63d return rank <= 2, declustered, multi-session
horizons. Plus the basket-breadth count and the NZDJPY 200d cross.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, summarize, sign_test, era_split,  # noqa
                       cluster_note, pct_rank, declusters, local_control)

CROSS = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X"]
px = close_panel(["JPY=X", "^GSPC", "SPY"] + CROSS)
usdjpy = px["JPY=X"].dropna()
spx = px["^GSPC"].dropna()
idx = usdjpy.index


def line(label, dates, s, h):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:40} h{h:<3} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum()); st = summarize(v, label)
    print(f"  {label:40} h{h:<3} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


r63 = pct_rank(usdjpy, 63, 252)
print(f"USDJPY 63d return rank on 2026-09-04: {r63.iloc[-1]:.2f}")
print(f"USDJPY 5d rank {pct_rank(usdjpy,5,252).iloc[-1]:.1f}, "
      f"21d rank {pct_rank(usdjpy,21,252).iloc[-1]:.1f}")

mask = (r63 <= 2.0)
trig = mask.index[mask.fillna(False)]
trig = pd.DatetimeIndex([d for d in trig if d < idx[-1]])
dec = declusters(trig, 10, idx)
print(f"\nUSDJPY 63d rank <= 2: {len(trig)} sessions, {len(dec)} declustered at 10 td")
print(f"  episodes by year: {pd.Series([d.year for d in dec]).value_counts().sort_index().to_dict()}")

print("\n=== USDJPY forward from the 63d-rank-<=2 state (declustered) ===")
for h in (1, 5, 10, 21):
    line("USDJPY, 63d rank <= 2", dec, usdjpy, h)
print("  controls")
ctrl = local_control(idx, dec, 126)
for h in (1, 5, 10, 21):
    line("  local +/-126td neighbourhood", ctrl, usdjpy, h)
for h in (1, 5, 10, 21):
    r = fwd_ret(usdjpy, h).dropna()
    up = int((r.values > 0).sum())
    print(f"  {'  all sessions':40} h{h:<3} n={len(r):5d} mean={100*r.values.mean():+7.3f}% "
          f"hit={100*up/len(r):5.1f}%")

r5 = line("USDJPY, 63d rank <= 2", dec, usdjpy, 5)
if r5 is not None:
    print("  era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}% hit={e['hit']:.1f}%"
                     for e in era_split(r5.index, r5.values)])
    print("  concentration:", cluster_note(r5.index, r5.values, 2))

print("\n=== the S&P alongside it ===")
for h in (1, 5, 10, 21):
    line("^GSPC from USDJPY 63d rank <= 2", dec, spx, h)

print("\n=== basket breadth: crosses in the bottom 5% of their year at once ===")
low5 = pd.DataFrame({c: (pct_rank(px[c], 5, 252) <= 5.0) for c in CROSS})
cnt = low5.sum(axis=1)
print(f"  today's count: {int(cnt.iloc[-1])} of {len(CROSS)}")
hist = cnt.index[(cnt >= 5) & (cnt.index < idx[-1])]
hd = declusters(pd.DatetimeIndex(hist), 10, idx)
print(f"  5+ at once: {len(hist)} sessions, {len(hd)} declustered episodes")
print(f"  episodes: {[str(d.date()) for d in hd]}")
for h in (1, 5, 10, 21):
    line("USDJPY after 5+ crosses at a 5d low", hd, usdjpy, h)
for h in (1, 5, 21):
    line("^GSPC after 5+ crosses at a 5d low", hd, spx, h)

print("\n=== NZDJPY 200d cross (state descriptor only) ===")
nz = px["NZDJPY=X"].dropna()
sma = nz.rolling(200).mean()
above = nz > sma
cross_dn = above.index[(~above) & above.shift(1).fillna(False)]
gaps = [cross_dn[i] for i in range(len(cross_dn))
        if i == 0 or (nz.index.get_loc(cross_dn[i]) - nz.index.get_loc(cross_dn[i - 1])) >= 63]
print(f"  first 200d cross down in 63+ sessions: {len(gaps)} episodes, "
      f"latest {gaps[-1].date() if gaps else 'n/a'}")
for h in (5, 21):
    line("NZDJPY after that cross", pd.DatetimeIndex(gaps[:-1]), nz, h)
