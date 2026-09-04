"""Corn: +6.54% to a 52-week high, 5d return at the 100th percentile of its year.

Five triggers fired on ZC=F tonight (P1, P1b, P5, P5b, P6) and that is one story, not
five. The base cells disagree in sign: the 21d-rank momentum cell is mildly positive
(n=417 +0.185% t=1.97) while the fresh-52w-high cell is mildly negative (n=17
-0.049%). Which governs when a big session and a 52w high land together?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_prices, fwd_ret, declusters, local_control, summarize,
    era_split, sign_test, cluster_note,
)

raw = load_prices(["ZC=F", "ZS=F", "ZW=F"])
px = close_panel(["ZC=F", "ZS=F", "ZW=F"]).dropna(how="all")
c = px["ZC=F"].dropna()
print(f"ZC=F {c.index[0].date()} .. {c.index[-1].date()}  n={len(c)}")
print(f"today {100*(c.iloc[-1]/c.iloc[-2]-1):+.2f}%, close {c.iloc[-1]:.2f}, "
      f"252d max {c.tail(252).max():.2f}")

df = raw["ZC=F"]
tr = pd.concat([df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()          # Wilder-14
move_atr = (df["Close"] - df["Close"].shift(1)) / atr.shift(1)
print(f"today's session in ATR terms: {move_atr.iloc[-1]:+.2f} ATR")

at_high = c >= c.rolling(252).max() - 1e-9
print(f"at a 252d high today: {bool(at_high.iloc[-1])}")


def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    return (f"n={d['n']:5d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:5.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} "
            f"signp={sign_test(up, d['n']):.4f}")


def report(name, idx, hs=(1, 5, 21)):
    idx = pd.DatetimeIndex(sorted(set(idx)))
    print(f"\n--- {name}  (n = {len(idx)}) ---")
    for h in hs:
        print(f"  h{h:<3d} {line(summarize(fwd_ret(c, h).reindex(idx).dropna().values))}")
    return idx


report("every close at a 252d high", c.index[at_high])

# fresh: first 52w high in 90+ calendar days
fresh = []
last = None
for d in c.index[at_high]:
    if last is None or (d - last).days >= 90:
        fresh.append(d)
    last = d
report("first 252d high in 90+ calendar days", fresh)

# the combination that fired tonight
combo = c.index[at_high & (move_atr.reindex(c.index) >= 2.0)]
combo_idx = report("252d high AND a 2+ ATR session (tonight's shape)", combo)

big = c.index[at_high & (move_atr.reindex(c.index) >= 2.0) &
              (c.pct_change() >= 0.05)]
report("252d high AND a 5%+ session", big)

print("\n--- controls ---")
for h in [1, 5, 21]:
    print(f"  h{h:<3d} all days {line(summarize(fwd_ret(c, h).dropna().values))}")
ctrl = local_control(c.index, combo_idx, win=126).difference(combo_idx)
for h in [1, 5, 21]:
    print(f"  h{h:<3d} local    {line(summarize(fwd_ret(c, h).reindex(ctrl).dropna().values))}")

print("\n--- combo cell: declustered 21td, era, concentration ---")
dec = declusters(combo_idx, 21, c.index)
print(f"  n declustered = {len(dec)}")
for h in [1, 5, 21]:
    print(f"  h{h:<3d} {line(summarize(fwd_ret(c, h).reindex(dec).dropna().values))}")
v = fwd_ret(c, 21).reindex(combo_idx).dropna()
for part in era_split(v.index, v.values):
    u = int(round(part["hit"] / 100 * part["n"]))
    print(f"  h21 {part.get('label',''):12s} n={part['n']:3d} "
          f"mean={part['mean_pct']:+.3f}% rec={u}-{part['n']-u}")
print(f"  {cluster_note(v.index, v.values, k=2)}")

print("\n--- declustered episodes ---")
for d in dec:
    row = " ".join(f"{100*fwd_ret(c, h).get(d, np.nan):+7.2f}%" for h in (1, 5, 21, 63))
    print(f"  {str(d.date())}  session {100*(c[d]/c.shift(1)[d]-1):+5.2f}% "
          f"({move_atr.get(d, np.nan):+.1f} ATR)  h1/h5/h21/h63 {row}")

print("\n--- the wider grain complex on the same day ---")
for t in ["ZS=F", "ZW=F"]:
    s = px[t].dropna()
    print(f"  {t} today {100*(s.iloc[-1]/s.iloc[-2]-1):+.2f}%, "
          f"252d pctile {100*(s.tail(252) <= s.iloc[-1]).mean():.0f}")
