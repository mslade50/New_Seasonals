"""EWZ rose 4.16% tonight, a 2-ATR up session, with z10 2.23 and a 5d return
rank of 96.0; ^BVSP rose 3.05% on a fifth straight up close with z10 2.71.
Three separate engine triggers, one country. Base cells are per-trigger
(EWZ P6 up n=40 hit 65%, P4 n=223, ^BVSP P7 n=251) and none of them cross.
Cross them, and control for the fact that EWZ is a dollar-denominated wrapper
on a local index.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["EWZ", "^BVSP", "SPY", "USDBRL=X", "EEM"])
ewz = px["EWZ"]["Close"].dropna()
bv = px["^BVSP"]["Close"].dropna()

for t in ["EWZ", "^BVSP", "USDBRL=X", "EEM"]:
    c = px[t]["Close"].dropna()
    print(f"{t:<10} {c.index[-1].date()} {c.iloc[-1]:>12.3f}  "
          f"1d {100*(c.iloc[-1]/c.iloc[-2]-1):+.2f}%  "
          f"5d {100*(c.iloc[-1]/c.iloc[-6]-1):+.2f}%")


def atr_wilder(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rec(v, lab):
    d = summarize(v.values, lab)
    u = int((v > 0).sum())
    d["up"], d["down"] = u, len(v) - u
    d["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4) if len(v) else None
    return d


a = atr_wilder(px["EWZ"]).shift(1)
two_atr = (px["EWZ"]["Close"] - px["EWZ"]["Close"].shift(1)) >= 2.0 * a
z = zscore(ewz, 10)
rk5 = pct_rank(ewz, 5)
print(f"\nlive EWZ: 2ATR up {bool(two_atr.iloc[-1])}, z10 {z.iloc[-1]:.2f}, "
      f"5d rank {rk5.iloc[-1]:.1f}")

combos = [
    ("2-ATR up alone", two_atr),
    ("2-ATR up AND z10 >= 2", two_atr & (z >= 2)),
    ("2-ATR up AND 5d rank >= 95", two_atr & (rk5 >= 95)),
]
for lab, mask in combos:
    trig = ewz.index[mask.reindex(ewz.index).fillna(False).values]
    trig = trig[trig < ewz.index[-1]]
    epi = declusters(trig, 5, ewz.index)
    print(f"\n=== EWZ: {lab} -> {len(trig)} days, {len(epi)} episodes ===")
    if len(epi) < 5:
        print("  too few"); continue
    rows = []
    for h in (1, 5, 10):
        r = fwd_ret(ewz, h)
        v = r.reindex(epi).dropna()
        d = rec(v, f"EWZ h={h}")
        d["ctl_pct"] = round(100 * r.dropna().mean(), 3)
        d["edge_pct"] = round(d["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(d)
        rs = fwd_ret(px["SPY"]["Close"].dropna(), h)
        vs = rs.reindex(epi).dropna()
        if len(vs):
            rows.append(rec(vs, f"SPY h={h}"))
    show(rows)
    print("  episodes:", [str(d.date()) for d in epi][-10:])
    r1 = fwd_ret(ewz, 1)
    vv = r1.reindex(epi).dropna()
    show(era_split(vv.index, vv.values), "era split h=1")
    print(" ", cluster_note(vv.index, vv.values, k=2))

# ^BVSP 5-day up streak with the stretch
up = bv > bv.shift(1)
streak = up.groupby((~up).cumsum()).cumsum()
zb = zscore(bv, 10)
print(f"\nlive ^BVSP streak {int(streak.iloc[-1])}, z10 {zb.iloc[-1]:.2f}")
mask = (streak >= 5) & up & (zb >= 2)
trig = bv.index[mask.fillna(False).values]
trig = trig[trig < bv.index[-1]]
epi = declusters(trig, 5, bv.index)
print(f"=== ^BVSP 5+ up closes AND z10 >= 2: {len(trig)} days, {len(epi)} episodes ===")
if len(epi) >= 5:
    rows = []
    for h in (1, 5, 10):
        r = fwd_ret(bv, h)
        v = r.reindex(epi).dropna()
        d = rec(v, f"^BVSP h={h}")
        d["ctl_pct"] = round(100 * r.dropna().mean(), 3)
        d["edge_pct"] = round(d["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(d)
    show(rows)
    print("  episodes:", [str(d.date()) for d in epi])
