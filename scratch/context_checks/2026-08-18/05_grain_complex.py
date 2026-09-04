"""Grains: corn's second leg, and all three grains extreme at once.

Last night published corn's 52-week high. Tonight corn added +4.89% on top of
that, its 5d return is the 100th percentile of its own year, and soybeans
(99.2) and wheat (97.2) are simultaneously extreme. Republishing the breakout
would be a countdown re-telling, so the only legal questions are about the
EXTENSION and the SIMULTANEITY, neither of which last night's cell contained.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
    declusters, pct_rank, wilder_atr, load_prices,
)

G = ["ZC=F", "ZS=F", "ZW=F"]
px = close_panel(G)
idx = px.index
r1 = px.pct_change(fill_method=None)

print("current state:")
for t in G:
    r5 = px[t].pct_change(5)
    rk = pct_rank(px[t], 5)
    print(f"  {t}: last close {px[t].iloc[-1]:.2f}, 1d {100*r1[t].iloc[-1]:+.2f}%, "
          f"5d {100*r5.iloc[-1]:+.2f}%, 5d pctile {rk.iloc[-1]:.1f}")

print("\n" + "=" * 74)
print("A. THE SIMULTANEITY CELL: corn, soy and wheat all >=95th pctile on 5d")
print("=" * 74)
rk = pd.DataFrame({t: pct_rank(px[t], 5) for t in G})
mask = (rk["ZC=F"] >= 95) & (rk["ZS=F"] >= 95) & (rk["ZW=F"] >= 95)
trig = idx[mask.fillna(False).values]
print(f"raw trigger days: {len(trig)}, first {trig[0].date()}, last {trig[-1].date()}")
epi = declusters(trig, 10, idx)
print(f"declustered episodes (10td gap): {len(epi)}")
print("  episodes:", [str(d.date()) for d in epi][-14:])

for t in G:
    out = []
    for h in (1, 5, 10, 21):
        f = px[t].shift(-h) / px[t] - 1.0
        v = f.loc[epi].dropna().values
        r = summarize(v, f"h={h}")
        r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        base = f.dropna()
        r["ctl_all_pct"] = round(100 * base.mean(), 3)
        r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        out.append(r)
    show(out, f"{t} after all three grains hit a 5d extreme together")

print("\n" + "=" * 74)
print("B. an equal-weight grain basket after the simultaneity")
print("=" * 74)
bask = px.pct_change(fill_method=None).mean(axis=1)
cum = (1 + bask).cumprod()
out = []
for h in (1, 5, 10, 21):
    f = cum.shift(-h) / cum - 1.0
    v = f.loc[epi].dropna().values
    r = summarize(v, f"basket h={h}")
    r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    r["ctl_all_pct"] = round(100 * base.mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    out.append(r)
show(out, "equal-weight corn/soy/wheat")
v10 = (cum.shift(-10) / cum - 1.0).loc[epi].dropna()
show(era_split(v10.index, v10.values), "basket h=10, era split")
print(" ", cluster_note(v10.index, v10.values, k=2))

print("\n" + "=" * 74)
print("C. THE EXTENSION: corn's second consecutive big up day at a 52w high")
print("=" * 74)
raw = load_prices(["ZC=F"])["ZC=F"]
atr = pd.Series(np.asarray(wilder_atr(raw["High"], raw["Low"], raw["Close"], 14)),
                index=raw.index)
c = raw["Close"]
chg = c.diff()
big = chg >= 1.5 * atr.shift(1)
hi52 = c >= c.rolling(252).max()
two_in_row = big & big.shift(1) & hi52
tt = raw.index[two_in_row.fillna(False).values]
tt = declusters(pd.DatetimeIndex(tt), 5, raw.index)
print(f"corn: 2 consecutive 1.5-ATR up days ending at a 252d high -> "
      f"{len(tt)} episodes")
print("  last ten:", [str(d.date()) for d in tt][-10:])
out = []
for h in (1, 5, 10, 21):
    f = c.shift(-h) / c - 1.0
    v = f.loc[f.index.intersection(tt)].dropna().values
    r = summarize(v, f"h={h}")
    r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    r["ctl_all_pct"] = round(100 * base.mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    out.append(r)
show(out, "corn after a two-day thrust into a 52-week high")

print("\n" + "=" * 74)
print("D. contrast: a SINGLE big up day at a 52w high (last night's shape)")
print("=" * 74)
one = big & hi52 & ~big.shift(1).fillna(False)
o = declusters(pd.DatetimeIndex(raw.index[one.fillna(False).values]), 5, raw.index)
out = []
for h in (1, 5, 10, 21):
    f = c.shift(-h) / c - 1.0
    v = f.loc[f.index.intersection(o)].dropna().values
    r = summarize(v, f"h={h}")
    r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    out.append(r)
show(out, "corn after ONE big up day at a 52-week high (n episodes above)")
print(f"  single-thrust episodes: {len(o)}, two-day-thrust episodes: {len(tt)}")
