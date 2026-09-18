"""Two repairs.

(a) Yesterday's brief established that Yahoo FX pairs and DX-Y.NYB stamp their
    daily bar ahead of the US afternoon, which is why it ran its dollar leg on
    UUP. Re-run the September-witching dollar cell on UUP, a US-listed ETF
    with a real 16:00 close, and see whether it survives the shorter history.

(b) A today-lane candidate that touches no FX: TLT closed 1.33% above its
    52-week low and IEF 0.57% above its own, while SPY sits 2.07% below a
    52-week high. How often are long bonds pinned at lows with equities at
    highs, and what has followed?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, declusters, cluster_note,
)

px = close_panel(["UUP", "DX-Y.NYB", "TLT", "IEF", "SPY", "IWM", "^TNX"])

# ---- (a) UUP on the September witching session -------------------------
u = px["UUP"].dropna()
f1 = u.shift(-1) / u - 1.0
idx = u.index
pos = pd.Series(range(len(idx)), index=idx)
qw = pd.DatetimeIndex(load_events(["quad_witching"])["date"])
anch = pd.DatetimeIndex([idx[pos[x] - 1] for x in qw if x in pos.index and pos[x] > 0])
anch = anch.intersection(f1.dropna().index)
print(f"(a) UUP history from {u.index[0].date()}; witching anchors {len(anch)}")
rows = [summarize(f1.loc[anch].values, "all witchings")]
for m, name in [(3, "March"), (6, "June"), (9, "September"), (12, "December")]:
    d = anch[anch.month == m]
    r = summarize(f1.loc[d].values, name)
    if r["n"]:
        k = int((f1.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["sign_p_up"] = round(sign_test(k, r["n"]), 4)
    rows.append(r)
rows.append(summarize(f1.dropna().values, "CTL all sessions"))
show(rows, "UUP on the witching session")
sep = anch[anch.month == 9]
print(f"  September years: {sorted(set(sep.year))}")
print(f"  values: {dict((d.year, round(100 * f1.loc[d], 3)) for d in sep)}")

# same cell on DXY restricted to UUP's era, to see whether the two agree
dx = px["DX-Y.NYB"].dropna()
fd = dx.shift(-1) / dx - 1.0
sepd = pd.DatetimeIndex([d for d in sep]).intersection(fd.dropna().index)
k = int((fd.loc[sepd] > 0).sum())
print(f"  DXY on the SAME sessions: n={len(sepd)} mean "
      f"{100 * fd.loc[sepd].mean():+.3f}% {k}-{len(sepd) - k} up")
corr = pd.concat([f1.loc[sepd], fd.loc[sepd]], axis=1).corr().iloc[0, 1]
print(f"  UUP vs DXY correlation on those sessions: {corr:.2f}")

# ---- (b) bonds at lows, stocks at highs --------------------------------
print("\n(b) long bonds at 52-week lows with equities near highs")
t, i, s = px["TLT"].dropna(), px["IEF"].dropna(), px["SPY"].dropna()
common = t.index.intersection(i.index).intersection(s.index)
t, i, s = t.reindex(common), i.reindex(common), s.reindex(common)
tl = t / t.rolling(252).min() - 1.0
il = i / i.rolling(252).min() - 1.0
sh = s / s.rolling(252).max() - 1.0
print(f"  today: TLT {100 * tl.iloc[-1]:+.2f}% above its 252d low, "
      f"IEF {100 * il.iloc[-1]:+.2f}%, SPY {100 * sh.iloc[-1]:+.2f}% from its 252d high")

mask = (tl <= 0.02) & (il <= 0.02) & (sh >= -0.03)
d = common[mask.fillna(False).values]
print(f"  sessions matching: {len(d)} of {len(common)}")
epi = declusters(d, 21, common)
print(f"  declustered (21td): {len(epi)}; by year "
      f"{dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")
print(f"  dates: {[str(x.date()) for x in epi]}")

for tkr, ser in [("SPY", s), ("TLT", t), ("IWM", px['IWM'].dropna().reindex(common))]:
    for h in (5, 21):
        f = ser.shift(-h) / ser - 1.0
        e = epi.intersection(f.dropna().index)
        if len(e) < 3:
            print(f"    {tkr} h={h}: n={len(e)} too few")
            continue
        k = int((f.loc[e] > 0).sum())
        print(f"    {tkr} h={h}: n={len(e)} mean {100 * f.loc[e].mean():+.2f}% "
              f"median {100 * f.loc[e].median():+.2f}% {k}-{len(e) - k} up "
              f"(ctl {100 * f.dropna().mean():+.2f}%) "
              f"sign p(down) = {sign_test(len(e) - k, len(e)):.4f}")

print("\n  looser: TLT within 3% of a 252d low and SPY within 5% of a 252d high")
mask2 = (tl <= 0.03) & (sh >= -0.05)
d2 = common[mask2.fillna(False).values]
epi2 = declusters(d2, 21, common)
print(f"    n={len(d2)} sessions, {len(epi2)} episodes, by year "
      f"{dict(pd.Series(1, index=epi2).groupby(epi2.year).sum())}")
for tkr, ser in [("SPY", s), ("TLT", t)]:
    for h in (5, 21, 63):
        f = ser.shift(-h) / ser - 1.0
        e = epi2.intersection(f.dropna().index)
        if len(e) < 3:
            continue
        k = int((f.loc[e] > 0).sum())
        print(f"    {tkr} h={h}: n={len(e)} mean {100 * f.loc[e].mean():+.2f}% "
              f"{k}-{len(e) - k} up (ctl {100 * f.dropna().mean():+.2f}%) "
              f"sign p(down) = {sign_test(len(e) - k, len(e)):.4f}")
print(f"    concentration SPY h=21: "
      f"{cluster_note(epi2.intersection((s.shift(-21) / s - 1.0).dropna().index), (s.shift(-21) / s - 1.0).loc[epi2.intersection((s.shift(-21) / s - 1.0).dropna().index)].values, k=2)}")
