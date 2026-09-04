"""Does the surviving dollar cell have a cross-asset leg worth trading on its
own, or is the dollar the only honest expression?

The cell: DX 5d return rank <= 25 AND 63d rank >= 70 AND an NFP print inside
the next 3 sessions. It survived every attack on DX itself. The question here
is whether the dollar-sensitive assets (gold, silver, miners, EM, bonds,
energy) carry a separate, tradeable move in the same window.

Six assets are examined. That is a six-cell sweep on one trigger and the
multiplicity is reported, not hidden. A prior sweep of UNCONDITIONAL
cross-asset NFP behaviour (36 cells) was killed today on exactly this ground,
so the bar is: beat the asset's own unconditional drift over the identical
MOO->MOC hold, on DECLUSTERED episodes, and survive a permutation null on the
grid's max |t|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 220)
ASSETS = ["GLD", "SLV", "GDX", "EEM", "TLT", "XLE"]
P = C.load(["DX-Y.NYB"] + ASSETS)
DX = P["DX-Y.NYB"]

ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv")
nfp = pd.DatetimeIndex(sorted(pd.to_datetime(
    ev.loc[ev["event"].astype(str).str.lower().str.contains("nfp|payroll"),
           "date"]).dt.normalize().unique()))

idx = DX.index
pos = {d: i for i, d in enumerate(idx)}
nfp_pos = np.array(sorted({pos[d] for d in nfp if d in pos}))
dist = np.full(len(idx), 10 ** 6)
for i in range(len(idx)):
    nxt = nfp_pos[nfp_pos > i]
    if len(nxt):
        dist[i] = nxt[0] - i
dist = pd.Series(dist, index=idx)

close = DX["Close"]
rk5 = C.pct_rank(C.ret(close, 5))
rk63 = C.pct_rank(C.ret(close, 63))
sig_dates = idx[((rk5 <= 25) & (rk63 >= 70) & (dist <= 3)).to_numpy()]
print(f"DX cell signal dates: {len(sig_dates)}  "
      f"({sig_dates.min().date()} .. {sig_dates.max().date()})")

HOLD = 3   # MOO at signal+1, MOC at signal+3 -- the live idea's geometry


def executable(df, hold, dates):
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    p = {d: i for i, d in enumerate(df.index)}
    out = {}
    for d in dates:
        i = p.get(d)
        if i is None or i + hold >= len(c):
            continue
        out[d] = (c[i + hold] / o[i + 1] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def uncond(df, hold):
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    v = (c[hold:] / o[1:len(c) - hold + 1] - 1.0) * 100.0
    return pd.Series(v, index=df.index[:len(v)])


rows, tmax = [], 0.0
for a in ASSETS + ["DX-Y.NYB"]:
    df = P[a]
    s = executable(df, HOLD, sig_dates)
    b = uncond(df, HOLD).dropna()
    ep = s[C.declusterize(s.index, gap_td=10)]
    for nm, v in (("days", s), ("eps g10", ep),
                  ("eps g21", s[C.declusterize(s.index, gap_td=21)])):
        d = C.describe(f"{a} {nm}", v, baseline=b)
        x, y = np.asarray(v, float), np.asarray(b, float)
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        d["welch_t"] = round(float((x.mean() - y.mean()) / se), 2)
        rows.append(d)
        if a != "DX-Y.NYB" and nm == "eps g10":
            tmax = max(tmax, abs(d["welch_t"]))
C.show(rows)

print(f"\nmax |lift welch t| across the 6 cross-asset episode cells: {tmax:.2f}")

# permutation null: same number of signal dates drawn at random, same grid
rng = np.random.default_rng(7)
null = []
pool = idx[250:-HOLD - 1]
for _ in range(2000):
    fake = pd.DatetimeIndex(rng.choice(pool, size=len(sig_dates), replace=False))
    m = 0.0
    for a in ASSETS:
        df = P[a]
        s = executable(df, HOLD, fake)
        if len(s) < 5:
            continue
        v = s[C.declusterize(s.index, gap_td=10)]
        b = uncond(df, HOLD).dropna()
        x, y = np.asarray(v, float), np.asarray(b, float)
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        m = max(m, abs(float((x.mean() - y.mean()) / se)))
    null.append(m)
null = np.array(null)
print(f"permutation null on the 6-cell grid max|welch t| (2000 draws): "
      f"median {np.median(null):.2f}  p90 {np.percentile(null, 90):.2f}  "
      f"p95 {np.percentile(null, 95):.2f}  "
      f"P(null >= observed {tmax:.2f}) = {(null >= tmax).mean():.3f}")

print("\n-- era split on each asset's episode series --")
for a in ASSETS:
    s = executable(P[a], HOLD, sig_dates)
    ep = s[C.declusterize(s.index, gap_td=10)]
    if len(ep) < 6:
        print(f"  {a}: n={len(ep)} too few")
        continue
    pre, post = ep[ep.index < "2018-01-01"], ep[ep.index >= "2018-01-01"]
    print(f"  {a}: pre-2018 n={len(pre)} avg {pre.mean():+.3f} t {C.tstat(pre.values):.2f}"
          f" | 2018+ n={len(post)} avg {post.mean():+.3f} t {C.tstat(post.values):.2f}")
