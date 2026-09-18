"""A9 round 2 - the only horizons where the cell is not already negative.

Round 1: at h=5 the defended cell is -0.614% over 16 episodes against its own
+0.270% drift, and BOTH gates are anti-filters (the discarded complements pay
+0.840% and +1.176% episode-level against the join's -0.614%). The trailing-year
dose response points the wrong way outright: break & 252ret in [-20,0)% pays
+1.785% while the live [50,1000)% bucket pays +0.980% and the [0,20)% bucket is
-0.183%.

The horizon scan leaves h=1 (+0.530%, edge +0.480pp) and h=2 (+1.391%, edge
+1.290pp) positive, and at h=1 the join beats both complements. This script
charges THAT, which is the strongest form of the candidate:
  - year decomposition (round 1 showed 10 of 16 episodes are 2026)
  - drop-best-2, drop-2026, era split
  - definition neighbours at h=1 and h=2
  - GLD beta at h=1
  - reference class across the commodity family at h=1
  - charged max-of-K over the (6 horizons x 9 threshold variants) walk
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
BREAK, DD_MIN = -0.040, 0.25

TK = ["SLV", "GLD", "GDX", "USO", "DBC", "XME", "PPLT", "SPY"]
px = load_prices(TK)
close = {t: px[t]["Close"] for t in px}


def st(t):
    s = close[t]
    r1 = s.pct_change()
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return s, r1, 1.0 - s / hi, s / s.shift(252) - 1.0


s, r1, dd, r252 = st("SLV")
base_mask = (r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0)

for H in (1, 2):
    print("=" * 78)
    print(f"h={H} - the defended cell, charged")
    f = fwd_lag(s, H, 1)
    val = f.dropna().index
    m = base_mask.reindex(val, fill_value=False).fillna(False)
    v = f.loc[val][m.values]
    allv = f.loc[val]
    w = int((v > 0).sum())
    print(f"  cell N={len(v)} mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
          f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f} "
          f"sign p vs 0.5 = {sign_test(w, len(v)):.4f}   "
          f"vs SLV's own up-rate {100*(allv>0).mean():.1f}%: "
          f"p={sign_test(w, len(v), float((allv>0).mean())):.4f}")
    print(f"  all days mean={100*allv.mean():+.3f}%  edge="
          f"{100*(v.mean()-allv.mean()):+.3f}pp  "
          f"cost 4 bps -> {100*v.mean()*100/4:.1f}x")
    byyear = v.groupby(v.index.year)
    print("  by year: " + "  ".join(
        f"{y}:n={len(g)},{100*g.mean():+.2f}%" for y, g in byyear))
    ordv = np.sort(v.values)
    print(f"  drop-best-2 {100*ordv[:-2].mean():+.3f}% (n={len(ordv)-2})   "
          f"drop-worst-2 {100*ordv[2:].mean():+.3f}%")
    pre26 = v[v.index.year < 2026]
    print(f"  ex-2026: N={len(pre26)} mean={100*pre26.mean():+.3f}% "
          f"hit={100*(pre26>0).mean():.1f}%" if len(pre26) else "  ex-2026: EMPTY")
    show(era_split(v.index, v.values), f"  era split h={H}")
    print("  " + cluster_note(v.index, v.values, k=2))

    print(f"\n  definition neighbours at h={H}:")
    for lbl, mm in [
        ("break<=-3%", (r1 <= -0.03) & (dd >= DD_MIN) & (r252 > 0)),
        ("break<=-3.5%", (r1 <= -0.035) & (dd >= DD_MIN) & (r252 > 0)),
        ("break<=-4.5%", (r1 <= -0.045) & (dd >= DD_MIN) & (r252 > 0)),
        ("break<=-5%", (r1 <= -0.05) & (dd >= DD_MIN) & (r252 > 0)),
        ("dd>=20%", (r1 <= BREAK) & (dd >= 0.20) & (r252 > 0)),
        ("dd>=30%", (r1 <= BREAK) & (dd >= 0.30) & (r252 > 0)),
        ("dd>=40%", (r1 <= BREAK) & (dd >= 0.40) & (r252 > 0)),
        ("yr>+20%", (r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0.20)),
        ("yr>+40%", (r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0.40)),
    ]:
        mmr = mm.reindex(val, fill_value=False).fillna(False)
        vv = f.loc[val][mmr.values]
        if len(vv) < 2:
            print(f"    {lbl:<14} N={len(vv)}")
            continue
        print(f"    {lbl:<14} N={len(vv):>3} mean={100*vv.mean():+.3f}% "
              f"hit={100*(vv>0).mean():>5.1f}% "
              f"t={vv.mean()/(vv.std(ddof=1)/np.sqrt(len(vv))):+.2f}")
    print()

print("=" * 78)
print("GLD BETA at h=1 and h=2")
for H in (1, 2):
    f = fwd_lag(s, 1 * H, 1)
    fg = fwd_lag(close["GLD"], H, 1)
    common = f.dropna().index.intersection(fg.dropna().index)
    b = np.polyfit(fg.loc[common].values, f.loc[common].values, 1)
    mm = base_mask.reindex(common, fill_value=False).fillna(False)
    y, x = f.loc[common][mm.values].values, fg.loc[common][mm.values].values
    res = y - (b[0] * x + b[1])
    print(f"  h={H}: beta {b[0]:.3f}   cell SLV {100*y.mean():+.3f}% "
          f"GLD {100*x.mean():+.3f}%   alpha {100*res.mean():+.3f}% "
          f"t={res.mean()/(res.std(ddof=1)/np.sqrt(len(res))):+.2f} "
          f"record {(res>0).sum()}-{(res<=0).sum()}")

print("\n" + "=" * 78)
print("REFERENCE CLASS at h=1 (identical rule, cached commodity vehicles)")
res = []
for t in ["SLV", "GLD", "GDX", "USO", "DBC", "XME", "PPLT"]:
    if t not in close:
        continue
    s2, rr1, ddt, rr252 = st(t)
    f = fwd_lag(s2, 1, 1)
    val = f.dropna().index
    m = ((rr1 <= BREAK) & (ddt >= DD_MIN) & (rr252 > 0)).reindex(
        val, fill_value=False).fillna(False)
    v = f.loc[val][m.values]
    if len(v) < 3:
        res.append((t, len(v), np.nan, np.nan, np.nan))
        continue
    res.append((t, len(v), 100 * v.mean(), 100 * (v.mean() - f.loc[val].mean()),
                v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))))
rd = pd.DataFrame(res, columns=["ticker", "n", "mean_pct", "excess_pct", "t"])
print(rd.round(3).to_string(index=False))
ok = rd.dropna(subset=["excess_pct"])
d = float(ok.set_index("ticker").loc["SLV", "excess_pct"])
print(f"  SLV excess {d:+.3f}% ranks {int((ok['excess_pct']>d).sum())+1} of {len(ok)}")

print("\n" + "=" * 78)
print("CHARGED MAX-OF-K: the walk that produced h=1 "
      "(6 horizons x 10 threshold variants = 60 cells), scored on the "
      "DEFENDED h=1 excess")
VARIANTS = {
    "base": base_mask,
    "b3": (r1 <= -0.03) & (dd >= DD_MIN) & (r252 > 0),
    "b35": (r1 <= -0.035) & (dd >= DD_MIN) & (r252 > 0),
    "b45": (r1 <= -0.045) & (dd >= DD_MIN) & (r252 > 0),
    "b5": (r1 <= -0.05) & (dd >= DD_MIN) & (r252 > 0),
    "dd20": (r1 <= BREAK) & (dd >= 0.20) & (r252 > 0),
    "dd30": (r1 <= BREAK) & (dd >= 0.30) & (r252 > 0),
    "dd40": (r1 <= BREAK) & (dd >= 0.40) & (r252 > 0),
    "yr20": (r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0.20),
    "yr40": (r1 <= BREAK) & (dd >= DD_MIN) & (r252 > 0.40),
}
cells = []
for h in (1, 2, 3, 5, 7, 10):
    f = fwd_lag(s, h, 1)
    val = f.dropna().index
    arr = f.loc[val].values
    for lbl, mm in VARIANTS.items():
        n = int(mm.reindex(val, fill_value=False).fillna(False).sum())
        if n >= 5:
            cells.append((arr, n))
f1 = fwd_lag(s, 1, 1)
v1 = f1.dropna()
m1 = base_mask.reindex(v1.index, fill_value=False).fillna(False)
defended = float(v1[m1.values].mean() - v1.mean())
rng = np.random.default_rng(9)
nulls = []
for _ in range(4000):
    best = -np.inf
    for arr, n in cells:
        best = max(best, arr[rng.integers(0, len(arr), size=n)].mean() - arr.mean())
    nulls.append(best)
nulls = np.asarray(nulls)
print(f"  K={len(cells)} cells; defended h=1 excess {100*defended:+.3f}%")
print(f"  null-max median {100*np.median(nulls):+.3f}%   "
      f"P(null max >= defended) = {float((nulls >= defended).mean()):.4f}")

print("\n" + "=" * 78)
print("IS h=1 AN OVERNIGHT-GAP ARTIFACT? decompose the h=1 hold")
o = px["SLV"]["Open"]
c = px["SLV"]["Close"]
idx = c.index
pos = pd.Series(range(len(idx)), index=idx)
gap, intraday, tot = [], [], []
for d in idx[base_mask.reindex(idx, fill_value=False).fillna(False).values]:
    p = pos[d]
    if p + 2 >= len(idx):
        continue
    entry = c.iloc[p + 1]          # MOC entry on D+1
    nxt_o, nxt_c = o.iloc[p + 2], c.iloc[p + 2]
    gap.append(nxt_o / entry - 1.0)
    intraday.append(nxt_c / nxt_o - 1.0)
    tot.append(nxt_c / entry - 1.0)
show([summarize(np.asarray(gap), "overnight gap D+1 close -> D+2 open"),
      summarize(np.asarray(intraday), "D+2 open -> D+2 close"),
      summarize(np.asarray(tot), "total h=1")], "h=1 decomposition")
