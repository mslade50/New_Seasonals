"""C6 round 2 -- gate attribution, placebo offsets, definition neighbours,
decluster sensitivity, drop-best, registry-overlap with the closed r5 cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

px = close_panel(["GDX", "GLD", "SPY"])
px = px[px.index >= "2006-05-22"]
g = px["GDX"]
r21 = g / g.shift(21) - 1.0
r5 = g / g.shift(5) - 1.0
rk21 = rolling_on_valid(r21, lambda x: x.rolling(252).rank(pct=True) * 100.0)
rk21_126 = rolling_on_valid(r21, lambda x: x.rolling(126).rank(pct=True) * 100.0)
r1 = g.pct_change(fill_method=None)
sma200 = rolling_on_valid(g, lambda x: x.rolling(200).mean())

mask = ((rk21 >= 99) & (r1 <= -0.02)).fillna(False)

print("=== is today's +38% a genuine 21d move? ===")
print("close 21 sessions ago %.2f -> today %.2f" % (g.iloc[-22], g.iloc[-1]))
print("max drawdown-free? path min over the 21d: %.2f  max: %.2f"
      % (g.iloc[-22:].min(), g.iloc[-22:].max()))
print("sum of |daily| over 21d = %.1f%%, net = %.1f%% -> efficiency %.2f"
      % (100 * r1.iloc[-21:].abs().sum(), 100 * r21.iloc[-1],
         r21.iloc[-1] / r1.iloc[-21:].abs().sum()))
print("bar that rolled off today moved %.2f%% (a roll-off cannot make a +38%% move)"
      % (100 * (g.iloc[-22] / g.iloc[-23] - 1)))

# ---------- 1. GATE ATTRIBUTION ----------
print("\n\n########## 1. gate attribution ##########")
cells = {
    "C6 rank>=99 & 1d<=-2": mask,
    "1d<=-2 ALONE (no thrust)": (r1 <= -0.02).fillna(False),
    "1d<=-2 & above 200sma": ((r1 <= -0.02) & (g > sma200)).fillna(False),
    "1d<=-2 & r21>=+20%": ((r1 <= -0.02) & (r21 >= 0.20)).fillna(False),
    "rank>=99 ALONE": (rk21 >= 99).fillna(False),
    "rank>=99 & 1d>=0 (up day)": ((rk21 >= 99) & (r1 >= 0)).fillna(False),
    "all days": pd.Series(True, index=px.index),
}
for h in (5, 10):
    rows = []
    ret = fwd_lag(g, h, 1)
    ok = ret.notna().values
    for nm, m in cells.items():
        s = px.index[m.reindex(px.index, fill_value=False).values & ok]
        e = declusters(s, 10, px.index)
        r = summarize(ret.loc[e].values, nm)
        r["n_days"] = len(s)
        rows.append(r)
    show(rows, f"GDX h={h}, episodes (min_gap 10)")

# ---------- 2. PLACEBO OFFSET LADDER ----------
print("\n\n########## 2. placebo offset ladder (entry shifted k td) ##########")
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
trig = declusters(px.index[mask], 10, px.index)
for h in (5, 10):
    rows = []
    c = g.values
    for k in range(-10, 11):
        vals = []
        for d in trig:
            p = pos[d] + k
            if p + 1 + h >= len(idx) or p < 0:
                continue
            vals.append(c[p + 1 + h] / c[p + 1] - 1.0)
        rows.append(summarize(np.array(vals), f"offset {k:+d}"))
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    print(f"\n-- h={h} --")
    print(df[["label", "n", "mean_pct", "hit", "rank"]].round(3).to_string(index=False))
    print("  TRUE anchor (offset 0) ranks",
          int(df.loc[df.label == "offset +0", "rank"].iloc[0]), "of 21")

# ---------- 3. definition neighbours + decluster sensitivity ----------
print("\n\n########## 3. definition neighbours ##########")
nb = {
    "252d rank>=99 & 1d<=-2 (base)": mask,
    "126d rank>=99 & 1d<=-2": ((rk21_126 >= 99) & (r1 <= -0.02)).fillna(False),
    "252d rank>=99 & 1d<=-2.5": ((rk21 >= 99) & (r1 <= -0.025)).fillna(False),
    "252d rank>=99 & 1d<=-3": ((rk21 >= 99) & (r1 <= -0.03)).fillna(False),
    "252d rank>=97 & 1d<=-2": ((rk21 >= 97) & (r1 <= -0.02)).fillna(False),
    "252d rank=100 & 1d<=-2": ((rk21 >= 99.99) & (r1 <= -0.02)).fillna(False),
    "MAG r21>=30% & 1d<=-2": ((r21 >= 0.30) & (r1 <= -0.02)).fillna(False),
}
for h in (5, 10):
    ret = fwd_lag(g, h, 1)
    ok = ret.notna().values
    rows = []
    for nm, m in nb.items():
        s = px.index[m.values & ok]
        e = declusters(s, 10, px.index)
        r = summarize(ret.loc[e].values, nm); r["n_days"] = len(s)
        rows.append(r)
    show(rows, f"neighbours, GDX h={h}")

print("\n########## decluster sensitivity (base cell) ##########")
for h in (5, 10):
    ret = fwd_lag(g, h, 1); ok = ret.notna().values
    s = px.index[mask.values & ok]
    rows = [summarize(ret.loc[declusters(s, gap, px.index)].values,
                      f"h={h} min_gap={gap}") for gap in (1, 5, 10, 21, 63)]
    show(rows, f"decluster h={h}")

# ---------- 4. drop-best / drop-worst ----------
print("\n\n########## 4. leave-one-out on episodes ##########")
for h in (5, 10):
    ret = fwd_lag(g, h, 1); ok = ret.notna().values
    e = declusters(px.index[mask.values & ok], 10, px.index)
    v = ret.loc[e].values
    print(f"h={h}: N={len(v)} mean {100*v.mean():+.3f}%  "
          f"drop-best {100*np.sort(v)[:-1].mean():+.3f}%  "
          f"drop-best-2 {100*np.sort(v)[:-2].mean():+.3f}%  "
          f"min {100*v.min():+.3f}%")
    for d, x in zip(e, v):
        print(f"    {d.date()}  {100*x:+.2f}%   (r21 {100*r21.loc[d]:+.1f}%, "
              f"1d {100*r1.loc[d]:+.2f}%, midterm={d.year % 4 == 2})")

# ---------- 5. registry overlap: the closed r5 thrust cell ----------
print("\n\n########## 5. overlap with the CLOSED r5>+10% cell ##########")
e = declusters(px.index[mask], 10, px.index)
print("cell episodes with GDX r5 > +10% on the trigger day:",
      int((r5.loc[e] > 0.10).sum()), "of", len(e))
print("r5 on cell days:", [f"{100*r5.loc[d]:+.1f}%" for d in e])
today = px.index[-1]
print("TODAY r5 = %+.1f%% (the closed cell's wrong half is >+10%%)"
      % (100 * r5.iloc[-1]))

# ---------- 6. horizon scan + loser paths ----------
print("\n\n########## 6. horizon scan ##########")
show(horizon_scan(px, e, [("GDX", 1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  min_gap=10), "GDX horizon scan (episodes)")
print("\nepisode paths, h=10:")
print((100 * episode_paths(px, e, [("GDX", 1.0)], 10)).round(2).to_string())
