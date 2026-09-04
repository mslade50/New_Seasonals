"""C2 round 2 -- tear down the ONE pretty thing round 1 produced.

Round 1 killed the pitched cell (TNX within 1% of its 252d high x MOVE level
pctile <= 50): the conjunction beats neither parent on TLT at h=3/5/10/21, the
episode-vs-control diff is -0.248% at Welch t -0.72 (13-9, sign p 0.26), and
midterm is -0.453% at a 42.9% hit.

What survived the depth split was the LIVE band [40,50): TLT h=5 +1.064% on 7
episodes at a 100% hit, gate +1.269pp. This script decides whether that band is
a cell or a scan artefact, and separately checks the ORDERING of the MOVE
ladder against the mechanism the candidate states.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "^MOVE", "TLT", "IEF"])
tnx, move = px["^TNX"], px["^MOVE"]
tnx_dist = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
move_pct = rolling_on_valid(move, lambda x: x.rolling(252).apply(
    lambda w: 100.0 * (w <= w[-1]).mean(), raw=True))
usable = tnx_dist.notna() & move_pct.notna()
yield_high = (tnx_dist >= -0.01) & usable

# ------------------------------------------------------------------ 1
print("=== 1. WHAT ARE THE 10 DAYS IN THE LIVE BAND? ===")
band = yield_high & (move_pct >= 40) & (move_pct < 50)
d = px.index[band.values]
print("  days:", ", ".join(str(x.date()) for x in d))
print("  years:", dict(pd.Series(1, index=d).groupby(d.year).sum()))
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    e = declusters(d, max(h, 10), px.index)
    v = r.loc[e].values
    print(f"  h={h}: episodes {len(e)} -> " +
          ", ".join(f"{str(x.date())} {100*y:+.2f}%" for x, y in zip(e, v)))
    print(f"        mean {100*np.nanmean(v):+.3f}%  drop-best-1 "
          f"{100*np.mean(np.sort(v)[:-1]):+.3f}%  drop-best-2 "
          f"{100*np.mean(np.sort(v)[:-2]):+.3f}%")
    yrs = pd.Series(v, index=e.year).groupby(level=0).mean()
    print("        by year:", {k: round(100*val, 2) for k, val in yrs.items()})

# ------------------------------------------------------------------ 2
print("\n=== 2. MECHANISM ORDERING: the candidate says DISORDERLY (high MOVE) "
      "mean-reverts -> LONG duration; ORDERLY (low MOVE) trends -> SHORT. ===")
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    ok = r.notna()
    print(f"  LONG-TLT mean by MOVE level pctile band, h={h} (yield-high only):")
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]:
        m = yield_high & (move_pct >= lo) & (move_pct < hi)
        dd = px.index[m.values & ok.values]
        if len(dd) < 2:
            continue
        e = declusters(dd, max(h, 10), px.index)
        print(f"    MOVE [{lo:3d},{hi:3d}): {100*np.nanmean(r.loc[e].values):+7.3f}%  "
              f"(N_epi={len(e)}, N_days={len(dd)})")
    print("    mechanism predicts this should INCREASE with the MOVE band "
          "(disorderly overshoot -> long duration pays).")

# ------------------------------------------------------------------ 3
print("\n=== 3. BAND DEFINITION FRAGILITY (neighbours of [40,50)) ===")
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    ok = r.notna()
    rows = []
    for lo, hi in [(40, 50), (35, 55), (38, 52), (42, 48), (30, 50), (40, 60),
                   (35, 50), (40, 55), (44.4 - 5, 44.4 + 5), (44.4 - 10, 44.4 + 10)]:
        m = yield_high & (move_pct >= lo) & (move_pct < hi)
        dd = px.index[m.values & ok.values]
        if len(dd) < 2:
            rows.append({"band": f"[{lo},{hi})", "n_days": len(dd), "n": 0})
            continue
        e = declusters(dd, max(h, 10), px.index)
        s = summarize(r.loc[e].values, f"[{lo},{hi})")
        s["n_days"] = len(dd)
        rows.append(s)
    show(rows, f"TLT h={h} LONG, band neighbours")

    # and the yield gate neighbours holding the MOVE band fixed
    rows = []
    for yd in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
        m = (tnx_dist >= -yd) & usable & (move_pct >= 40) & (move_pct < 50)
        dd = px.index[m.values & ok.values]
        if len(dd) < 2:
            rows.append({"yield_gate": f"{100*yd:.2f}%", "n": 0})
            continue
        e = declusters(dd, max(h, 10), px.index)
        s = summarize(r.loc[e].values, f"TNX within {100*yd:.2f}%")
        s["n_days"] = len(dd)
        rows.append(s)
    show(rows, f"TLT h={h} LONG, yield-gate ladder at MOVE [40,50)")

# ------------------------------------------------------------------ 4
print("\n=== 4. MULTIPLICITY: max-of-6-bands permutation ===")
rng = np.random.default_rng(42)
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    ok = r.notna()
    parent_days = px.index[yield_high.values & ok.values]
    parent_e = declusters(parent_days, max(h, 10), px.index)
    parent_vals = r.loc[parent_e].values
    parent_mean = np.nanmean(parent_vals)
    # observed gate of the live band, episode level
    bd = px.index[band.values & ok.values]
    be = declusters(bd, max(h, 10), px.index)
    obs = np.nanmean(r.loc[be].values) - parent_mean
    # null: reshuffle which parent episodes land in which band, keeping the
    # band size distribution fixed
    sizes = []
    for lo, hi in [(0, 20), (20, 40), (40, 50), (50, 60), (60, 80), (80, 101)]:
        m = yield_high & (move_pct >= lo) & (move_pct < hi)
        dd = px.index[m.values & ok.values]
        sizes.append(len(declusters(dd, max(h, 10), px.index)))
    maxes = []
    for _ in range(20000):
        perm = rng.permutation(parent_vals)
        i, best = 0, -np.inf
        for s in sizes:
            if s >= 2:
                best = max(best, perm[i:i + s].mean() - parent_mean)
            i += s
        maxes.append(best)
    maxes = np.asarray(maxes)
    print(f"  h={h}: band sizes {sizes} (parent episodes {len(parent_e)}); "
          f"observed live-band gate {100*obs:+.3f}pp; "
          f"P(max-of-6 gate >= observed) = {(maxes >= obs).mean():.4f}")

# ------------------------------------------------------------------ 5
print("\n=== 5. cost bar on the live band (TLT 3 bps round trip, 5x bar) ===")
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    e = declusters(px.index[band.values & r.notna().values], max(h, 10), px.index)
    v = r.loc[e].values
    bps = 100 * np.nanmean(v) * 100
    print(f"  h={h}: {bps:+.1f} bps -> {bps/3.0:.1f}x cost; "
          f"drop-best-1 {100*np.mean(np.sort(v)[:-1])*100:+.1f} bps -> "
          f"{100*np.mean(np.sort(v)[:-1])*100/3.0:.1f}x; "
          f"sign record {int((v>0).sum())}-{int((v<=0).sum())} "
          f"p={sign_test(int((v>0).sum()), len(v)):.4f}")

# ------------------------------------------------------------------ 6
print("\n=== 6. midterm + era on the live band ===")
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    e = declusters(px.index[band.values & r.notna().values], max(h, 10), px.index)
    v = r.loc[e].values
    mid = (e.year % 4 == 2)
    show([summarize(v[mid], f"midterm N={int(mid.sum())}"),
          summarize(v[~mid], f"non-midterm N={int((~mid).sum())}")] +
         era_split(e, v), f"live band TLT h={h}")

# ------------------------------------------------------------------ 7
print("\n=== 7. dial out-of-sample check on the pitched cell ===")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean()
cell = yield_high & (move_pct <= 50)
for name, m in [("pitched cell (MOVE<=50)", cell), ("live band [40,50)", band)]:
    dd = pd.DatetimeIndex(px.index[m.values]).intersection(ma.dropna().index)
    if len(dd) == 0:
        print(f"  {name}: ZERO trigger days have a dial reading at all "
              f"(dial starts {ma.dropna().index[0].date()})")
        continue
    vals = ma.loc[dd]
    print(f"  {name}: {len(dd)} of {int(m.sum())} trigger days have a dial; "
          f"max {vals.max():.1f}, median {vals.median():.1f} vs TODAY 87.6")
