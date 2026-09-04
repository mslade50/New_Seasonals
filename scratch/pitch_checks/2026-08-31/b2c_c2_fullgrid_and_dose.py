"""C2 round 3 -- the honest multiplicity charge and the dose response.

Round 2 left one thing standing: the LIVE MOVE band [40,50) crossed with the
yield-high parent pays +1.064% (h=5, 7-0) / +1.881% (h=10, 6-1), 35x/63x cost,
plateau-stable across the YIELD gate, and clears a max-of-6-bands permutation
at P=0.029/0.012.

Two things it has not yet been charged for:
  (a) the candidate explicitly scanned FOUR sign/vehicle combinations and this
      script scanned SIX bands x FOUR horizons. The permutation in round 2
      charged the bands only.
  (b) the MOVE ladder is an INVERTED U, not monotone, so the stated mechanism
      ("compressed = orderly = trends") predicts the wrong shape. If the real
      object is "MOVE not in either tail", the honest cell is [40,80) and the
      number falls by half. Both readings are priced here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "^MOVE", "TLT", "IEF", "LQD", "SPY"])
tnx, move = px["^TNX"], px["^MOVE"]
tnx_dist = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
move_pct = rolling_on_valid(move, lambda x: x.rolling(252).apply(
    lambda w: 100.0 * (w <= w[-1]).mean(), raw=True))
usable = tnx_dist.notna() & move_pct.notna()
yield_high = (tnx_dist >= -0.01) & usable
BANDS = [(0, 20), (20, 40), (40, 50), (50, 60), (60, 80), (80, 101)]
HS = (3, 5, 10, 21)

# ------------------------------------------------------------------ 1
print("=== 1. FULL-GRID PERMUTATION: max |gate| over 6 bands x 4 horizons "
      "x 2 vehicles (long/short are mirrors, so |.| covers the sign scan) ===")
rng = np.random.default_rng(7)
obs_grid, null_draws = {}, []
per_h = {}
for veh in ("TLT", "IEF"):
    for h in HS:
        r = fwd_lag(px[veh], h, 1)
        ok = r.notna()
        pe = declusters(px.index[yield_high.values & ok.values], max(h, 10), px.index)
        pv = r.loc[pe].values
        pm = np.nanmean(pv)
        sizes = []
        for lo, hi in BANDS:
            m = yield_high & (move_pct >= lo) & (move_pct < hi)
            sizes.append(len(declusters(px.index[m.values & ok.values],
                                        max(h, 10), px.index)))
        per_h[(veh, h)] = (pv, pm, sizes)
        m = yield_high & (move_pct >= 40) & (move_pct < 50)
        be = declusters(px.index[m.values & ok.values], max(h, 10), px.index)
        obs_grid[(veh, h)] = np.nanmean(r.loc[be].values) - pm

live = obs_grid[("TLT", 5)]
print("  observed live-band gates (pp):",
      {f"{v} h={h}": round(100 * g, 3) for (v, h), g in obs_grid.items()})
print(f"  the pitched number is TLT h=5 gate = {100*live:+.3f}pp "
      f"(TLT h=10 = {100*obs_grid[('TLT',10)]:+.3f}pp)")

NB = 20000
maxes = np.zeros(NB)
for b in range(NB):
    best = 0.0
    for key, (pv, pm, sizes) in per_h.items():
        perm = rng.permutation(pv)
        i = 0
        for s in sizes:
            if s >= 2:
                best = max(best, abs(perm[i:i + s].mean() - pm))
            i += s
    maxes[b] = best
for lbl, o in [("TLT h=5", obs_grid[("TLT", 5)]),
               ("TLT h=10", obs_grid[("TLT", 10)])]:
    print(f"  P(max|gate| over the 6x4x2 grid >= |{lbl} observed|) = "
          f"{(maxes >= abs(o)).mean():.4f}")

# ------------------------------------------------------------------ 2
print("\n=== 2. DOSE RESPONSE: is the MOVE percentile a monotone conditioner? ===")
for veh in ("TLT",):
    for h in (5, 10):
        r = fwd_lag(px[veh], h, 1)
        ok = r.notna()
        e = declusters(px.index[yield_high.values & ok.values], max(h, 10), px.index)
        y = r.loc[e].values
        x = move_pct.loc[e].values
        good = ~np.isnan(y) & ~np.isnan(x)
        y, x = y[good], x[good]
        sp = pd.Series(y).corr(pd.Series(x), method="spearman")
        b1 = np.polyfit(x, y, 1)
        b2 = np.polyfit(x, y, 2)
        print(f"  {veh} h={h}: N={len(y)}  Spearman(MOVE pctile, fwd ret) = "
              f"{sp:+.3f}   linear slope {100*b1[0]:+.4f}pp/pctile   "
              f"quadratic peak at pctile "
              f"{-b2[1]/(2*b2[0]) if b2[0] else float('nan'):.1f}")
        print("     -> the stated mechanism needs a NEGATIVE monotone slope "
              "(more compressed = more trend = worse for long duration).")

# ------------------------------------------------------------------ 3
print("\n=== 3. THE HONEST POOLED READING: 'MOVE not in either tail' [40,80) ===")
pooled = yield_high & (move_pct >= 40) & (move_pct < 80)
for h in (5, 10):
    r = fwd_lag(px["TLT"], h, 1)
    ok = r.notna()
    pe = declusters(px.index[yield_high.values & ok.values], max(h, 10), px.index)
    pm = 100 * np.nanmean(r.loc[pe].values)
    e = declusters(px.index[pooled.values & ok.values], max(h, 10), px.index)
    v = r.loc[e].values
    s = summarize(v, f"[40,80) h={h}")
    mid = (e.year % 4 == 2)
    show([s] + [summarize(v[mid], f"  midterm N={int(mid.sum())}"),
                summarize(v[~mid], f"  non-midterm")] + era_split(e, v),
         f"pooled 'not-extreme MOVE' vs yield-high parent {pm:+.3f}%")
    bps = 100 * np.nanmean(v) * 100
    print(f"  gate {s['mean_pct']-pm:+.3f}pp | {bps:+.1f} bps = {bps/3.0:.1f}x cost | "
          f"drop-best-2 {100*np.mean(np.sort(v)[:-2])*100:+.1f} bps | "
          f"record {int((v>0).sum())}-{int((v<=0).sum())} "
          f"p={sign_test(int((v>0).sum()), len(v), p=float((r[ok]>0).mean())):.4f} "
          f"(vs TLT's own {100*float((r[ok]>0).mean()):.1f}% up-rate)")

# ------------------------------------------------------------------ 4
print("\n=== 4. REFERENCE CLASS: is [40,50) x yield-high a RATES cell, or does "
      "the same state pay on everything? ===")
for h in (5, 10):
    rows = []
    for veh in ("TLT", "IEF", "LQD", "SPY"):
        r = fwd_lag(px[veh], h, 1)
        ok = r.notna()
        pe = declusters(px.index[yield_high.values & ok.values], max(h, 10), px.index)
        pm = np.nanmean(r.loc[pe].values)
        m = yield_high & (move_pct >= 40) & (move_pct < 50)
        e = declusters(px.index[m.values & ok.values], max(h, 10), px.index)
        v = r.loc[e].values
        alld = float(r[ok].mean())
        rows.append({"veh": veh, "n": len(v), "band_pct": round(100*np.nanmean(v), 3),
                     "parent_pct": round(100*pm, 3), "alldays_pct": round(100*alld, 3),
                     "gate_pp": round(100*(np.nanmean(v)-pm), 3),
                     "vs_alldays_pp": round(100*(np.nanmean(v)-alld), 3),
                     "sd_pct": round(100*float(r[ok].std()), 3),
                     "excess_per_sd": round((np.nanmean(v)-alld)/float(r[ok].std()), 4)})
    show(rows, f"h={h}: live band across vehicles (duration proportionality)")

# ------------------------------------------------------------------ 5
print("\n=== 5. dial, excluding today's own trigger ===")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean().dropna()
for name, m in [("pitched cell MOVE<=50", yield_high & (move_pct <= 50)),
                ("live band [40,50)", yield_high & (move_pct >= 40) & (move_pct < 50))]:
    dd = pd.DatetimeIndex(px.index[m.values]).intersection(ma.index)
    dd = dd[dd < pd.Timestamp("2026-01-01")]
    if len(dd) == 0:
        print(f"  {name}: ZERO pre-2026 trigger days carry a dial reading.")
        continue
    v = ma.loc[dd]
    print(f"  {name}: {len(dd)} pre-2026 trigger days with a dial; "
          f"max {v.max():.1f}, p90 {v.quantile(.9):.1f}, median {v.median():.1f} "
          f"vs TODAY 87.6  -> today is {v.max() < 87.6 and 'OUTSIDE' or 'inside'} "
          f"the measured range")
