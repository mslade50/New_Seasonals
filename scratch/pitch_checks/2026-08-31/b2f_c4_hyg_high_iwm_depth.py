"""C4 -- HYG at a 52-week high while IWM sits >=2% below its own. Long IWM.

This is a deliberate RE-ASK of the cell killed on 2026-08-26 ("HYG at a fresh
52w high while SPY is off its own high"), with IWM substituted into the depth
slot because SPY at -1.10% is in that kill's DEAD band (-0.042pp at h=3 in the
1.0-2.0% band) while IWM at -3.06% is in its ALIVE band (+0.615pp at >=2%).

The question this script answers is NOT "does the IWM version work". It is
"is the substitution a real distinction or a re-skin of a dead cell". So:
  1. reproduce the SPY depth split exactly, to make the comparison apples to
     apples and confirm the 2026-08-26 numbers;
  2. run the IWM version with full gate attribution and a depth ladder marking
     today's -3.06%;
  3. run the REFERENCE CLASS (SPY, QQQ, DIA, IWM, EFA, EEM) BEFORE round 2,
     per the registry's instruction that this is now the modal kill;
  4. the 2026-08-31 dial objection: max historical dial on trigger days vs 87.6.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

INDEXES = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]
px = close_panel(["HYG"] + INDEXES)
idx = px.index


def dist_from_high(t):
    s = px[t]
    return s / rolling_on_valid(s, lambda x: x.rolling(252).max()) - 1.0


hyg_d = dist_from_high("HYG")
dists = {t: dist_from_high(t) for t in INDEXES}

print("=== 0. LIVE STATE (asof %s) ===" % idx[-1].date())
print(f"  HYG below its trailing-252 high: {100*hyg_d.iloc[-1]:+.3f}%")
for t in INDEXES:
    print(f"  {t:4s} below its own trailing-252 high: {100*dists[t].iloc[-1]:+.3f}%")

HYG_THR = -0.0025          # "within 0.25% of its trailing-252 high"
hyg_hi = hyg_d >= HYG_THR
print(f"\n  HYG within 0.25% of its 252d high: {int(hyg_hi.sum())} days "
      f"since {idx[hyg_hi][0].date()}")

# ------------------------------------------------------------------ 1
print("\n=== 1. REPRODUCE the 2026-08-26 SPY depth split (h=3, long SPY) ===")
DEPTH = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1.0)]
for tkr in ("SPY", "IWM"):
    for h in (3, 5, 10):
        r = fwd_lag(px[tkr], h, 1)
        ok = r.notna()
        parent_d = idx[hyg_hi.values & ok.values]
        parent_e = declusters(parent_d, max(h, 5), idx)
        pm = 100 * float(np.nanmean(r.loc[parent_e].values))
        alld = 100 * float(r[ok].mean())
        live = -100 * float(dists[tkr].iloc[-1])
        rows = []
        for lo, hi in DEPTH:
            m = hyg_hi & (dists[tkr] <= -lo) & (dists[tkr] > -hi)
            dd = idx[m.values & ok.values]
            if len(dd) < 2:
                rows.append({"label": f"{100*lo:.1f}-{100*hi:.1f}%", "n": 0}); continue
            e = declusters(dd, max(h, 5), idx)
            s = summarize(r.loc[e].values, f"{100*lo:.1f}-{100*hi:.1f}% below high")
            s["gate_pp"] = round(s["mean_pct"] - alld, 3)
            s["n_days"] = len(dd)
            s["LIVE"] = "<<<" if lo <= live / 100 * 100 / 100 * 100 / 100 else ""
            s["LIVE"] = "<<<" if (lo <= live / 100.0 < hi) else ""
            rows.append(s)
        show(rows, f"LONG {tkr} h={h}: HYG-high parent {pm:+.3f}%, all days "
                   f"{alld:+.3f}%, LIVE depth {live:.2f}%")

# ------------------------------------------------------------------ 2
print("\n=== 2. THE PITCHED CELL: HYG within 0.25% of high AND IWM >=2% below "
      "its own. GATE ATTRIBUTION. ===")
cell = hyg_hi & (dists["IWM"] <= -0.02)
iwm_deep = dists["IWM"] <= -0.02
print(f"  cell days: {int(cell.sum())}, years "
      f"{dict(pd.Series(1, index=idx[cell.values]).groupby(idx[cell.values].year).sum())}")
for h in (3, 5, 10, 21):
    r = fwd_lag(px["IWM"], h, 1)
    ok = r.notna()

    def mean_of(m):
        dd = idx[m.values & ok.values]
        if len(dd) < 2:
            return np.nan, 0
        e = declusters(dd, max(h, 5), idx)
        return 100 * float(np.nanmean(r.loc[e].values)), len(e)

    j, nj = mean_of(cell)
    p1, n1 = mean_of(hyg_hi)
    p2, n2 = mean_of(iwm_deep)
    alld = 100 * float(r[ok].mean())
    print(f"  h={h:2d}: joint {j:+.3f}% (N={nj:3d}) | HYG-high only {p1:+.3f}% "
          f"(N={n1:3d}) | IWM-deep only {p2:+.3f}% (N={n2:3d}) | all days "
          f"{alld:+.3f}%  -> beats both parents? "
          f"{'YES' if (j > p1 and j > p2) else 'NO'}   vs all days "
          f"{j-alld:+.3f}pp")

# ------------------------------------------------------------------ 3
print("\n=== 3. FULL BATTERY on the pitched cell (long IWM) ===")
variants = {
    "HYG<=0.10% + IWM>=2%": (hyg_d >= -0.001) & (dists["IWM"] <= -0.02),
    "HYG<=0.25% + IWM>=2%": cell,
    "HYG<=0.50% + IWM>=2%": (hyg_d >= -0.005) & (dists["IWM"] <= -0.02),
    "HYG<=1.00% + IWM>=2%": (hyg_d >= -0.01) & (dists["IWM"] <= -0.02),
    "HYG<=0.25% + IWM>=2.5%": hyg_hi & (dists["IWM"] <= -0.025),
    "HYG<=0.25% + IWM>=3.0%": hyg_hi & (dists["IWM"] <= -0.03),
    "HYG<=0.25% + IWM 2-5%": hyg_hi & (dists["IWM"] <= -0.02) & (dists["IWM"] > -0.05),
}
for h in (3, 5):
    battery(px, cell, [("IWM", 1.0)], h, "C4 LONG IWM (HYG high x IWM 2% off)",
            2.0, variants=variants, min_gap=max(h, 5),
            event_kinds=("cpi", "ppi", "fomc_decision"))

# ------------------------------------------------------------------ 4
print("\n=== 4. REFERENCE CLASS (run BEFORE round 2): the identical rule with "
      "each index in the depth slot ===")
for h in (3, 5):
    rows, ests, ses = [], [], []
    for t in INDEXES:
        r = fwd_lag(px[t], h, 1)
        ok = r.notna()
        m = hyg_hi & (dists[t] <= -0.02)
        dd = idx[m.values & ok.values]
        if len(dd) < 2:
            rows.append({"index": t, "n": 0}); continue
        e = declusters(dd, max(h, 5), idx)
        v = r.loc[e].values
        base = r[ok].values
        exc = v.mean() - base.mean()
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        ests.append(exc); ses.append(se)
        rows.append({"index": t, "n_epi": len(e), "n_days": len(dd),
                     "cell_pct": round(100 * v.mean(), 3),
                     "alldays_pct": round(100 * base.mean(), 3),
                     "excess_pp": round(100 * exc, 3),
                     "se_pp": round(100 * se, 3),
                     "t": round(exc / se, 2),
                     "hit": round(100 * (v > 0).mean(), 1)})
    show(rows, f"h={h} reference class: HYG-high x <index> 2% off its own high")
    ests, ses = np.asarray(ests), np.asarray(ses)
    w = 1 / ses ** 2
    fe = (w * ests).sum() / w.sum()
    Q = (w * (ests - fe) ** 2).sum()
    df = len(ests) - 1
    I2 = max(0.0, 100 * (Q - df) / Q) if Q > 0 else 0.0
    print(f"  fixed-effect common excess {100*fe:+.4f}pp | Cochran Q {Q:.2f} on "
          f"{df} df | I-squared {I2:.1f}% | observed cross-sectional sd "
          f"{100*ests.std(ddof=1):.3f}pp vs mean sampling SE "
          f"{100*ses.mean():.3f}pp (ratio {ests.std(ddof=1)/ses.mean():.2f})")
    # max-of-6 permutation on random anchor dates
    rng = np.random.default_rng(19)
    n_epi = rows[INDEXES.index("IWM")].get("n_epi", 0)
    mx = []
    for _ in range(5000):
        best = -np.inf
        for t in INDEXES:
            r = fwd_lag(px[t], h, 1)
            base = r.dropna()
            if len(base) <= n_epi or n_epi < 2:
                continue
            samp = rng.choice(base.values, size=n_epi, replace=False)
            best = max(best, samp.mean() - base.values.mean())
        mx.append(best)
    obs_iwm = ests[INDEXES.index("IWM")] if len(ests) == len(INDEXES) else np.nan
    print(f"  random-date max-of-6 at N={n_epi}: P(max >= IWM's "
          f"{100*obs_iwm:+.3f}pp) = {(np.asarray(mx) >= obs_iwm).mean():.4f}")

# ------------------------------------------------------------------ 5
print("\n=== 5. ERA + MIDTERM + local control + cost (long IWM) ===")
for h in (3, 5):
    r = fwd_lag(px["IWM"], h, 1)
    ok = r.notna()
    dd = idx[cell.values & ok.values]
    e = declusters(dd, max(h, 5), idx)
    v = r.loc[e].values
    mid = (e.year % 4 == 2)
    loc = local_control(idx[ok.values], dd, 126)
    show([summarize(v, f"cell episodes N={len(e)}"),
          summarize(r.loc[loc].values, "local +/-126td ex-trigger"),
          summarize(v[mid], f"MIDTERM N={int(mid.sum())}"),
          summarize(v[~mid], "non-midterm")] + era_split(e, v),
         f"LONG IWM h={h}")
    bps = 100 * v.mean() * 100
    print(f"  cost: {bps:+.1f} bps -> {bps/2.0:.1f}x a 2 bp IWM round trip; "
          f"drop-best-2 {100*np.mean(np.sort(v)[:-2])*100:+.1f} bps -> "
          f"{100*np.mean(np.sort(v)[:-2])*100/2.0:.1f}x")
    print("  concentration:", cluster_note(e, v, k=3))
    print("  years:", {int(y): round(100*x, 2) for y, x in
                       pd.Series(v, index=e.year).groupby(level=0).mean().items()})

# ------------------------------------------------------------------ 6
print("\n=== 6. THE DIAL OBJECTION (registry 2026-08-26) ===")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean().dropna()
for name, m in [("SPY version (>=2% off)", hyg_hi & (dists["SPY"] <= -0.02)),
                ("IWM version (>=2% off)", cell)]:
    dd = pd.DatetimeIndex(idx[m.values]).intersection(ma.index)
    dd = dd[dd < pd.Timestamp("2026-08-01")]
    if len(dd) == 0:
        print(f"  {name}: no pre-Aug-2026 trigger day carries a dial reading")
        continue
    v = ma.loc[dd]
    print(f"  {name}: {len(dd)} of {int(m.sum())} trigger days have a dial "
          f"(dial starts {ma.index[0].date()}); max {v.max():.1f}, "
          f"p90 {v.quantile(.9):.1f}, median {v.median():.1f}  vs TODAY 87.6 "
          f"-> today is {'OUTSIDE' if v.max() < 87.6 else 'inside'} the range "
          f"by {87.6 - v.max():+.1f} points")
