"""rdlab_probe.py — redundancy probe on the fragility dials.

Vintages:
- data/rd2_fragility.parquet: APPEND-ONLY PIT since 2026-07-02; pre-2026-07-02
  rows are a recompute vintage (backfilled at freeze).
- data/rd2_fragility_ts.parquet: full recompute (2026-05-07 vintage).
- scratch/pit_signals.pkl: fires + frag_df_current are RECOMPUTE vintage
  (current code/params on full history). All results carry the PIT caveat.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ts = pd.read_parquet(ROOT / "data" / "rd2_fragility_ts.parquet")
pk = pickle.load(open(ROOT / "scratch" / "pit_signals.pkl", "rb"))
fires: pd.DataFrame = pk["fires"]
frag_cur: pd.DataFrame = pk["frag_df_current"]
hstats = pk["horizon_stats_current"]["signals"]

H = ["5d", "21d", "63d"]

print("=" * 70)
print("(1) HORIZON REDUNDANCY  [rd2_fragility.parquet: PIT>=2026-07-02, recompute before]")
print("=" * 70)
for name, df in [("append-only", frag), ("full recompute (ts)", frag_ts)]:
    c = df[H].corr()
    print(f"\n{name}  n={len(df)}  pearson:")
    print(c.round(3).to_string())
    cs = df[H].corr(method="spearman")
    print("spearman:")
    print(cs.round(3).to_string())

# dial vs own MAs
print("\nDial vs own 10d MA (append-only):")
for h in H:
    ma = frag[h].rolling(10).mean()
    print(f"  {h}: corr(raw, 10dMA) = {frag[h].corr(ma):.3f}")
print("Cross: 63d raw vs 21d 10dMA:", round(frag["63d"].corr(frag["21d"].rolling(10).mean()), 3))
print("Cross: 63d 10dMA vs 21d 10dMA:",
      round(frag["63d"].rolling(10).mean().corr(frag["21d"].rolling(10).mean()), 3))

# PCA variance explained
X = frag[H].dropna()
Xz = (X - X.mean()) / X.std()
ev = np.linalg.eigvalsh(np.cov(Xz.T))[::-1]
print(f"\nPCA on 3 dials (z-scored): PC1 explains {ev[0]/ev.sum()*100:.1f}%, "
      f"PC2 {ev[1]/ev.sum()*100:.1f}%, PC3 {ev[2]/ev.sum()*100:.1f}%")

# agreement on the >=50 state
for h in ["5d", "21d"]:
    a = (frag[h] >= 50)
    b = (frag["63d"] >= 50)
    print(f"{h}>=50 vs 63d>=50 agreement: {(a == b).mean()*100:.1f}%  "
          f"({h} active {a.mean()*100:.1f}%, 63d active {b.mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("(2) 63d DIAL / 10d-MA DISTRIBUTION + FAMILY4 BAND FLIPS  [same vintage caveat]")
print("=" * 70)
d63 = frag["63d"]
ma10 = d63.rolling(10).mean()
print("63d raw pctiles:", {p: round(float(np.nanpercentile(d63, p)), 1) for p in [5, 25, 50, 75, 90, 95, 99]})
print("63d 10dMA pctiles:", {p: round(float(np.nanpercentile(ma10.dropna(), p)), 1) for p in [50, 75, 90, 95, 99]})
band = (ma10 >= 50).astype(int)
tbl = pd.DataFrame({"ma10": ma10, "on": band})
tbl["year"] = tbl.index.year
g = tbl.dropna().groupby("year")
out = pd.DataFrame({
    "days": g.size(),
    "pct_days_on": (g["on"].mean() * 100).round(1),
    "flips": g["on"].apply(lambda s: int(s.diff().abs().sum())),
})
print("\nFAMILY4 band (10dMA of 63d >= 50) by year:")
print(out.to_string())
runs = band.dropna()
chg = runs.diff().fillna(0) != 0
run_id = chg.cumsum()
on_runs = runs.groupby(run_id).agg(["first", "size"])
on_lens = on_runs.loc[on_runs["first"] == 1, "size"]
print(f"\nTotal ON episodes: {len(on_lens)}, median len {on_lens.median():.0f}d, "
      f"mean {on_lens.mean():.1f}d, min {on_lens.min()}d, max {on_lens.max()}d")
print(f"ON episodes lasting <=5 days (whipsaw): {(on_lens <= 5).sum()}")
print(f"Overall time ON: {band.mean()*100:.1f}%")

print("\n" + "=" * 70)
print("(3) SIMPLIFICATION: k-of-n COUNT & MAX-SIGNAL vs WEIGHTED COMPOSITE")
print("[fires + frag_df_current = RECOMPUTE vintage from pit_signals.pkl, to 2026-07-02]")
print("=" * 70)
idx = fires.index.intersection(frag_cur.index)
f = fires.loc[idx]
comp = frag_cur.loc[idx]
count = f.sum(axis=1)
print("Unweighted active-signal count: distribution",
      count.value_counts().sort_index().to_dict())
for h in H:
    print(f"  count vs weighted {h}: pearson {count.corr(comp[h]):.3f}, "
          f"spearman {count.corr(comp[h], method='spearman'):.3f}")

# max-signal: strongest active signal's |diff_mean| weight per horizon
for h in H:
    w = {}
    for sig in f.columns:
        st = hstats.get(sig, {}).get("horizons", {}).get(h)
        w[sig] = abs(st["diff_mean"]) if st else 0.0
    wser = pd.Series(w)
    mx = f.mul(wser, axis=1).max(axis=1)
    print(f"  max-active-|diff_mean| vs weighted {h}: pearson {mx.corr(comp[h]):.3f}, "
          f"spearman {mx.corr(comp[h], method='spearman'):.3f}")

# does count reproduce the FAMILY4 gate?
ma63 = comp["63d"].rolling(10).mean()
gate = (ma63 >= 50)
cma = count.rolling(10).mean()
best = None
for thr in np.arange(0.5, 3.01, 0.1):
    g2 = cma >= thr
    agr = (g2 == gate).mean()
    if best is None or agr > best[1]:
        best = (thr, agr)
print(f"\nFAMILY4 gate replication by 10dMA of raw count: best threshold "
      f"{best[0]:.1f} signals -> {best[1]*100:.1f}% day agreement "
      f"(gate ON {gate.mean()*100:.1f}% of days, count-gate ON {(cma>=best[0]).mean()*100:.1f}%)")
both_on = (gate & (cma >= best[0])).sum()
print(f"  overlap: gate ON days captured by count-gate: {both_on}/{gate.sum()} "
      f"= {both_on/max(gate.sum(),1)*100:.1f}%")

# weighted composite vs count, per-signal marginal weight spread
print("\nEffective per-signal weights (|diff_mean|, current stats):")
for h in H:
    row = {s: abs(hstats.get(s, {}).get("horizons", {}).get(h, {"diff_mean": 0})["diff_mean"])
           for s in f.columns}
    tot = sum(row.values())
    print(f"  {h}: " + ", ".join(f"{s.split()[0]}={v/tot*100:.0f}%" for s, v in row.items()))
