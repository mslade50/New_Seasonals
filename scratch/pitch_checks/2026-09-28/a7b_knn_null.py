import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from a7_knn import F, F8, zstd, knn, eligible, grid, PROX, FWD  # noqa: F401

pd.set_option("display.width", 250)
Z = zstd(F)
ap = len(Z) - 1
ok = eligible(Z, anchor_pos=ap)
live_picks, _ = knn(Z, ap, 20)
live = grid(live_picks, ok)
cells = live.dropna(subset=["mean"])
PX10 = [p for p in PROX if p != "SVXY"]  # ten class proxies (^VIX is the vol proxy)
g10 = cells[cells.px.isin(PX10)]
live_max_t = g10.edge_t.abs().max()
live_max_cons = g10.cons.max()
print(f"live grid (10 proxies x 2 h = {len(g10)} cells): max|edge_t| {live_max_t:.2f}, max sign consistency "
      f"{live_max_cons:.2f}, share of cells with negative edge {100*(g10.edge < 0).mean():.0f}%")

# 1) random-date null: 20 declustered random dates from the same eligible pool
rng = np.random.default_rng(7)
pool = np.where(ok)[0]
null_t, null_cons, null_iwm10 = [], [], []
fw10 = FWD[10]["IWM"].values
for _ in range(1500):
    sh = rng.permutation(pool)
    pk = []
    for p in sh:
        if all(abs(p - q) >= 10 for q in pk):
            pk.append(p)
            if len(pk) == 20:
                break
    pk = np.array(pk)
    g = grid(pk, ok)
    g = g[g.px.isin(PX10)].dropna(subset=["mean"])
    null_t.append(g.edge_t.abs().max())
    null_cons.append(g.cons.max())
    null_iwm10.append(np.nanmean(fw10[pk]))
null_t, null_cons, null_iwm10 = map(np.array, (null_t, null_cons, null_iwm10))
print(f"random-20-date null (1500 draws): P(max|edge_t| >= live) = {(null_t >= live_max_t).mean():.3f}; "
      f"P(max cons >= live) = {(null_cons >= live_max_cons).mean():.3f}")
iwm10_live = g10[(g10.px == 'IWM') & (g10.h == 10)]["mean"].iloc[0] / 100
print(f"IWM h=10 single-cell random-20 null: P(mean >= live {100*iwm10_live:+.3f}%) = {(null_iwm10 >= iwm10_live).mean():.3f}")

# 2) reference class: random anchors' own kNN grids (does today's grid stand out?)
ref_t, ref_neg = [], []
anchors = rng.choice(pool[pool < len(Z) - 80], size=150, replace=False)
for a in anchors:
    pk, _ = knn(Z, a, 20)
    oka = eligible(Z, anchor_pos=a)
    g = grid(pk, oka)
    g = g[g.px.isin(PX10)].dropna(subset=["mean"])
    ref_t.append(g.edge_t.abs().max())
    ref_neg.append((g.edge < 0).mean())
ref_t = np.array(ref_t)
print(f"reference class (150 random anchors' k=20 grids): P(best|edge_t| >= live) = {(ref_t >= live_max_t).mean():.3f}; "
      f"median share negative-edge cells {100*np.median(ref_neg):.0f}%")

# 3) k sensitivity and drop-one-feature, on the most consistent cells
focus = [("IWM", 10), ("HYG", 10), ("EFA", 10), ("SLV", 5), ("SPY", 5)]
rows = []
for k in (10, 20, 30):
    pk, _ = knn(Z, ap, k)
    g = grid(pk, ok)
    for t, h in focus:
        r = g[(g.px == t) & (g.h == h)].iloc[0]
        rows.append({"set": f"k={k}", "cell": f"{t} h{h}", "rec": r.rec, "mean": round(r["mean"], 3),
                     "edge": round(r.edge, 3), "overlap_w_live": len(set(pk) & set(live_picks))})
for c in F.columns:
    cols = [x for x in F.columns if x != c]
    pk, _ = knn(Z, ap, 20, cols=cols)
    g = grid(pk, ok)
    for t, h in focus:
        r = g[(g.px == t) & (g.h == h)].iloc[0]
        rows.append({"set": f"drop {c}", "cell": f"{t} h{h}", "rec": r.rec, "mean": round(r["mean"], 3),
                     "edge": round(r.edge, 3), "overlap_w_live": len(set(pk) & set(live_picks))})
# 8-feature (with fragility ma10, 2016-07+ coverage, recompute vintage pre-2026-07-02)
Z8 = zstd(F8)
pk8, d8 = knn(Z8, ap, 20)
ok8 = eligible(Z8, anchor_pos=ap)
g8 = grid(pk8, ok8)
for t, h in focus:
    r = g8[(g8.px == t) & (g8.h == h)].iloc[0]
    rows.append({"set": "8-feat +frag (2016+)", "cell": f"{t} h{h}", "rec": r.rec, "mean": round(r["mean"], 3),
                 "edge": round(r.edge, 3), "overlap_w_live": len(set(pk8) & set(live_picks))})
out = pd.DataFrame(rows)
print(out.pivot_table(index="set", columns="cell", values="rec", aggfunc="first").to_string())
print(out.pivot_table(index="set", columns="cell", values="edge", aggfunc="first").round(3).to_string())
print(out.groupby("set").overlap_w_live.first().to_string())
print("\n8-feature grid, frag at neighbours:", F8["frag_ma10"].iloc[pk8].round(1).tolist())
print(g8[g8.px.isin(PX10)].round(3).to_string(index=False))
