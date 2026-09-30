import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from a7_knn import F, F8, zstd, knn, eligible, grid, PROX, FWD  # noqa: F401

PX10 = [p for p in PROX if p != "SVXY"]
rng = np.random.default_rng(11)


def rand_picks(pool, k):
    pk = []
    for p in rng.permutation(pool):
        if all(abs(p - q) >= 10 for q in pk):
            pk.append(p)
            if len(pk) == k:
                break
    return np.array(pk)


def run(label, Z, k, cells):
    ap = len(Z) - 1
    ok = eligible(Z, anchor_pos=ap)
    pk, _ = knn(Z, ap, k)
    g = grid(pk, ok)
    g = g[g.px.isin(PX10)].dropna(subset=["mean"])
    live_t, live_c = g.edge_t.abs().max(), g.cons.max()
    pool = np.where(ok)[0]
    nt, nc, cellnull = [], [], {c: [] for c in cells}
    for _ in range(1000):
        rp = rand_picks(pool, k)
        gr = grid(rp, ok)
        gr = gr[gr.px.isin(PX10)].dropna(subset=["mean"])
        nt.append(gr.edge_t.abs().max())
        nc.append(gr.cons.max())
        for (t, h) in cells:
            cellnull[(t, h)].append(np.nanmean(FWD[h][t].values[rp]))
    print(f"\n[{label}] k={k} pool {ok.sum()} days from {Z.index[ok][0].date()}: max|edge_t| {live_t:.2f} "
          f"(P null {np.mean(np.array(nt) >= live_t):.3f}), max cons {live_c:.2f} (P null {np.mean(np.array(nc) >= live_c):.3f}), "
          f"neg-edge cells {100*(g.edge<0).mean():.0f}%")
    for (t, h) in cells:
        r = g[(g.px == t) & (g.h == h)].iloc[0]
        base = FWD[h][t].values[ok]
        base = base[~np.isnan(base)]
        bh = (base > 0).mean()
        w = int(r.rec.split("-")[0])
        n = int(r.n)
        print(f"   {t} h{h}: {r.rec} mean {r['mean']:+.3f}% drift {r.drift:+.3f}% edge {r.edge:+.3f}pp | "
              f"sign p vs own up-rate {bh:.2f}: {sign_test(w, n, bh):.3f} | single-cell random P(mean>=live) "
              f"{np.mean(np.array(cellnull[(t, h)]) >= r['mean']/100):.3f}")


cells = [("EFA", 10), ("HYG", 10), ("IWM", 10), ("SPY", 10)]
Z7 = zstd(F)
run("7-feature", Z7, 10, cells)
run("7-feature", Z7, 20, cells)
run("7-feature", Z7, 30, cells)
Z8 = zstd(F8)
run("8-feature +frag ma10(63d), 2016+", Z8, 20, cells)
print("\nfrag ma10(63d) today", round(F8['frag_ma10'].iloc[-1], 1), "(the state file's main_score reads 80.9)")
