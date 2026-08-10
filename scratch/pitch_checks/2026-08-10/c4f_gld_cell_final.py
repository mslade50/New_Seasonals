"""C4 round 3c -- the object that is actually left standing, attacked directly.

The GDX form is dead as specified: GDX-1.78*GLD residual on the cell is
+0.049% (t=0.07, 49.0% hit) at h=10, and GLD beats GDX on mean/sd at every
horizon.  So the surviving object is LONG GLD, anchored CPI-3 (= today), on a
metals thrust.  The registry kills "GLD into CPI", so this has to clear a
higher bar than anything else checked today.

Attacks: (a) is the GDX-rank trigger a cross-instrument fishing artefact, or
does GLD's OWN rank work, (b) month-position + disjoint-anchor placebos on
GLD directly, (c) era / cycle / LOYO / concentration, (d) is this just the
registry's dead unconditioned cell wearing a conditioner, (e) cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG, K = 1, 3
P = close_panel(["GDX", "GLD", "SLV"]).dropna(subset=["GLD"])
g, gl = P["GDX"], P["GLD"]
idx = gl.index
rk5g, rk5l = pct_rank(g, 5), pct_rank(gl, 5)
ASOF = idx[-1]
print(f"sample {idx.min().date()} .. {ASOF.date()}  n={len(idx)}")
print(f"TODAY: GLD rank5={rk5l.loc[ASOF]:.1f}  GDX rank5={rk5g.loc[ASOF]:.1f}")


def anchor_k(kind: str, k: int) -> pd.DatetimeIndex:
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


A3 = anchor_k("cpi", K)

# ---------------------------------------------- a. which rank is the trigger?
print("\n### a. GDX-rank trigger vs GLD's OWN rank (cross-instrument fishing check) ###")
rows = []
for H in (3, 5, 10):
    fw = fwd_lag(gl, H, LAG)
    ok = fw.notna()
    V = idx[ok.values]
    for lbl, m in (("GDX rank5>=80", (rk5g >= 80).fillna(False)),
                   ("GLD rank5>=80", (rk5l >= 80).fillna(False)),
                   ("GLD rank5>=90", (rk5l >= 90).fillna(False)),
                   ("EITHER rank>=80", ((rk5g >= 80) | (rk5l >= 80)).fillna(False)),
                   ("no thrust gate", pd.Series(True, index=idx))):
        t = pd.DatetimeIndex(A3).intersection(idx[m.reindex(idx).fillna(False).values & ok.values])
        if len(t) < 5:
            rows.append({"label": f"{lbl} h={H}", "n": len(t)})
            continue
        e = declusters(t, H, V)
        v = fw.loc[e].values
        r = summarize(v, f"{lbl} h={H}")
        r["edge"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
show(rows, "GLD forward return by trigger definition (anchor CPI-3)")

# ---------------------------------------------- b/c. full drill on GLD-own-rank
for H in (5, 10):
    print(f"\n{'='*70}\n### DRILL: LONG GLD, anchor CPI-3, GLD rank5>=80, h={H} ###")
    fw = fwd_lag(gl, H, LAG)
    ok = fw.notna()
    V = idx[ok.values]
    M = (rk5l >= 80).fillna(False)
    T = pd.DatetimeIndex(A3).intersection(idx[M.values & ok.values])
    e = declusters(T, H, V)
    v = fw.loc[e].values
    tstat = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    print(f"N={len(v)} mean={100*v.mean():+.3f}% med={100*np.median(v):+.3f}% "
          f"hit={100*(v>0).mean():.1f}% t={tstat:.2f} "
          f"sign_p={sign_test(int((v>0).sum()), len(v)):.5f} boot_p={bootstrap_p_le0(v):.4f}")
    print(f"  GLD own drift all days  = {100*fw[ok].mean():+.3f}% -> edge "
          f"{100*(v.mean()-fw[ok].mean()):+.3f}pp")
    print(f"  ALL CPI-3, no thrust    = "
          f"{100*fw.loc[declusters(pd.DatetimeIndex(A3).intersection(V), H, V)].mean():+.3f}%  "
          f"<- the registry's dead unconditioned cell")
    print(f"  thrust, NOT on CPI-3    = "
          f"{100*fw.loc[declusters(idx[M.values & ok.values].difference(A3), H, V)].mean():+.3f}%")
    print(f"  worst episode           = {100*v.min():+.2f}%")
    print("  concentration:", cluster_note(e, v, k=2))
    o = np.argsort(v)
    print(f"  drop-2-best {100*np.delete(v, o[-2:]).mean():+.3f}%  "
          f"drop-2-worst {100*np.delete(v, o[:2]).mean():+.3f}%")
    print("  worst 3:", [(str(pd.Timestamp(e[i]).date()), round(100*v[i], 2)) for i in o[:3]])
    show(era_split(e, v), "era")
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(v[mt], f"MIDTERM (N={int(mt.sum())})"), summarize(v[~mt], "non-midterm")],
         "cycle -- 2026 IS midterm")
    yrs = pd.DatetimeIndex(e).year
    loyo = [(int(y), round(100 * v[yrs.values != y].mean(), 3)) for y in sorted(set(yrs))]
    print("  LOYO min:", min(x[1] for x in loyo), "over", len(loyo), "years;",
          "positive years:", int((pd.Series(v).groupby(yrs.values).mean() > 0).sum()),
          "/", len(set(yrs)))
    print("  LOYO:", loyo)
    # placebos
    dom = pd.Series(0, index=idx)
    cur, lm = 0, None
    for i, d in enumerate(idx):
        if lm != (d.year, d.month):
            cur, lm = 0, (d.year, d.month)
        cur += 1
        dom.iloc[i] = cur
    tp = dom.loc[T]
    lo, hi = int(tp.quantile(.10)), int(tp.quantile(.90))
    matched = pd.DatetimeIndex(idx[M.values & ok.values
                                   & ((dom >= lo) & (dom <= hi)).values]).difference(A3)
    rows = [summarize(v, "the cell"),
            summarize(fw.loc[declusters(matched, H, V)].values,
                      f"month-pos matched bd{lo}-{hi}, thrust, NOT CPI-3")]
    for kind in ("ppi", "nfp", "opex", "fomc_decision"):
        a = pd.DatetimeIndex(anchor_k(kind, K)).intersection(
            idx[M.values & ok.values]).difference(A3)
        rows.append(summarize(fw.loc[declusters(a, H, V)].values, f"{kind}-3 ex-CPI x thrust"))
    show(rows, "placebos")
    # definition neighbours
    rows = []
    for thr in (60.0, 70.0, 80.0, 85.0, 90.0, 95.0):
        m2 = (rk5l >= thr).fillna(False)
        t2 = pd.DatetimeIndex(A3).intersection(idx[m2.values & ok.values])
        e2 = declusters(t2, H, V)
        v2 = fw.loc[e2].values
        r = summarize(v2, f"GLD rank5>={thr:.0f}")
        r["sign_p"] = round(sign_test(int((v2 > 0).sum()), len(v2)), 4)
        rows.append(r)
    show(rows, "definition neighbours (GLD's own rank)")
    print(f"  cost: GLD ~1 bp/side -> 2 bps rt; edge {100*v.mean()*100:.0f} bps "
          f"-> {100*v.mean()*100/2:.0f}x")
    # loser paths
    paths = episode_paths(P, e, [("GLD", 1.0)], H, LAG)
    d1 = paths[1].values
    print(f"  P(loses | day1 <= -0.5%) = {100*(v[d1 <= -0.005] < 0).mean():.0f}% "
          f"(n={int((d1 <= -0.005).sum())}), mean {100*v[d1 <= -0.005].mean():+.2f}%")
    print(f"  P(loses | day1 > 0)      = {100*(v[d1 > 0] < 0).mean():.0f}% "
          f"(n={int((d1 > 0).sum())}), mean {100*v[d1 > 0].mean():+.2f}%")

# ---------------------------------------------- e. today's cluster depth
print(f"\n### today's cluster depth on the GLD trigger ###")
M = (rk5l >= 80).fillna(False)
p = list(idx).index(ASOF)
d = 0
while p - d >= 0 and bool(M.iloc[p - d]):
    d += 1
runs, run = [], 0
for x in M.values:
    run = run + 1 if x else 0
    if x:
        runs.append(run)
runs = np.array(runs)
dep = pd.Series(runs, index=idx[M.values])
H = 10
fw = fwd_lag(gl, H, LAG)
ok = fw.notna()
T = pd.DatetimeIndex(A3).intersection(idx[M.values & ok.values])
td = dep.reindex(T).values
print(f"  today depth={d}; trigger population p50={np.median(td):.0f} mean={td.mean():.1f}")
show([summarize(fw.loc[pd.DatetimeIndex(T)[td <= 3]].values, "trigger depth<=3"),
      summarize(fw.loc[pd.DatetimeIndex(T)[td > 3]].values, "trigger depth>3")],
     "depth split (h=10)")
r5 = gl.pct_change(5) * 100
print(f"  today's GLD 5d = {r5.loc[ASOF]:+.2f}%; cell population p50={np.median(r5.loc[T]):+.2f}% "
      f"p90={np.percentile(r5.loc[T],90):+.2f}% max={r5.loc[T].max():+.2f}%")
