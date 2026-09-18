"""C2 round 1: the leader's 63d floor inside a top-decile year while STILL
FALLING (W25 live leg), tested as a FAMILY with the r5<15 conditioner charged
as post hoc (found on SMH alone inside a kill report, N=7).

Definitions reused verbatim from W25 (2026-08-27 b2_c7_smh_refclass.py):
  base B  = pct_rank(63) <= 5 AND 252d return >= +40%
  cond C  = B AND pct_rank(5) < 15
Long the member, h=10, lag=1 (MOC tomorrow), declustered gap 10.

Questions:
 1. SMH reproduction on data through 2026-09-14 (both the W25 episode-split
    form and the mask form).
 2. The same conditioner on the 22 NON-SMH members (out of sample for the
    conditioner): pooled excess, C vs complement, heterogeneity.
 3. Shrunk family estimate for SMH -> cost multiple.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import (ASOF, REF23, cell_stats, cochran, date_clusters, fwd,  # noqa
                       rec, shrink, vret)

H, GAP = 10, 10
px = load_prices(REF23)


def masks(s):
    r63 = pct_rank(s, 63)
    r5 = pct_rank(s, 5)
    r252 = vret(s, 252)
    B = (r63 <= 5) & (r252 >= 0.40)
    return {"B": B, "C": B & (r5 < 15), "D25": B & (r5 >= 25),
            "Dge15": B & (r5 >= 15), "r5": r5, "r63": r63, "r252": r252}


rows, pool = [], {"C": [], "Dge15": [], "D25": [], "B": [], "Csplit": [], "Dsplit": []}
for t in REF23:
    s = px[t]["Close"].dropna()
    s = s[s.index <= ASOF]
    M = masks(s)
    r = fwd(s, H, 1)
    out = {"t": t}
    for k in ("B", "C", "Dge15", "D25"):
        cs = cell_stats(r, M[k], GAP)
        out[f"{k}_n"] = cs["n"]
        if cs["n"]:
            out[f"{k}_exc"] = cs["excess_pct"]
            out[f"{k}_se"] = cs["se_pct"]
            pool[k] += [(t, d, x, x - cs["_drift"]) for d, x in zip(cs["_dates"], cs["_vals"])]
    # W25's episode-split form: declustered B episodes split by r5 AT the episode date
    csB = cell_stats(r, M["B"], GAP)
    if csB["n"]:
        r5e = M["r5"].reindex(csB["_dates"]).values
        for d, x, q in zip(csB["_dates"], csB["_vals"], r5e):
            key = "Csplit" if q < 15 else "Dsplit"
            pool[key].append((t, d, x, x - csB["_drift"]))
    out["live_r63"] = M["r63"].iloc[-1]
    out["live_r5"] = M["r5"].iloc[-1]
    out["live_r252"] = 100 * M["r252"].iloc[-1]
    rows.append(out)

T = pd.DataFrame(rows).set_index("t")
print("=" * 78)
print(f"0. per-member cells, h={H}, gap {GAP}; excess = episode mean - own full drift (pp)")
print("=" * 78)
print(T.round(2).to_string())


def pool_summary(key, ex_smh=True, label=""):
    P = [p for p in pool[key] if (p[0] != "SMH" or not ex_smh)]
    if not P:
        return {"label": label, "n": 0}
    x = np.array([p[2] for p in P])
    e = np.array([p[3] for p in P])
    d = pd.DatetimeIndex([p[1] for p in P])
    s = summarize(e, label)
    s["raw_mean_pct"] = 100 * x.mean()
    cl, _ = date_clusters(d, e, 14)
    s["date_clusters"] = len(cl)
    s["clust_mean_exc_pct"] = 100 * cl.mean()
    s["clust_rec"] = rec(cl)
    s["names"] = len(set(p[0] for p in P))
    return s


print("\n" + "=" * 78)
print("1. SMH reproduction (data through 2026-09-14)")
print("=" * 78)
show([pool_summary(k, ex_smh=False, label=f"SMH {k}") if False else
      summarize(np.array([p[2] for p in pool[k] if p[0] == "SMH"]), f"SMH {k} raw")
      for k in ("B", "C", "Dge15", "D25", "Csplit", "Dsplit")])
for k in ("C", "Csplit"):
    v = np.array([p[2] for p in pool[k] if p[0] == "SMH"])
    dd = [str(p[1].date()) for p in pool[k] if p[0] == "SMH"]
    print(f"  SMH {k}: record {rec(v)}  dates {dd}")
    print(f"     vals {[round(100*x,2) for x in v]}")

print("\n" + "=" * 78)
print("2. EX-SMH pooled (the conditioner OUT OF SAMPLE), excess over own drift")
print("=" * 78)
show([pool_summary("B", label="ex-SMH B (base, all r5)"),
      pool_summary("C", label="ex-SMH C mask r5<15"),
      pool_summary("Dge15", label="ex-SMH B & r5>=15 mask"),
      pool_summary("D25", label="ex-SMH B & r5>=25 mask"),
      pool_summary("Csplit", label="ex-SMH B-episode r5<15 (W25 split form)"),
      pool_summary("Dsplit", label="ex-SMH B-episode r5>=15 (split form)")])
for kc, kd in (("C", "Dge15"), ("Csplit", "Dsplit")):
    c = np.array([p[3] for p in pool[kc] if p[0] != "SMH"])
    d = np.array([p[3] for p in pool[kd] if p[0] != "SMH"])
    if len(c) > 1 and len(d) > 1:
        se = np.sqrt(c.var(ddof=1) / len(c) + d.var(ddof=1) / len(d))
        print(f"  {kc} minus {kd}: {100*(c.mean()-d.mean()):+.3f}pp  welch t {(c.mean()-d.mean())/se:+.2f}"
              f"   C record {rec(c)}")

print("\n  ex-SMH C mask episodes (name, date, raw %, excess pp):")
for p in sorted([p for p in pool["C"] if p[0] != "SMH"], key=lambda z: z[1]):
    print(f"    {p[0]:5s} {p[1].date()}  {100*p[2]:+7.2f}  {100*p[3]:+7.2f}")

print("\n" + "=" * 78)
print("3. heterogeneity + shrinkage on the C mask cell (members with n>=2)")
print("=" * 78)
H2 = T[(T.C_n >= 2)]
for lbl, sub in (("all incl SMH", H2), ("ex-SMH", H2.drop(index="SMH", errors="ignore"))):
    q = cochran(sub.index, sub.C_exc.values, sub.C_se.values)
    print(f"  {lbl}: k={q.get('k')} FE common excess {q.get('fe_pct', np.nan):+.3f}pp "
          f"(se {q.get('fe_se_pct', np.nan):.3f})  Q {q.get('Q', np.nan):.2f} p {q.get('p', np.nan):.3f} "
          f"I2 {q.get('I2', np.nan):.1f}%  tau2 {q.get('tau2', np.nan):.3f}")
qa = cochran(H2.index, H2.C_exc.values, H2.C_se.values)
if "SMH" in H2.index:
    sh = shrink(H2.loc["SMH", "C_exc"], H2.loc["SMH", "C_se"], qa["fe_pct"], qa["tau2"])
    print(f"  SMH C excess {H2.loc['SMH','C_exc']:+.3f}pp (se {H2.loc['SMH','C_se']:.3f}) "
          f"-> EB shrunk {sh:+.3f}pp")
qb = cochran(T[T.B_n >= 2].index, T[T.B_n >= 2].B_exc.values, T[T.B_n >= 2].B_se.values)
print(f"  base B family (k={qb['k']}): FE {qb['fe_pct']:+.3f}pp Q p {qb['p']:.3f} I2 {qb['I2']:.1f}%")

print("\n  live state across members (who else holds B / C today):")
live = T[["live_r63", "live_r5", "live_r252"]]
print(live[(live.live_r63 <= 10)].round(2).to_string())
