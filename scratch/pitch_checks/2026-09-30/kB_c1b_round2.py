"""C1 round 2 (W55 fired on its own trigger): concentration, definition
neighbours, era/regime, gate (QE label) attribution, session decomposition,
beta-neutral and crude-hedged forms. Live form = rank@QE-1, QE close -> QE+5,
long bottom-2 / short top-2 of the nine SPDRs by raw 63d return."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line, exante_beta  # noqa

idx = nyse_index()
px = close_panel(SPDR9 + ["SPY", "USO"]).reindex(idx)
P = px[SPDR9].values
T = me_table(idx)
T = T[T.me_pos > 130].reset_index(drop=True)
LB = {n: pd.DataFrame({t: px[t] / px[t].shift(n) - 1 for t in SPDR9}).values for n in (21, 63, 126)}
beta = np.column_stack([exante_beta(px[t], px["SPY"]).shift(1).values for t in SPDR9])
bUSO = exante_beta(px["XLE"], px["USO"]).shift(1).values
S, U = px["SPY"].values, px["USO"].values
iXLE = SPDR9.index("XLE")


def pair(m, h, k=2, lb=63, roff=-1, uni=None, mode="raw"):
    s = m + roff
    if m + h >= len(idx):
        return np.nan
    cols = np.arange(len(SPDR9)) if uni is None else np.array(uni)
    rr = LB[lb][s, cols]
    f = P[m + h, cols] / P[m, cols] - 1
    if mode == "beta":
        f = f - beta[m, cols] * (S[m + h] / S[m] - 1)
    if mode == "uso":  # XLE leg hedged with beta-USO, others raw
        if np.isfinite(U[m]) and np.isfinite(bUSO[m]):
            j = np.where(cols == iXLE)[0]
            if len(j):
                f = f.copy()
                f[j] = f[j] - bUSO[m] * (U[m + h] / U[m] - 1)
        else:
            return np.nan
    ok = np.isfinite(rr) & np.isfinite(f)
    if ok.sum() < 2 * k + 1:
        return np.nan
    o = np.argsort(rr[ok])
    return f[ok][o[:k]].mean() - f[ok][o[-k:]].mean()


def col(**kw):
    return np.array([pair(int(m), **kw) for m in T.me_pos])


T["rev"] = col(h=5)
q, nq = T[T.qe].dropna(subset=["rev"]), T[~T.qe].dropna(subset=["rev"])
Q18 = q[q.year >= 2018]

# 1. concentration
print("1. CONCENTRATION")
print("  all QE:", cluster_note(pd.DatetimeIndex(q.me_date), q.rev.values))
print("  2018+ :", cluster_note(pd.DatetimeIndex(Q18.me_date), Q18.rev.values))
by = Q18.groupby("year").rev.sum() * 100
print("  2018+ sum by year (pp):", by.round(2).to_dict(), f" total {by.sum():+.2f}pp")
top2 = by.sort_values(ascending=False).index[:2]
rest = Q18[~Q18.year.isin(top2)]
print(f"  2018+ ex best 2 years {list(top2)}: {stats_line(rest.rev, rest.me_date, '')}")
for y in (2008, 2020):
    r = q[q.year != y]
    print(f"  all QE ex-{y}: mean {100*r.rev.mean():+.3f}% n={len(r)}")

# 2. definition neighbours (QE, all / 2018+) vs non-QE
print("\n2. DEFINITION NEIGHBOURS (QE all | QE 2018+ | non-QE all)")
rows = []
grid = [("k=2 lb63 h5 (live)", dict(h=5)), ("k=1", dict(h=5, k=1)), ("k=3", dict(h=5, k=3)),
        ("lb21", dict(h=5, lb=21)), ("lb126", dict(h=5, lb=126)),
        ("h=3", dict(h=3)), ("h=4", dict(h=4)), ("h=6", dict(h=6)), ("h=7", dict(h=7)), ("h=10", dict(h=10)),
        ("rank@QE-5", dict(h=5, roff=-5)), ("rank@QE-10", dict(h=5, roff=-10)),
        ("ex-XLE universe", dict(h=5, uni=[i for i in range(9) if i != iXLE])),
        ("beta-SPY neutral legs", dict(h=5, mode="beta")),
        ("XLE leg hedged beta-USO", dict(h=5, mode="uso"))]
for lbl, kw in grid:
    v = col(**kw)
    vq, vn = v[T.qe.values], v[~T.qe.values]
    y18 = (T.year.values >= 2018)
    a = stats_line(vq, T.me_date[T.qe], lbl)
    b = summarize(v[T.qe.values & y18])
    c = summarize(vn)
    rows.append({"form": lbl, "n": a["n"], "QE_mean": a["mean_pct"], "QE_rec": a["rec"],
                 "QE18_mean": b["mean_pct"], "QE18_hit": b["hit"], "nonQE_mean": c["mean_pct"],
                 "QE_minus_nonQE": a["mean_pct"] - c["mean_pct"]})
show(rows, "neighbours")

# 3. era / regime
print("\n3. ERA / REGIME")
d = T.dropna(subset=["rev"]).copy()
for lbl, m in (("pre-2018", d.year < 2018), ("2018+", d.year >= 2018)):
    a, b = d[m & d.qe], d[m & ~d.qe]
    print(f"  {lbl}: QE {100*a.rev.mean():+.3f}% (n={len(a)}) non-QE {100*b.rev.mean():+.3f}% (n={len(b)}) "
          f"-> label {100*(a.rev.mean()-b.rev.mean()):+.3f}pp")
gap = []
for m in T.me_pos:
    rr = LB[63][int(m) - 1]
    o = np.sort(rr[np.isfinite(rr)])
    gap.append(o[-2:].mean() - o[:2].mean() if len(o) >= 5 else np.nan)
T["gap"] = gap
q = T[T.qe].dropna(subset=["rev"])
med = q.gap.median()
live_gap = np.sort(LB[63][-1])[-2:].mean() - np.sort(LB[63][-1])[:2].mean()
pct_live = 100 * (q.gap < live_gap).mean()
show([stats_line(q[q.gap >= med].rev, q[q.gap >= med].me_date, "QE, 63d dispersion >= median"),
      stats_line(q[q.gap < med].rev, q[q.gap < med].me_date, "QE, dispersion < median"),
      stats_line(q[q.gap >= q.gap.quantile(.8)].rev, q[q.gap >= q.gap.quantile(.8)].me_date, "QE, dispersion top quintile"),
      stats_line(q[q.midterm].rev, q[q.midterm].me_date, "QE midterm"),
      stats_line(q[~q.midterm].rev, q[~q.midterm].me_date, "QE non-midterm"),
      stats_line(q[q.month == 9].rev, q[q.month == 9].me_date, "QE September")],
     "regime splits (live form)")
print(f"  live top2-bottom2 63d gap {100*live_gap:.1f}pp = {pct_live:.0f}th pct of QE history (median {100*med:.1f}pp)")

# 4. session decomposition (which session pays)
print("\n4. SESSION DECOMPOSITION, QE -> QE+5 (live form)")
rows = []
for day in range(1, 6):
    v = []
    for m in q.me_pos:
        m = int(m)
        a, b = pair(m, day), pair(m, day - 1) if day > 1 else 0.0
        v.append(a - b if np.isfinite(a) else np.nan)  # approx session increment
    v = np.array(v)
    r = stats_line(v, q.me_date, f"QE+{day} session increment")
    r["2018+_mean"] = 100 * np.nanmean(v[q.year.values >= 2018])
    rows.append(r)
show(rows, "by session (cum-pair increments; mechanism names QE+1)")
