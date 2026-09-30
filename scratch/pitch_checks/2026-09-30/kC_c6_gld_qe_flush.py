"""C6 round 1: long GLD from the quarter-end close after a quarter-end-week flush.

Pre-specified: long GLD (GC=F before GLD's first QE, seam-checked) from the QE close,
h=1..10, gate GLD 5d return <= -3% at the QE close (neighbour: r5 trailing-252
percentile <= 15). Calendar anchor: entry = the QE close (lag=0).
Controls: (a) all QEs ungated, (b) the same flush on NON-QE days (the decisive test:
does the quarter-end matter at all?), (c) ordinary (non-Q) month-ends with the flush,
(d) own drift. Split with ^TNX within 2% of its 252 max.
Rank convention: r5 = trailing-252 percentile of the 5-session return (pitch_lab.pct_rank).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["GLD", "GC=F", "^TNX"])
g = P["GLD"]["Close"].dropna()
gc = P["GC=F"]["Close"].dropna()
tnx = P["^TNX"]["Close"].dropna()
idx = g.index
LAST = idx[-1]

# ---------------------------------------------------------------- 0. GC=F seam check vs GLD
d = pd.DataFrame({"gld": g.pct_change(), "gc": gc.pct_change()}).dropna()
d["diff"] = d.gc - d.gld
d["dom"] = d.index.day
d["m"] = d.index.month
big = d[d["diff"].abs() > 0.015]
print(f"0. GC=F minus GLD daily return: sd {100*d['diff'].std():.3f}%, days |diff|>1.5%: {len(big)}")
# seam signature: mean diff by day-of-month in the last 8 calendar days, even (contract) months vs odd
late = d[d.dom >= 22]
tab = late.groupby([late.m % 2 == 0, late.dom])["diff"].mean().unstack(0) * 1e4
tab.columns = ["odd month bp", "even month bp"]
print("   mean GC=F-GLD diff (bp) by day of month >= 22:")
print(tab.round(1).T.to_string())
roll = d[(d.m % 2 == 1)].groupby("dom")["diff"].mean() * 1e4
print("   odd months (Jan/Mar/May/Jul/Sep/Nov, where GC's active contract rolls at the month end), by dom:")
print(roll.tail(10).round(1).to_string())

# ---------------------------------------------------------------- anchors and series
def qe_me(ix: pd.DatetimeIndex, last: pd.Timestamp):
    s = pd.Series(ix, index=ix)
    me = pd.DatetimeIndex(s.groupby([ix.year, ix.month]).max().values)
    me = me[me < last]
    qe = me[me.month.isin([3, 6, 9, 12])]
    return me, qe


ME_g, QE_g = qe_me(idx, LAST)
QE_g = QE_g[QE_g >= "2004-12-01"]
ME_g = ME_g[ME_g >= "2004-12-01"]
ME_c, QE_c = qe_me(gc.index, gc.index[-1])
QE_c_pre = QE_c[QE_c < "2004-12-01"]
ME_c_pre = ME_c[ME_c < "2004-12-01"]

r5g = g / g.shift(5) - 1.0
r5c = gc / gc.shift(5) - 1.0
rk5g = pct_rank(g, 5)
rk5c = pct_rank(gc, 5)
tnx_hi = (tnx >= 0.98 * rolling_on_valid(tnx, lambda x: x.rolling(252).max()))

print(f"\nlive: GLD 09-29 {g.iloc[-1]:.2f}, 5d {100*r5g.iloc[-1]:+.2f}%, r5 rank {rk5g.iloc[-1]:.1f}; "
      f"09-23 close {g.loc['2026-09-23']:.2f} -> today's QE close needs <= {0.97*g.loc['2026-09-23']:.2f} for the -3% gate; "
      f"TNX-high {bool(tnx_hi.iloc[-1])}")


def fwd(s: pd.Series, anchors, h: int) -> pd.Series:
    r = s.shift(-h) / s - 1.0
    return r.reindex(pd.DatetimeIndex(anchors)).dropna()


def row(v: pd.Series, lbl: str) -> dict:
    s = summarize(v.values, lbl)
    w = int((v > 0).sum())
    s["rec"] = f"{w}-{len(v)-w}"
    s["sign_p"] = sign_test(w, len(v)) if len(v) else np.nan
    return s


def spliced(anchors_g, anchors_c, h):
    return pd.concat([fwd(gc, anchors_c, h), fwd(g, anchors_g, h)])


gate_g = r5g <= -0.03
gate_c = r5c <= -0.03
gq_g = QE_g[gate_g.reindex(QE_g).fillna(False).values]
gq_c = QE_c_pre[gate_c.reindex(QE_c_pre).fillna(False).values]
print(f"\nGated QEs: GLD-era {len(gq_g)} of {len(QE_g)}; GC=F pre-GLD {len(gq_c)} of {len(QE_c_pre)}")
print("  dates:", ", ".join(str(x.date()) for x in list(gq_c) + list(gq_g)))
# tradeable variant: gate read at QE-1 close
qm1 = pd.DatetimeIndex([idx[idx.get_loc(q) - 1] for q in QE_g])
gate_m1 = gate_g.reindex(qm1).fillna(False).values
gq_m1 = QE_g[gate_m1]

rows = []
for h in (1, 2, 3, 5, 7, 10):
    allq = spliced(QE_g, QE_c_pre, h)
    gated = spliced(gq_g, gq_c, h)
    gated_glds = fwd(g, gq_g, h)
    m1 = fwd(g, gq_m1, h)
    drift = (g.shift(-h) / g - 1.0).dropna()
    # same flush on NON-QE days (day level, GLD era), and first-day-of-cluster
    nonqe = idx[(gate_g.values) & (~idx.isin(QE_g))]
    nonqe = nonqe[nonqe >= "2004-12-01"]
    nq = fwd(g, nonqe, h)
    nq_ep = fwd(g, declusters(nonqe, max(h, 5), idx), h)
    ome = ME_g[~ME_g.isin(QE_g)]
    ome_g = ome[gate_g.reindex(ome).fillna(False).values]
    om = fwd(g, ome_g, h)
    for lbl, v in (("QE gated (spliced)", gated), ("QE gated GLD-only", gated_glds),
                   ("QE gated at QE-1 (tradeable)", m1), ("CTRL a: all QEs", allq),
                   ("CTRL b: flush non-QE days (day-lvl)", nq), ("CTRL b': flush non-QE, 1st/cluster", nq_ep),
                   ("CTRL c: flush at ordinary MEs", om), ("CTRL d: own drift all days", drift)):
        r = row(v, lbl)
        r["h"] = h
        rows.append(r)
df = pd.DataFrame(rows)
for h, grp in df.groupby("h"):
    print(f"\n=== LONG GLD, h={h} ===")
    print(grp[["label", "n", "mean_pct", "median_pct", "hit", "rec", "sign_p", "worst_pct"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- neighbours and TNX split at h=5 / h=3
for h in (3, 5):
    print(f"\n--- neighbours / splits, h={h} ---")
    rr = []
    for lbl, thr in (("5d <= -2%", -0.02), ("5d <= -2.5%", -0.025), ("5d <= -3%", -0.03),
                     ("5d <= -3.5%", -0.035), ("5d <= -4%", -0.04)):
        a = QE_g[(r5g.reindex(QE_g) <= thr).values]
        ac = QE_c_pre[(r5c.reindex(QE_c_pre) <= thr).values]
        rr.append(row(spliced(a, ac, h), f"QE {lbl}"))
        b = idx[(r5g <= thr).values & ~idx.isin(QE_g)]
        rr.append(row(fwd(g, b, h), f"   non-QE {lbl} (day-lvl)"))
    for q in (10, 15, 20):
        a = QE_g[(rk5g.reindex(QE_g) <= q).values]
        ac = QE_c_pre[(rk5c.reindex(QE_c_pre) <= q).values]
        rr.append(row(spliced(a, ac, h), f"QE r5 rank <= {q}"))
        b = idx[(rk5g <= q).values & ~idx.isin(QE_g)]
        rr.append(row(fwd(g, b, h), f"   non-QE r5 rank <= {q} (day-lvl)"))
    th = tnx_hi.reindex(gq_g).fillna(False).values
    rr.append(row(fwd(g, gq_g[th], h), "QE gated, TNX within 2% of 252 max"))
    rr.append(row(fwd(g, gq_g[~th], h), "QE gated, TNX not at high"))
    b = idx[(gate_g.values) & ~idx.isin(QE_g) & tnx_hi.reindex(idx).fillna(False).values]
    rr.append(row(fwd(g, b, h), "   non-QE flush, TNX at high (day-lvl)"))
    show(rr)

# ---------------------------------------------------------------- episode list
print("\nEpisodes (gated QEs), long GLD/GC=F from the QE close:")
for a in list(gq_c) + list(gq_g):
    s = gc if a < pd.Timestamp("2004-12-01") else g
    r5 = (r5c if s is gc else r5g).loc[a]
    vals = [100 * (s.shift(-h) / s - 1.0).get(a, np.nan) for h in (1, 3, 5, 10)]
    th = bool(tnx_hi.get(a, False))
    print(f"  {a.date()} {'GC=F' if s is gc else 'GLD '} 5d {100*r5:+.2f}%  h1 {vals[0]:+.2f} h3 {vals[1]:+.2f} "
          f"h5 {vals[2]:+.2f} h10 {vals[3]:+.2f}  TNXhi={th}  midterm={a.year % 4 == 2}")
