"""C6 round 2c: mechanism checks for the QE-flush gold rebound.

(a) Is there a quarter-end LIQUIDATION in gold at all? Flush base rate and mean 5d
    return into QE vs ordinary MEs vs all days.
(b) Is the flush RATES-DRIVEN (information) or liquidation (positioning)? Split the
    QE-flush sets by ^TNX's 5-session change into the QE (live: TNX 5d rank 100).
(c) Cross-asset: the same QE-flush rule on SLV and GDX (collateral liquidation should
    not be gold-only).
(d) Ungated post-QE gold drift by era (pre/post 2013 fossil check).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

P = load_prices(["GLD", "SLV", "GDX", "^TNX"])
g = P["GLD"]["Close"].dropna()
tnx = P["^TNX"]["Close"].dropna()
idx = g.index
LAST = idx[-1]


def qe_me(ix):
    s_ = pd.Series(ix, index=ix)
    me = pd.DatetimeIndex(s_.groupby([ix.year, ix.month]).max().values)
    me = me[(me < ix[-1])]
    return me, me[me.month.isin([3, 6, 9, 12])]


ME, QE = qe_me(idx)
ME, QE = ME[ME >= "2004-12-01"], QE[QE >= "2004-12-01"]
OME = ME[~ME.isin(QE)]
r5 = g / g.shift(5) - 1.0

print("(a) is there QE liquidation in gold?")
for lbl, a in (("QE", QE), ("ordinary ME", OME), ("all days", idx[idx >= "2004-12-01"])):
    v = r5.reindex(a).dropna()
    print(f"  {lbl:12s} n={len(v):5d}  mean 5d into it {100*v.mean():+.3f}%  P(5d<=-3%) {100*(v<=-0.03).mean():.1f}%  "
          f"P(5d<=-2%) {100*(v<=-0.02).mean():.1f}%")

# (b) rates-driven split
tnx5 = (tnx - tnx.shift(5)).reindex(idx)  # yield points (x10 in ^TNX units = %), keep raw
tnx5rk = pct_rank(tnx, 5).reindex(idx)
qm1 = pd.DatetimeIndex([idx[idx.get_loc(q) - 1] for q in QE])
print("\n(b) QE-flush sets split by the rates move into the QE (^TNX 5d change, rank = trailing-252 pct of 5d return)")
for thr, name in ((-0.03, "5d<=-3% at QE or QE-1"), (-0.02, "5d<=-2% at QE or QE-1")):
    sel = QE[(r5.reindex(QE).values <= thr) | (r5.reindex(qm1).values <= thr)]
    rows = []
    for q in sel:
        p = idx.get_loc(q)
        rows.append({"QE": q.date(), "tnx5d_bp": 10 * tnx5.loc[q] * 10, "tnx5_rank": tnx5rk.loc[q],
                     "h3": 100 * (g.iloc[p + 3] / g.iloc[p] - 1), "h5": 100 * (g.iloc[p + 5] / g.iloc[p] - 1),
                     "h7": 100 * (g.iloc[p + 7] / g.iloc[p] - 1)})
    df = pd.DataFrame(rows)
    print(f"  --- {name} ---")
    print(df.round(2).to_string(index=False))
    for lbl, m in (("yields UP into QE (tnx 5d > 0)", df.tnx5d_bp > 0), ("yields DOWN/flat", df.tnx5d_bp <= 0),
                   ("tnx5 rank >= 80 (live 100)", df.tnx5_rank >= 80)):
        sub = df[m]
        out = []
        for h in ("h3", "h5", "h7"):
            w = int((sub[h] > 0).sum())
            out.append(f"{h} {sub[h].mean():+.2f}% {w}-{len(sub)-w}")
        print(f"    {lbl:32s} n={len(sub)}  " + " | ".join(out))
print(f"  live: TNX 5d change {100*tnx5.iloc[-1]:+.1f} bp, rank {tnx5rk.iloc[-1]:.1f}")

# (c) cross-asset
print("\n(c) same rule on SLV / GDX (5d <= -3% at the QE close, long from the QE close)")
for t in ("SLV", "GDX"):
    s = P[t]["Close"].dropna()
    ix = s.index
    me_, qe_ = qe_me(ix)
    rr5 = s / s.shift(5) - 1
    rows = []
    for h in (3, 5, 7):
        f = s.shift(-h) / s - 1
        gq = qe_[(rr5.reindex(qe_) <= -0.03).values]
        oq = me_[~me_.isin(qe_)]
        oq = oq[(rr5.reindex(oq) <= -0.03).values]
        nq = ix[(rr5 <= -0.03).values & ~ix.isin(qe_)]
        vq, vo, vn = f.reindex(gq).dropna(), f.reindex(oq).dropna(), f.reindex(nq).dropna()
        allq = f.reindex(qe_).dropna()
        w = int((vq > 0).sum())
        rows.append({"t": t, "h": h, "QE_flush": 100 * vq.mean(), "rec": f"{w}-{len(vq)-w}",
                     "p": sign_test(w, len(vq)), "all_QE": 100 * allq.mean(), "ordME_flush": 100 * vo.mean(),
                     "nonQE_flush_day": 100 * vn.mean(), "drift": 100 * f.dropna().mean()})
    show(rows)

# (d) ungated post-QE drift by era
print("\n(d) ungated post-QE GLD, by era (h=7), vs own drift")
f7 = g.shift(-7) / g - 1
for lbl, lo, hi in (("2004-2012", "2004", "2013"), ("2013-2017", "2013", "2018"), ("2018+", "2018", "2027")):
    a = QE[(QE >= lo) & (QE < hi)]
    v = f7.reindex(a).dropna()
    d = f7[(f7.index >= lo) & (f7.index < hi)].dropna()
    w = int((v > 0).sum())
    print(f"  {lbl}: all-QE h7 {100*v.mean():+.3f}% ({w}-{len(v)-w}) vs drift {100*d.mean():+.3f}%")
