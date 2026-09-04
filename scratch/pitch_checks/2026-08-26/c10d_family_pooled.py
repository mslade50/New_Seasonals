"""C10 round 2b: if XLRE is not special, is the FAMILY form real?

The era-matched reference class says Cochran Q 6.07 on 9 df, I-squared 0%,
9 of 10 sectors positive -- i.e. a homogeneous family, which is the same
verdict watchlist #25 carries on a different construction.  So the honest
follow-up is whether the POOLED family cell survives outside XLRE's 2015+
window and outside 2020.  That decides whether anything gets parked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

DEEP = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ALL = DEEP + ["XLRE", "TLT", "SPY"]
px = load_prices(ALL)
CAL = px["SPY"].index.intersection(px["TLT"].index)
P = pd.DataFrame({t: px[t]["Close"].reindex(CAL) for t in ALL})
tlt_r5 = pct_rank(P["TLT"], 5)
Z_, R_, T_ = 0.5, 50, 75


def epi_ret(tkr, h, gated=True, since=None, gap=5):
    z = zscore(P[tkr], 10)
    r21 = pct_rank(P[tkr], 21)
    m = (z >= Z_) & (r21 <= R_)
    if gated:
        m = m & (tlt_r5 >= T_)
    ret = fwd_lag(P[tkr], h)
    ok = m.fillna(False).values & ret.notna().values & P[tkr].notna().values
    if since is not None:
        ok = ok & np.asarray(P.index >= since)
    sig = P.index[ok]
    if len(sig) == 0:
        return np.array([]), pd.DatetimeIndex([])
    e = declusters(sig, gap, P.index[P[tkr].notna()])
    return ret.loc[e].values, e


print("=" * 78)
print("POOLED FAMILY CELL (9 long-history SPDRs), gated vs bare vs drift")
print("=" * 78)
for since_lbl, since in (("2000+ FULL", None),
                         ("2015-10+", pd.Timestamp("2015-10-08")),
                         ("2021+", pd.Timestamp("2021-01-01"))):
    for h in (3, 5, 7):
        g, gd, b, bd, base = [], [], [], [], []
        for t in DEEP:
            x, e = epi_ret(t, h, True, since)
            g.append(x); gd.append(e)
            y, _ = epi_ret(t, h, False, since)
            b.append(y)
            f = fwd_lag(P[t], h)
            if since is not None:
                f = f[P.index >= since]
            base.append(f.dropna().values)
        G = np.concatenate(g); B = np.concatenate(b); Z = np.concatenate(base)
        D = pd.DatetimeIndex(np.concatenate([d.values for d in gd]))
        cov = np.array([pd.Timestamp("2020-02-01") <= d <= pd.Timestamp("2020-12-31")
                        for d in D])
        exc = 100 * (G.mean() - Z.mean())
        exc_ex = 100 * (G[~cov].mean() - Z.mean())
        print(f"  {since_lbl:10s} h={h}: gated {100*G.mean():+.3f}% N={len(G)} "
              f"hit {100*(G>0).mean():.1f}% | bare {100*B.mean():+.3f}% N={len(B)} "
              f"| drift {100*Z.mean():+.3f}% | gate "
              f"{100*(G.mean()-B.mean()):+.3f}pp | excess {exc:+.3f}pp = "
              f"{exc*100/6:.1f}x | EX-2020 excess {exc_ex:+.3f}pp = "
              f"{exc_ex*100/6:.1f}x (N={int((~cov).sum())})")
    print()

print("=" * 78)
print("MIDTERM split on the pooled family cell (today is a midterm year)")
print("=" * 78)
for h in (3, 5, 7):
    g, gd = [], []
    for t in DEEP:
        x, e = epi_ret(t, h, True, None)
        g.append(x); gd.append(e)
    G = np.concatenate(g)
    D = pd.DatetimeIndex(np.concatenate([d.values for d in gd]))
    mid = np.array([d.year % 4 == 2 for d in D])
    base = np.concatenate([fwd_lag(P[t], h).dropna().values for t in DEEP])
    show([summarize(G, f"h={h} pooled gated"),
          summarize(G[mid], f"  MIDTERM N={int(mid.sum())}"),
          summarize(G[~mid], "  non-midterm"),
          summarize(base, "  CTRL pooled drift")])
