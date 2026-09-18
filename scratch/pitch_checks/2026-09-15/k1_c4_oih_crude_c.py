"""C4 round 1c: the parent's family estimate (23-ETF 5d flush above SPY's
200d, FE +0.160pp at h=5) sits right at the 5x bar. Before it can re-price the
OIH member, check it on its own terms against today's actual state:

 - idiosyncratic vs broad: split by SPY's own 5d rank (today 18.3) and by
   SPY's 1d move on the flush day (today -0.45%)
 - era split pre-2018 / 2018+, and h=3/5/10
 - the hedged relative-value form the candidate names: OIH minus beta*XLE
   after the bare flush (does services-vs-energy mean-revert?)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, REF23, date_clusters, fwd, rec  # noqa

P = load_prices(REF23 + ["SPY"])
spy = P["SPY"]["Close"]
spy = spy[spy.index <= ASOF]
spy200 = spy / spy.rolling(200).mean() - 1
spy_r5 = pct_rank(spy, 5)
print(f"today SPY r5 {spy_r5.iloc[-1]:.1f}, SPY vs 200d {100*spy200.iloc[-1]:+.1f}%")

recs = []
for t in REF23:
    x = P[t]["Close"]
    x = x[x.index <= ASOF]
    m = ((pct_rank(x, 5) <= 10) & (spy200.reindex(x.index) >= 0)).fillna(False)
    for h in (3, 5, 10):
        r = fwd(x, h, 1)
        valid = r.dropna().index
        drift = r.loc[valid].mean()
        ep = declusters(x.index[m.values].intersection(valid), 10, valid)
        for d in ep:
            recs.append({"t": t, "h": h, "d": d, "exc": r.loc[d] - drift,
                         "spy_r5": spy_r5.get(d, np.nan)})
R = pd.DataFrame(recs)


def line(sub, label):
    x = summarize(sub.exc.values, label)
    cl, _ = date_clusters(sub.d, sub.exc.values, 7)
    x["clusters"], x["cl_exc_pct"], x["cl_rec"] = len(cl), 100 * cl.mean(), rec(cl)
    return x


rows = []
for h in (3, 5, 10):
    S = R[R.h == h]
    rows += [line(S, f"h={h} all (SPY>=200d)"),
             line(S[S.spy_r5 >= 30], f"h={h} idiosyncratic: SPY r5>=30"),
             line(S[S.spy_r5 < 30], f"h={h} SPY r5<30 (today 18.3)"),
             line(S[(S.spy_r5 >= 10) & (S.spy_r5 < 30)], f"h={h} SPY r5 10-30"),
             line(S[S.d < "2018-01-01"], f"h={h} pre-2018"),
             line(S[S.d >= "2018-01-01"], f"h={h} 2018+"),
             line(S[(S.d >= "2018-01-01") & (S.spy_r5 >= 10) & (S.spy_r5 < 30)], f"h={h} 2018+ & SPY r5 10-30")]
show(rows, "23-ETF 5d flush above SPY 200d, excess over own drift (gap 10)")

# hedged relative-value parent: OIH - beta*XLE after the bare OIH flush
o = P["OIH"]["Close"]
o = o[o.index <= ASOF]
e = P["XLE"]["Close"].reindex(o.index)
r1o, r1e = o.pct_change(), e.pct_change()
b = r1o.rolling(126).cov(r1e) / r1e.rolling(126).var()
m = (pct_rank(o, 5) <= 10).fillna(False)
rr = []
for h in (3, 5, 10):
    res = fwd(o, h, 1) - b * fwd(e, h, 1)
    valid = res.dropna().index
    ep = declusters(o.index[m.values].intersection(valid), 10, valid)
    v = res.loc[ep].values
    s = summarize(v, f"OIH - beta*XLE after OIH flush h={h}")
    s["ctrl_all_pct"] = 100 * res.loc[valid].mean()
    s["rec"] = rec(v)
    d = pd.DatetimeIndex(ep)
    s["pre2018_pct"] = 100 * v[d < "2018-01-01"].mean()
    s["post2018_pct"] = 100 * v[d >= "2018-01-01"].mean()
    rr.append(s)
show(rr, "hedged parent (the relative-value form)")
