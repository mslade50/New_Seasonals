"""C2 round 2, part d: the one slice arguing for C2 is SPY >= 200d
(ex-SMH C mask +1.533pp on 73, clusters 26-14). Today sits in it (+6.7%).
Is that slice era-stable, or is it 2021 again?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, REF23, cell_stats, date_clusters, fwd, rec, vret  # noqa

px = load_prices(REF23 + ["SPY"])
spy = px["SPY"]["Close"]
spy = spy[spy.index <= ASOF]
spy200 = spy / spy.rolling(200).mean() - 1
rows = []
for t in REF23:
    if t == "SMH":
        continue
    s = px[t]["Close"]
    s = s[s.index <= ASOF]
    m = (pct_rank(s, 63) <= 5) & (vret(s, 252) >= 0.40) & (pct_rank(s, 5) < 15)
    cs = cell_stats(fwd(s, 10, 1), m, 10)
    for d, v in zip(cs["_dates"], cs["_vals"]):
        rows.append({"t": t, "d": d, "exc": v - cs["_drift"], "spy200": spy200.get(d, np.nan)})
D = pd.DataFrame(rows)
up = D[D.spy200 >= 0]


def line(sub, label):
    x = summarize(sub.exc.values, label)
    cl, _ = date_clusters(sub.d, sub.exc.values, 14)
    x["clusters"], x["cl_pct"], x["cl_rec"] = len(cl), 100 * cl.mean() if len(cl) else np.nan, rec(cl)
    return x


show([line(up, "SPY>=200d all"), line(up[up.d < "2018-01-01"], "SPY>=200d pre-2018"),
      line(up[up.d >= "2018-01-01"], "SPY>=200d 2018+"),
      line(up[up.d.dt.year != 2021], "SPY>=200d drop 2021"),
      line(up[~up.d.dt.year.isin([2021, 2026])], "SPY>=200d drop 2021+2026")],
     "ex-SMH C mask, h=10, SPY >= 200d")
print(up.groupby(up.d.dt.year).exc.agg(["size", "sum"]).assign(sum=lambda x: 100 * x["sum"]).round(2).T.to_string())
