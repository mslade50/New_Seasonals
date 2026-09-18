"""C7 round 1c: entry timing. The lag=0 h=1 residual was -0.143% (the session
after the shock), i.e. the only visible drift lives in a session a MOC order
cannot reach. Does a MOO entry on D+1 capture it? Decompose D+1 into the
overnight gap (untradeable) and open->close (MOO-tradeable), and run MOO
entries to D+1/D+3/D+5 closes. Short residual = -(member - beta*SPY), both
legs entered at the D+1 open. Calm shock, per-name gap 5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, REF23, date_clusters, rec  # noqa

R = pd.read_pickle(Path(__file__).with_name("k1_c7_shocks.pkl"))
px = load_prices(REF23 + ["SPY"])
spy = px["SPY"][px["SPY"].index <= ASOF]
nyse = spy.index
keep = []
for t, g in R[R.kind == "calm"].groupby("t"):
    dd = declusters(pd.DatetimeIndex(g.d), 5, nyse)
    keep.append(g[g.d.isin(dd)])
C = pd.concat(keep).reset_index(drop=True)

out = []
for row in C.itertuples():
    d = px[row.t][px[row.t].index <= ASOF]
    if row.d not in d.index:
        continue
    p = d.index.get_loc(row.d)
    if p + 6 >= len(d):
        continue
    sp = spy.reindex(d.index)
    c0, o1 = d["Close"].iloc[p], d["Open"].iloc[p + 1]
    s0, so1 = sp["Close"].iloc[p], sp["Open"].iloc[p + 1]
    rec_ = {"t": row.t, "d": row.d, "beta": row.beta,
            "gap": -((o1 / c0 - 1) - row.beta * (so1 / s0 - 1))}
    for h in (1, 3, 5):
        ch, sch = d["Close"].iloc[p + h], sp["Close"].iloc[p + h]
        rec_[f"moo{h}"] = -((ch / o1 - 1) - row.beta * (sch / so1 - 1))
        rec_[f"lag0_{h}"] = -((ch / c0 - 1) - row.beta * (sch / s0 - 1))
    out.append(rec_)
O = pd.DataFrame(out)


def line(col, label, sub=None):
    sub = O if sub is None else sub
    x = summarize(sub[col].values, label)
    cl, _ = date_clusters(sub.d, sub[col].values, 7)
    x["clusters"], x["cl_pct"], x["cl_rec"] = len(cl), 100 * np.nanmean(cl), rec(cl)
    return x


show([line("gap", "D+1 overnight gap (untradeable), SHORT residual"),
      line("moo1", "MOO D+1 -> close D+1"), line("moo3", "MOO D+1 -> close D+3"),
      line("moo5", "MOO D+1 -> close D+5"),
      line("lag0_1", "signal close -> close D+1 (lag0, contrast)"),
      line("moo3", "MOO h=3 pre-2018", O[O.d < "2018-01-01"]),
      line("moo3", "MOO h=3 2018+", O[O.d >= "2018-01-01"])],
     "C7 entry timing, SHORT residual (positive = drift pays)")
