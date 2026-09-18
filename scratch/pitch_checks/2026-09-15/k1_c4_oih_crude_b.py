"""C4 round 1b: the crude clause does not filter (OIH flush with crude NOT up
pays the same +0.446% at h=5 as the bare flush), so score the PARENT on its
own terms: long OIH after a 5d flush (pct_rank(5) <= 10), h=3/5/10, gap 10.

 - regime split (SPY >= 200d, today +6.7%), era split, drop-2020
 - reference class: identical flush on the 23 W25 ETFs, SPY >= 200d, excess
   over own drift; OIH's rank; Cochran
 - the crude clause as a continuous dose: CL=F 5d return terciles inside the
   OIH flush (does a bigger crude move buy a bigger services catch-up?)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, REF23, cell_stats, cochran, date_clusters, fwd, rec  # noqa

P = load_prices(REF23 + ["SPY", "CL=F"])
spy = P["SPY"]["Close"]
spy = spy[spy.index <= ASOF]
spy200 = spy / spy.rolling(200).mean() - 1
cl = P["CL=F"]["Close"].dropna()
cl = cl[cl.index <= ASOF]
cl5 = (cl / cl.shift(5) - 1)

s = P["OIH"]["Close"]
s = s[s.index <= ASOF]
r5 = pct_rank(s, 5)
flush = (r5 <= 10).fillna(False)
rows = []
E = {}
for h in (3, 5, 10):
    r = fwd(s, h, 1)
    cs = cell_stats(r, flush, 10)
    ep = cs["_dates"]
    df = pd.DataFrame({"d": ep, "v": cs["_vals"], "exc": cs["_vals"] - cs["_drift"],
                       "spy200": spy200.reindex(ep).values,
                       "cl5": cl5.reindex(ep, method="ffill").values})
    E[h] = df
    for lbl, sub in [("all", df), ("SPY>=200d", df[df.spy200 >= 0]), ("SPY<200d", df[df.spy200 < 0]),
                     ("SPY>=+5% over 200d", df[df.spy200 >= 0.05]),
                     ("pre-2018", df[df.d < "2018-01-01"]), ("2018+", df[df.d >= "2018-01-01"]),
                     ("2018+ & SPY>=200d", df[(df.d >= "2018-01-01") & (df.spy200 >= 0)]),
                     ("drop 2020", df[df.d.dt.year != 2020])]:
        x = summarize(sub.exc.values, f"h={h} {lbl}")
        x["raw_pct"] = 100 * sub.v.mean() if len(sub) else np.nan
        x["rec_exc"] = rec(sub.exc.values)
        rows.append(x)
show(rows, "PARENT: long OIH after 5d flush (r5<=10), excess over own drift, gap 10")

print("\n=== crude dose inside the OIH flush, h=5 (CL=F 5d return terciles + today's bucket) ===")
df = E[5].dropna(subset=["cl5"])
df["bucket"] = pd.cut(df.cl5, [-9, -0.05, 0.0, 0.05, 0.08, 9],
                      labels=["<-5%", "-5..0", "0..+5", "+5..+8", ">=+8% (today +10.8)"])
g = df.groupby("bucket", observed=False).agg(n=("exc", "size"), mean_exc=("exc", "mean"), raw=("v", "mean"))
g["mean_exc"] *= 100
g["raw"] *= 100
print(g.round(3).to_string())
from scipy.stats import spearmanr  # noqa
print(f"  Spearman(CL 5d, OIH fwd5 excess) = {spearmanr(df.cl5, df.exc).correlation:+.3f} on {len(df)}")
same = df[(df.spy200 >= 0)]
print(f"  within SPY>=200d: Spearman {spearmanr(same.cl5, same.exc).correlation:+.3f} on {len(same)}")

print("\n=== REFERENCE CLASS: identical 5d flush on 23 ETFs, SPY>=200d only, h=5, gap 10 ===")
out = []
pool = []
for t in REF23:
    x = P[t]["Close"]
    x = x[x.index <= ASOF]
    m = ((pct_rank(x, 5) <= 10) & (spy200.reindex(x.index) >= 0)).fillna(False)
    r = fwd(x, 5, 1)
    cs = cell_stats(r, m, 10)
    if cs["n"] >= 3:
        out.append({"t": t, "n": cs["n"], "exc": cs["excess_pct"], "se": cs["se_pct"], "hit": cs["hit"]})
        pool += [(d, v - cs["_drift"]) for d, v in zip(cs["_dates"], cs["_vals"])]
O = pd.DataFrame(out).set_index("t").sort_values("exc", ascending=False)
print(O.round(3).T.to_string())
q = cochran(O.index, O.exc, O.se)
print(f"  FE common excess {q['fe_pct']:+.3f}pp (se {q['fe_se_pct']:.3f})  Q p {q['p']:.3f}  I2 {q['I2']:.1f}%")
print(f"  OIH rank {list(O.index).index('OIH')+1} of {len(O)}; OIH excess {O.loc['OIH','exc']:+.3f}pp")
pv = np.array([p[1] for p in pool])
cl_, _ = date_clusters([p[0] for p in pool], pv, 7)
print(f"  pooled {len(pv)} eps mean excess {100*pv.mean():+.3f}pp; {len(cl_)} date clusters mean "
      f"{100*cl_.mean():+.3f}pp {rec(cl_)}")
