"""C2 round 2, part c: score the PARENT on its own terms (the r5<15 conditioner
re-anchors rather than filters ex-SMH), and check whether the W25 split form
(r5 measured at the declustered B-episode start) even applies to today.

Parent B = r63<=5 & 252d >= +40%, long h=10, gap 10, 23-member family.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import (ASOF, REF23, cell_stats, cochran, date_clusters, dial_ma10,  # noqa
                       fwd, rec, vret)

px = load_prices(REF23 + ["SPY"])
spy = px["SPY"]["Close"].dropna()
spy = spy[spy.index <= ASOF]
spy200 = spy / spy.rolling(200).mean() - 1
dial = dial_ma10()

rows, per = [], []
for t in REF23:
    s = px[t]["Close"].dropna()
    s = s[s.index <= ASOF]
    r = fwd(s, 10, 1)
    r63, r5, r252 = pct_rank(s, 63), pct_rank(s, 5), vret(s, 252)
    B = (r63 <= 5) & (r252 >= 0.40)
    cs = cell_stats(r, B, 10)
    per.append({"t": t, "n": cs["n"], "exc": cs.get("excess_pct"), "se": cs.get("se_pct")})
    for d, x in zip(cs["_dates"], cs["_vals"]):
        rows.append({"t": t, "d": d, "exc": x - cs["_drift"], "raw": x,
                     "r5": r5.get(d), "spy200": spy200.get(d, np.nan), "dial": dial.get(d, np.nan)})
    if t == "SMH":
        valid = r.dropna().index
        bm = B.reindex(s.index, fill_value=False).fillna(False)
        trig_all = s.index[bm.values]
        epi_all = declusters(trig_all, 10, s.index)  # includes unrealised recent anchors
        print("SMH B declustered episode starts since 2026-06:",
              [(str(d.date()), round(float(r5.loc[d]), 1)) for d in epi_all if d >= pd.Timestamp("2026-06-01")])

P = pd.DataFrame(rows)


def line(df, label):
    if len(df) == 0:
        return {"label": label, "n": 0}
    s = summarize(df.exc.values, label)
    cl, _ = date_clusters(df.d, df.exc.values, 14)
    s["clusters"], s["cl_exc_pct"], s["cl_rec"] = len(cl), 100 * cl.mean(), rec(cl)
    return s


ex = P[P.t != "SMH"]
show([line(P, "B parent, all 23"), line(ex, "B parent ex-SMH"),
      line(ex[ex.d < "2018-01-01"], "ex-SMH pre-2018"), line(ex[ex.d >= "2018-01-01"], "ex-SMH 2018+"),
      line(ex[ex.d.dt.year != 2021], "ex-SMH drop 2021"),
      line(ex[~ex.d.dt.year.isin([2021, 2026])], "ex-SMH drop 2021+2026"),
      line(ex[ex.spy200 >= 0], "ex-SMH SPY>=200d"), line(ex[ex.spy200 < 0], "ex-SMH SPY<200d"),
      line(ex[ex.dial >= 50], "ex-SMH dial>=50"),
      ], "PARENT B, h=10 excess over own drift")
Q = pd.DataFrame(per).dropna()
Q = Q[Q.n >= 2]
q = cochran(Q.t, Q.exc, Q.se)
print(f"\n  B family k={q['k']} FE {q['fe_pct']:+.3f}pp (se {q['fe_se_pct']:.3f}, z {q['fe_pct']/q['fe_se_pct']:+.2f})"
      f"  Q p {q['p']:.3f} I2 {q['I2']:.1f}% tau2 {q['tau2']:.3f}")
q2 = cochran(Q[Q.t != "SMH"].t, Q[Q.t != "SMH"].exc, Q[Q.t != "SMH"].se)
print(f"  ex-SMH FE {q2['fe_pct']:+.3f}pp (z {q2['fe_pct']/q2['fe_se_pct']:+.2f})")
yr = ex.groupby(ex.d.dt.year).exc.agg(["size", "sum"])
yr["share"] = 100 * yr["sum"] / ex.exc.sum()
print("\n  ex-SMH parent by year (sum excess, share %):")
print((yr.assign(sum=100 * yr["sum"])).round(1).T.to_string())

# W25 split form era check (r5 at B-episode start < 15), ex-SMH
sp = ex[ex.r5 < 15]
show([line(sp, "split form r5<15 at B start, ex-SMH"),
      line(sp[sp.d < "2018-01-01"], "  pre-2018"), line(sp[sp.d >= "2018-01-01"], "  2018+"),
      line(sp[sp.d.dt.year != 2021], "  drop 2021"),
      line(ex[ex.r5 >= 15], "split form r5>=15, ex-SMH"),
      line(ex[(ex.r5 >= 15) & (ex.d.dt.year != 2021)], "  drop 2021")], "W25 split form, ex-SMH")
