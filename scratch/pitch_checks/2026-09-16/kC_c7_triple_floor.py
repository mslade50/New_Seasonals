"""kC c7 round 1 (+ the decisive regime splits): long the SPDR members at a
5/21/63 <= 10 triple floor, h=10, lag=1 (MOC tomorrow), declustered gap 10.

Re-derives the watchlist-35 pooled number from scratch (definition from
2026-09-02 c5c7_supplement.py: triple(s, k) on pct_rank 5/21/63), then asks
whether the LIVE slice (midterm year, SPY above its 200d, ^TNX at a 252 max)
is the paying side or the wrong-signed side.

Pooled statistics are reported two ways: naive episode-level (inflated by
same-date cross-name correlation) and DATE-CLUSTERED (episodes within 10 td of
each other on the SPY calendar merged, cluster mean of excess).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)
H, GAP = 10, 10
SPDR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
EXTRA = ["XLRE", "XLC"]
raw = load_prices(SPDR9 + EXTRA + ["SPY", "^TNX"])
C = {t: raw[t]["Close"].dropna() for t in raw}
spy = C["SPY"]
cal = spy.index
spy200 = (spy / spy.rolling(200).mean() - 1.0)
tnx = C["^TNX"]
tnx_at = tnx >= rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1e-9
tnx_near = tnx >= 0.97 * rolling_on_valid(tnx, lambda x: x.rolling(252).max())


def fwd(s, h=H, lag=1):
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def triple(s, k=10, wins=(5, 21, 63)):
    m = None
    for w in wins:
        x = pct_rank(s, w) <= k
        m = x if m is None else (m & x)
    return m.fillna(False)


def episodes(tickers, k=10, wins=(5, 21, 63), h=H, gap=GAP):
    rows = []
    for t in tickers:
        s = C[t]
        r = fwd(s, h)
        valid = r.dropna().index
        m = triple(s, k, wins)
        trig = s.index[m.values].intersection(valid)
        if len(trig) == 0:
            continue
        epi = declusters(trig, gap, valid)
        drift = r.loc[valid].mean()
        loc = local_control(valid, trig)
        locm = r.loc[loc].mean()
        for d in epi:
            rows.append({"tkr": t, "date": d, "ret": r.loc[d], "drift": drift,
                         "local": locm, "exc": r.loc[d] - drift})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["year"] = df.date.dt.year
    df["midterm"] = df.year % 4 == 2
    df["spy200"] = spy200.reindex(df.date).values
    df["above200"] = df.spy200 > 0
    df["tnx_at"] = tnx_at.reindex(df.date).fillna(False).values
    df["tnx_near"] = tnx_near.reindex(df.date).fillna(False).values
    df["post2018"] = df.date >= "2018-01-01"
    cnt = df.groupby("date").tkr.transform("count")
    df["n_same_day"] = cnt
    return df


def clusters(df):
    """Merge episodes within GAP td of each other on the SPY calendar."""
    pos = pd.Series(range(len(cal)), index=cal)
    d = df.sort_values("date").copy()
    p = d.date.map(lambda x: pos.get(x, np.nan)).values
    cid, last, c = [], -10**9, -1
    for q in p:
        if q - last >= GAP:
            c += 1
        cid.append(c)
        last = q
    d["cid"] = cid
    g = d.groupby("cid").agg(date=("date", "first"), ret=("ret", "mean"),
                             exc=("exc", "mean"), n=("tkr", "count"))
    return g


def line(df, label):
    if df is None or len(df) == 0:
        print(f"  {label:44s} N=0")
        return
    v, e = df.ret.values, df.exc.values
    g = clusters(df)
    ce = g.exc.values
    t_c = ce.mean() / (ce.std(ddof=1) / np.sqrt(len(ce))) if len(ce) > 2 else np.nan
    w = int((g.ret > 0).sum())
    print(f"  {label:44s} N={len(v):4d} mean {100*v.mean():+.3f}% hit {100*(v>0).mean():5.1f}% "
          f"exc {100*e.mean():+.3f}pp | clusters {len(g):3d} exc {100*ce.mean():+.3f}pp "
          f"t_c {t_c:+.2f} rec {w}-{len(g)-w} sign p {sign_test(w, len(g)):.4f} "
          f"worst {100*v.min():+.2f}%")


def cochran_tab(df):
    rows = []
    for t, g in df.groupby("tkr"):
        if len(g) < 3:
            continue
        s = C[t]
        r = fwd(s).dropna()
        se = np.sqrt(g.ret.var(ddof=1) / len(g) + r.var(ddof=1) / len(r))
        rows.append({"tkr": t, "n": len(g), "mean_pct": 100 * g.ret.mean(),
                     "exc_pct": 100 * g.exc.mean(), "se_pct": 100 * se,
                     "t": g.exc.mean() / se, "hit": 100 * (g.ret > 0).mean()})
    d = pd.DataFrame(rows).sort_values("t", ascending=False)
    w = 1 / d.se_pct ** 2
    mu = (w * d.exc_pct).sum() / w.sum()
    Q = float((w * (d.exc_pct - mu) ** 2).sum())
    k = len(d)
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    print(d.round(3).to_string(index=False))
    print(f"  FE common {mu:+.3f}pp (t {mu*np.sqrt(w.sum()):+.2f})  Cochran Q {Q:.2f}/{k-1} "
          f"p {1-sps.chi2.cdf(Q, k-1):.3f}  I2 {I2:.1f}%")


print("live:", {t: (round(float(pct_rank(C[t], 5).iloc[-1]), 1),
                    round(float(pct_rank(C[t], 21).iloc[-1]), 1),
                    round(float(pct_rank(C[t], 63).iloc[-1]), 1)) for t in ["XLU", "XLI", "XLY", "XLRE"]})
print(f"live SPY vs 200d {100*spy200.iloc[-1]:+.2f}%  TNX at 252 max {bool(tnx_at.iloc[-1])}  last {spy.index[-1].date()}")

E9 = episodes(SPDR9)
E11 = episodes(SPDR9 + EXTRA)
print("\n==== 1. existence (h=10, gap 10) ====")
line(E9, "nine SPDRs pooled")
line(E11, "eleven SPDRs pooled")
allr = np.concatenate([fwd(C[t]).dropna().values for t in SPDR9])
print(f"  CTRL-b all days nine SPDRs pooled mean {100*allr.mean():+.3f}%  "
      f"CTRL-c local +/-126 (mean of name locals) {100*E9.groupby('tkr').local.first().mean():+.3f}%")
cochran_tab(E9)

print("\n==== 2. splits (nine SPDRs, excess over own drift) ====")
for lbl, m in [("pre-2018", ~E9.post2018), ("2018+", E9.post2018),
               ("midterm", E9.midterm), ("non-midterm", ~E9.midterm),
               ("SPY above 200d", E9.above200), ("SPY below 200d", ~E9.above200),
               ("midterm & above200  <- LIVE", E9.midterm & E9.above200),
               ("midterm & below200", E9.midterm & ~E9.above200),
               ("non-mid & above200", ~E9.midterm & E9.above200),
               ("non-mid & below200", ~E9.midterm & ~E9.above200),
               ("TNX at 252 max  <- LIVE", E9.tnx_at), ("TNX not at max", ~E9.tnx_at),
               ("TNX within 3% of max", E9.tnx_near),
               ("above200 & TNX near max", E9.above200 & E9.tnx_near),
               ("above200 & SPY within 0-8% of 200d", E9.above200 & (E9.spy200 < 0.08)),
               (">=2 SPDRs same day  <- LIVE", E9.n_same_day >= 2),
               ("1 SPDR alone", E9.n_same_day == 1),
               ("XLU episodes", E9.tkr == "XLU"), ("XLI episodes", E9.tkr == "XLI"),
               ("XLU & above200", (E9.tkr == "XLU") & E9.above200),
               ("XLI & above200", (E9.tkr == "XLI") & E9.above200)]:
    line(E9[m], lbl)

print("\n==== 3. the live slice in detail: above200 episodes by member ====")
A = E9[E9.above200]
print(A.groupby("tkr").agg(n=("ret", "size"), mean=("ret", lambda x: 100 * x.mean()),
                           exc=("exc", lambda x: 100 * x.mean()),
                           hit=("ret", lambda x: 100 * (x > 0).mean())).round(3).to_string())
print("\n  midterm & above200 episodes:")
MA = E9[E9.midterm & E9.above200].sort_values("date")
print(MA[["tkr", "date", "ret", "exc", "spy200", "tnx_at"]].assign(
    ret=lambda x: (100 * x.ret).round(2), exc=lambda x: (100 * x.exc).round(2),
    spy200=lambda x: (100 * x.spy200).round(1)).to_string(index=False))

print("\n==== 4. the tradeable basket: EW of members at the floor on each cluster's first date ====")
g = clusters(E9)
for lbl, sel in [("all clusters", slice(None))]:
    pass
g["above200"] = spy200.reindex(g.date).values > 0
g["midterm"] = g.date.dt.year % 4 == 2
g["tnx_at"] = tnx_at.reindex(g.date).fillna(False).values
for lbl, m in [("all", g.index == g.index), ("above200", g.above200), ("below200", ~g.above200),
               ("midterm&above200", g.midterm & g.above200), ("nonmid&above200", ~g.midterm & g.above200),
               ("TNX at max", g.tnx_at), ("pre-2018", g.date < "2018"), ("2018+", g.date >= "2018")]:
    x = g[m]
    if len(x) < 2:
        print(f"  {lbl:20s} n={len(x)}")
        continue
    w = int((x.ret > 0).sum())
    print(f"  {lbl:20s} clusters {len(x):3d} basket mean {100*x.ret.mean():+.3f}% exc {100*x.exc.mean():+.3f}pp "
          f"hit {100*w/len(x):5.1f}% sign p {sign_test(w, len(x)):.4f} bootP {bootstrap_p_le0(x.exc.values):.3f}")
print("  concentration (all clusters, exc):", cluster_note(pd.DatetimeIndex(g.date), g.exc.values))
print("  2022 + 2026 clusters:")
print(g[(g.date.dt.year == 2022) | (g.date.dt.year == 2026)].assign(
    ret=lambda x: (100 * x.ret).round(2), exc=lambda x: (100 * x.exc).round(2)).to_string())
