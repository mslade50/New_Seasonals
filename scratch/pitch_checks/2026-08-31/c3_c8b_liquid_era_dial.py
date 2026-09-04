"""C8 round 2 -- give the pooled cell its best shot before killing it.

Round 1 found the sector clause worth -0.026pp over the bare name cell while
discarding 96.1% of the population, and a NON-MONOTONE dose response whose
BEST bucket is the anti-cell (sector r21 < 25). This probe tests the four
rescues that are still open:

 A. the LIQUID single-name universe (the population the brief actually claims,
    and the one TJX belongs to) rather than the 840-name mapped tape, which
    is full of microcaps (UPBD, INOD, PRDO, CALX)
 B. an ERA-MATCHED control (round 1 compared 2018+ dates to a full-sample base)
 C. a RANDOM-DISCARD placebo: is the sector clause distinguishable from
    throwing away 96% of the bare-name days at random?
 D. the LIVE SHAPE: dial >= 70/80, and TJX's own configuration
    (r21 <= 2 AND r63 <= 5, deep below the 200d)
Plus a data-integrity pass: round 1's bare cell carried a +1201.97% five-day
observation, which is a bad bar, not a trade.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

SECTOR_ETF = {
    "Utilities": "XLU", "Technology": "XLK", "Healthcare": "XLV",
    "Financial Services": "XLF", "Energy": "XLE", "Industrials": "XLI",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Basic Materials": "XLB", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
NAME_MAX, SEC_MIN, H_MAIN, MIN_BARS = 2.0, 75.0, 5, 500


def main():
    t0 = time.time()
    sys.path.insert(0, str(ROOT))
    from strategy_config import LIQUID_PLUS_COMMODITIES as LIQ  # noqa

    sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    mp["date"] = pd.to_datetime(mp["date"])
    have = set(mp["ticker"].unique())
    smap = {r.ticker: SECTOR_ETF[r.sector] for r in sm.itertuples()
            if r.sector in SECTOR_ETF and r.ticker in have}
    etfs = sorted(set(SECTOR_ETF.values()))
    wide = mp.pivot_table(index="date", columns="ticker", values="Close",
                          aggfunc="last").sort_index()
    names = [t for t in sorted(smap) if t not in etfs
             and wide[t].notna().sum() >= MIN_BARS]
    idx = wide.index
    liq = set(LIQ)
    liq_names = [n for n in names if n in liq]
    print(f"mapped single names {len(names)}; LIQUID_PLUS_COMMODITIES "
          f"single-name intersection {len(liq_names)}")
    print(f"  TJX in liquid subset: {'TJX' in liq_names}")

    etf_rk = {e: pct_rank(wide[e], 21) for e in etfs if e in wide.columns}
    nr21, nr63 = {}, {}
    for n in names:
        nr21[n] = pct_rank(wide[n], 21)
    for n in liq_names:
        nr63[n] = pct_rank(wide[n], 63)
    sma200 = {n: rolling_on_valid(wide[n], lambda x: x.rolling(200).mean())
              for n in liq_names}
    print(f"  ranks computed in {time.time()-t0:.0f}s")

    def fwd(n, h):
        c = wide[n].dropna()
        return (c.shift(-(1 + h)) / c.shift(-1) - 1.0).reindex(idx)

    def build(pool, h=H_MAIN, extra=False):
        recs = []
        for n in pool:
            e = smap[n]
            if e not in etf_rk:
                continue
            a = nr21[n]; b = etf_rk[e].reindex(idx); f = fwd(n, h)
            m = a.notna() & b.notna() & f.notna()
            if not m.any():
                continue
            d = {"date": idx[m.values], "name": n, "nr": a[m].values,
                 "er": b[m].values, "ret": f[m].values}
            if extra:
                c63 = nr63[n].reindex(idx); s2 = sma200[n].reindex(idx)
                d["nr63"] = c63[m].values
                d["p200"] = (wide[n][m].values / s2[m].values - 1.0)
            recs.append(pd.DataFrame(d))
        return pd.concat(recs, ignore_index=True)

    def clus(sub, label, base=None):
        if len(sub) == 0:
            return {"label": label, "n": 0}
        g = sub.groupby("date")["ret"].mean()
        s = summarize(g.values, label)
        s["n_days"] = len(sub)
        if base is not None:
            s["excess_pp"] = s["mean_pct"] - 100 * base
        return s

    # ================= A. LIQUID UNIVERSE ==============================
    print("\n" + "=" * 78)
    print("A. LIQUID single-name universe -- does the sector clause earn its "
          "place there?")
    print("=" * 78)
    DL = build(liq_names)
    baseL = DL["ret"].mean()
    print(f"  liquid pooled name-days {len(DL):,} over "
          f"{DL['date'].nunique():,} dates; base rate {100*baseL:+.3f}%")
    rows = [
        clus(DL[DL.nr <= NAME_MAX], "(a) name r21<=2 ALONE", baseL),
        clus(DL[(DL.nr <= NAME_MAX) & (DL.er >= SEC_MIN)], "(c) JOINT", baseL),
        clus(DL[(DL.nr <= NAME_MAX) & (DL.er < 25)], "(d) ANTI sector<25", baseL),
        clus(DL, "CTRL all liquid name-days", baseL),
    ]
    show(rows, f"liquid subset, h={H_MAIN}, date-clustered")
    print(f"\n  sector clause in the LIQUID subset is worth "
          f"{rows[1]['mean_pct']-rows[0]['mean_pct']:+.3f}pp over bare name "
          f"(bar was ~+0.15pp); population kept "
          f"{rows[1]['n_days']}/{rows[0]['n_days']} = "
          f"{100*rows[1]['n_days']/rows[0]['n_days']:.1f}%")
    JL = DL[(DL.nr <= NAME_MAX) & (DL.er >= SEC_MIN)]
    gl = JL.groupby("date")["ret"].mean()
    dec = declusters(pd.DatetimeIndex(gl.index), H_MAIN, idx)
    s = summarize(gl.loc[dec].values, "liquid joint, declustered dates")
    w = int((gl.loc[dec].values > 0).sum())
    print(f"  declustered: N={s['n']} mean {s['mean_pct']:+.3f}% hit "
          f"{s['hit']:.1f}% t {s['t']:.2f} sign p {sign_test(w, s['n']):.4f}; "
          f"excess vs liquid base {s['mean_pct']-100*baseL:+.3f}pp = "
          f"{(s['mean_pct']-100*baseL)*100:.1f} bps -> "
          f"{(s['mean_pct']-100*baseL)*100/10:.1f}x a 10 bps round trip")

    # ================= B. ERA-MATCHED CONTROL ==========================
    print("\n" + "=" * 78)
    print("B. ERA-MATCHED control (round 1 scored 2018+ against a full-sample "
          "base)")
    print("=" * 78)
    D = build(names)
    J = D[(D.nr <= NAME_MAX) & (D.er >= SEC_MIN)]
    A = D[D.nr <= NAME_MAX]
    for lo, hi, lbl in [(1999, 2018, "pre-2018"), (2018, 2027, "2018+"),
                        (2021, 2027, "2021+")]:
        msk = lambda x: (x["date"].dt.year >= lo) & (x["date"].dt.year < hi)
        b = D[msk(D)]["ret"].mean()
        j = J[msk(J)]; a = A[msk(A)]
        gj = j.groupby("date")["ret"].mean(); ga = a.groupby("date")["ret"].mean()
        print(f"  {lbl:9s} base {100*b:+.3f}%   bare-name {100*ga.mean():+.3f}% "
              f"(excess {100*(ga.mean()-b):+.3f}pp)   JOINT "
              f"{100*gj.mean():+.3f}% (excess {100*(gj.mean()-b):+.3f}pp, "
              f"N dates {len(gj)})   clause worth "
              f"{100*(gj.mean()-ga.mean()):+.3f}pp")

    # ================= C. RANDOM-DISCARD PLACEBO =======================
    print("\n" + "=" * 78)
    print("C. RANDOM-DISCARD PLACEBO -- is the sector clause better than "
          "throwing away 96% at random?")
    print("=" * 78)
    rng = np.random.default_rng(42)
    k = len(J)
    obs = J.groupby("date")["ret"].mean().mean()
    draws = []
    for _ in range(2000):
        samp = A.iloc[rng.choice(len(A), size=k, replace=False)]
        draws.append(samp.groupby("date")["ret"].mean().mean())
    draws = np.array(draws)
    print(f"  observed JOINT date-clustered mean {100*obs:+.3f}%")
    print(f"  random {k:,}-of-{len(A):,} draws from the BARE cell: "
          f"mean {100*draws.mean():+.3f}%, 5th {100*np.quantile(draws,.05):+.3f}%, "
          f"95th {100*np.quantile(draws,.95):+.3f}%")
    print(f"  P(random discard >= the sector clause) = "
          f"{float((draws >= obs).mean()):.4f}   <- a filter that filters "
          f"should sit in the right tail")

    # ================= data integrity ==================================
    print("\n" + "=" * 78)
    print("DATA INTEGRITY -- round 1's bare cell carried a +1201.97% 5-day bar")
    print("=" * 78)
    for lbl, X in [("bare name r21<=2", A), ("JOINT", J)]:
        v = X["ret"].values
        big = X[np.abs(v) > 1.0]
        print(f"  {lbl}: {len(big)} name-days with |5d return| > 100%; "
              f"max {100*v.max():+.1f}% min {100*v.min():+.1f}%")
        if len(big):
            print("   ", big.nlargest(4, "ret")[["date", "name", "ret"]]
                  .assign(ret=lambda d: (100*d.ret).round(1)).to_string(index=False))
        w = np.clip(v, np.quantile(v, .005), np.quantile(v, .995))
        Xw = X.assign(ret=w)
        gw = Xw.groupby("date")["ret"].mean()
        print(f"    winsorized 0.5/99.5: date-clustered "
              f"{100*gw.mean():+.3f}% (raw "
              f"{100*X.groupby('date')['ret'].mean().mean():+.3f}%)")

    # ================= D. LIVE SHAPE ===================================
    print("\n" + "=" * 78)
    print("D. THE LIVE SHAPE -- dial 87.6, TJX's own configuration")
    print("=" * 78)
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma = frag["63d"].rolling(10).mean()
    dial = ma.reindex(idx).ffill()
    JJ = J.assign(dial=dial.reindex(J["date"]).values)
    for lo in (50, 60, 70, 80, 85):
        s = JJ[JJ.dial >= lo]
        if len(s) == 0:
            print(f"  dial >= {lo}: NO observations")
            continue
        g = s.groupby("date")["ret"].mean()
        b = D.assign(dial=dial.reindex(D["date"]).values)
        b = b[b.dial >= lo]["ret"].mean()
        w = int((g.values > 0).sum())
        print(f"  dial >= {lo}: {len(s):4d} name-days / {len(g):3d} dates  "
              f"mean {100*g.mean():+.3f}%  hit {100*(g>0).mean():.1f}%  "
              f"same-dial universe base {100*b:+.3f}%  excess "
              f"{100*(g.mean()-b):+.3f}pp  sign p {sign_test(w,len(g)):.4f}")

    print("\n  TJX configuration: name r21<=2 AND r63<=5 AND >8% below the "
          "200d, LIQUID names only")
    DX = build(liq_names, extra=True)
    bx = DX["ret"].mean()
    for lbl, m in [
        ("r21<=2 & sector>=75 (liquid joint)",
         (DX.nr <= 2) & (DX.er >= 75)),
        ("+ r63<=5", (DX.nr <= 2) & (DX.er >= 75) & (DX.nr63 <= 5)),
        ("+ r63<=5 & >8% below 200d",
         (DX.nr <= 2) & (DX.er >= 75) & (DX.nr63 <= 5) & (DX.p200 <= -0.08)),
        ("TJX shape WITHOUT the sector clause",
         (DX.nr <= 2) & (DX.nr63 <= 5) & (DX.p200 <= -0.08)),
    ]:
        s = DX[m.values]
        if len(s) == 0:
            print(f"  {lbl}: N=0")
            continue
        g = s.groupby("date")["ret"].mean()
        dd = declusters(pd.DatetimeIndex(g.index), H_MAIN, idx)
        gd = g.loc[dd]
        w = int((gd.values > 0).sum())
        print(f"  {lbl:42s} name-days {len(s):5d}  dates {len(g):4d}  "
              f"declustered N={len(gd):3d} mean {100*gd.mean():+.3f}% "
              f"hit {100*(gd>0).mean():.1f}% excess {100*(gd.mean()-bx):+.3f}pp "
              f"sign p {sign_test(w,len(gd)):.4f}")

    print(f"\n  runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
