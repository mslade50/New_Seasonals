"""C8 round 3 -- the narrowed liquid cell, charged for the paths that found it.

Round 2 left one live shape: LIQUID names, r21<=2 AND r63<=5 AND >8% below the
200d AND sector r21>=75 -> declustered N=90, +1.345%, excess +1.069pp,
sign p 0.0010. That cell was reached by adding two conditioners chosen to match
TJX after the pre-declared gate attribution had already failed on the full tape
(clause worth -0.026pp) . This probe charges it:

 1. DISJOINT complement: sector>=75 vs sector<75 INSIDE the narrowed shape,
    plus the full sector-rank dose response there
 2. the LIQUID restriction itself -- does the clause work on the OTHER 667
    mapped names, or is 'liquid' the subsample doing the work?
 3. era (pre-2018 / 2018+ / 2021+) and midterm
 4. multiple comparisons: the 5x5x5 grid the chosen cell came from, with a
    permutation max-of-grid null
 5. concentration by name / year / top-k episodes
 6. dial distribution and the dial>=70 slice, since today reads 87.6
 7. book overlap and earnings hazard on the narrowed cell
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
H = 5
MIN_BARS = 500


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
    lq = [n for n in names if n in liq]
    nl = [n for n in names if n not in liq]
    print(f"liquid single names {len(lq)}, non-liquid {len(nl)}")
    print("NOTE: LIQUID_PLUS_COMMODITIES is TODAY's liquid list, so using it as "
          "a historical filter is itself a survivorship/selection choice.")

    etf_rk = {e: pct_rank(wide[e], 21) for e in etfs if e in wide.columns}

    def frame(pool):
        recs = []
        for n in pool:
            e = smap[n]
            if e not in etf_rk:
                continue
            a = pct_rank(wide[n], 21)
            c = pct_rank(wide[n], 63)
            s2 = rolling_on_valid(wide[n], lambda x: x.rolling(200).mean())
            b = etf_rk[e].reindex(idx)
            cl = wide[n].dropna()
            f = (cl.shift(-(1 + H)) / cl.shift(-1) - 1.0).reindex(idx)
            m = a.notna() & b.notna() & c.notna() & s2.notna() & f.notna()
            if not m.any():
                continue
            recs.append(pd.DataFrame({
                "date": idx[m.values], "name": n, "nr": a[m].values,
                "nr63": c[m].values, "er": b[m].values,
                "p200": (wide[n][m].values / s2[m].values - 1.0),
                "ret": f[m].values}))
        return pd.concat(recs, ignore_index=True)

    DL = frame(lq)
    DN = frame(nl)
    print(f"  built in {time.time()-t0:.0f}s: liquid {len(DL):,} name-days, "
          f"non-liquid {len(DN):,}")

    def stat(sub, label, base):
        if len(sub) == 0:
            return {"label": label, "n": 0}
        g = sub.groupby("date")["ret"].mean()
        d = declusters(pd.DatetimeIndex(g.index), H, idx)
        gd = g.loc[d]
        s = summarize(gd.values, label)
        s["n_days"] = len(sub)
        s["n_dates"] = len(g)
        s["excess_pp"] = s["mean_pct"] - 100 * base
        s["sign_p"] = sign_test(int((gd.values > 0).sum()), len(gd))
        return s

    def shape(D, r63=5, p200=-0.08):
        return (D.nr <= 2) & (D.nr63 <= r63) & (D.p200 <= p200)

    bL = DL["ret"].mean()
    bN = DN["ret"].mean()
    print(f"  liquid base {100*bL:+.3f}%   non-liquid base {100*bN:+.3f}%")

    # ---------------- 1. DISJOINT COMPLEMENT + DOSE RESPONSE ---------------
    print("\n" + "=" * 78)
    print("1. INSIDE the narrowed shape (liquid, r21<=2 & r63<=5 & >8% below "
          "200d): does the sector clause separate?")
    print("=" * 78)
    S = DL[shape(DL).values]
    print(f"  narrowed shape population: {len(S):,} name-days on "
          f"{S['date'].nunique():,} dates, {S['name'].nunique()} names")
    rows = [stat(S[S.er >= 75], "sector r21 >= 75 (the cell)", bL),
            stat(S[S.er < 75], "sector r21 <  75 (COMPLEMENT)", bL),
            stat(S, "narrowed shape, all sector states", bL),
            stat(DL[DL.nr <= 2], "bare liquid name r21<=2", bL)]
    show(rows, "disjoint split inside the narrowed shape")
    print(f"\n  clause vs its DISJOINT complement: "
          f"{rows[0]['mean_pct']-rows[1]['mean_pct']:+.3f}pp")
    br = []
    for lo, hi in [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]:
        br.append(stat(S[(S.er >= lo) & (S.er < hi)], f"sector [{lo},{hi})", bL))
    show(br, "sector-rank dose response INSIDE the narrowed shape")

    # ---------------- 2. THE LIQUID RESTRICTION ----------------------------
    print("\n" + "=" * 78)
    print("2. IS 'LIQUID' THE SUBSAMPLE DOING THE WORK? same cell on the other "
          f"{len(nl)} mapped names")
    print("=" * 78)
    SN = DN[shape(DN).values]
    rows = [stat(S[S.er >= 75], "LIQUID cell", bL),
            stat(SN[SN.er >= 75], "NON-LIQUID cell", bN),
            stat(SN[SN.er < 75], "NON-LIQUID complement", bN),
            stat(DN[DN.nr <= 2], "bare non-liquid r21<=2", bN)]
    show(rows, "liquid vs non-liquid, same narrowed cell")
    print(f"  clause value: liquid "
          f"{rows[0]['mean_pct']-stat(S[S.er<75],'',bL)['mean_pct']:+.3f}pp   "
          f"non-liquid {rows[1]['mean_pct']-rows[2]['mean_pct']:+.3f}pp")

    # ---------------- 3. ERA + MIDTERM -------------------------------------
    print("\n" + "=" * 78)
    print("3. ERA AND MIDTERM on the narrowed liquid cell")
    print("=" * 78)
    C = S[S.er >= 75]
    K = S[S.er < 75]
    for lo, hi, lbl in [(1999, 2018, "pre-2018"), (2018, 2027, "2018+"),
                        (2021, 2027, "2021+")]:
        c = C[(C.date.dt.year >= lo) & (C.date.dt.year < hi)]
        k = K[(K.date.dt.year >= lo) & (K.date.dt.year < hi)]
        b = DL[(DL.date.dt.year >= lo) & (DL.date.dt.year < hi)]["ret"].mean()
        sc = stat(c, "", b); sk = stat(k, "", b)
        print(f"  {lbl:9s} base {100*b:+.3f}%   CELL N={sc.get('n',0):3d} "
              f"{sc.get('mean_pct',float('nan')):+.3f}% (excess "
              f"{sc.get('excess_pp',float('nan')):+.3f}pp, hit "
              f"{sc.get('hit',float('nan')):.1f}%, sign p "
              f"{sc.get('sign_p',float('nan')):.4f})   COMPLEMENT N="
              f"{sk.get('n',0):3d} {sk.get('mean_pct',float('nan')):+.3f}% "
              f"-> clause {sc.get('mean_pct',np.nan)-sk.get('mean_pct',np.nan):+.3f}pp")
    g = C.groupby("date")["ret"].mean()
    d = declusters(pd.DatetimeIndex(g.index), H, idx)
    gd = g.loc[d]
    mid = np.array([x.year % 4 == 2 for x in gd.index])
    show([summarize(gd.values[mid], f"midterm (N={int(mid.sum())})"),
          summarize(gd.values[~mid], f"non-midterm (N={int((~mid).sum())})")],
         "midterm split (declustered episodes)")

    # ---------------- 4. MULTIPLE COMPARISONS ------------------------------
    print("\n" + "=" * 78)
    print("4. THE GRID THE CELL CAME FROM (5 r63 x 5 p200 x 5 sector = 125)")
    print("=" * 78)
    r63s = [2, 5, 10, 20, 101]
    p200s = [-0.12, -0.08, -0.04, 0.0, 9.9]
    secs = [50, 60, 75, 85, 90]
    cells = []
    for a in r63s:
        for b in p200s:
            for c in secs:
                m = (DL.nr <= 2) & (DL.nr63 <= a) & (DL.p200 <= b) & (DL.er >= c)
                sub = DL[m.values]
                if len(sub) < 30:
                    continue
                gg = sub.groupby("date")["ret"].mean()
                dd = declusters(pd.DatetimeIndex(gg.index), H, idx)
                if len(dd) < 20:
                    continue
                cells.append({"r63": a, "p200": b, "sec": c, "N": len(dd),
                              "mean_pct": 100 * gg.loc[dd].mean(),
                              "excess_pp": 100 * (gg.loc[dd].mean() - bL)})
    G = pd.DataFrame(cells).sort_values("excess_pp", ascending=False)
    print(f"  {len(G)} cells with N>=20 declustered episodes")
    print("  top 8:")
    print(G.head(8).round(3).to_string(index=False))
    print("  bottom 5:")
    print(G.tail(5).round(3).to_string(index=False))
    chosen = G[(G.r63 == 5) & (G.p200 == -0.08) & (G.sec == 75)]
    if len(chosen):
        rk = int((G["excess_pp"] > chosen["excess_pp"].iloc[0]).sum()) + 1
        print(f"\n  the CHOSEN cell (r63<=5, p200<=-8%, sector>=75) ranks "
              f"{rk} of {len(G)}; grid excess sd = {G['excess_pp'].std():.3f}pp, "
              f"median {G['excess_pp'].median():+.3f}pp")
    # permutation max-of-grid: shuffle the sector-rank column within date
    rng = np.random.default_rng(42)
    obs_max = G["excess_pp"].max()
    maxes = []
    Dp = DL[["date", "nr", "nr63", "p200", "er", "ret"]].copy()
    for it in range(300):
        Dp["er"] = Dp.groupby("date")["er"].transform(
            lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).values)
        best = -9e9
        for a in r63s:
            for b in p200s:
                for c in secs:
                    m = (Dp.nr <= 2) & (Dp.nr63 <= a) & (Dp.p200 <= b) & (Dp.er >= c)
                    sub = Dp[m.values]
                    if len(sub) < 30:
                        continue
                    gg = sub.groupby("date")["ret"].mean()
                    dd = declusters(pd.DatetimeIndex(gg.index), H, idx)
                    if len(dd) < 20:
                        continue
                    best = max(best, 100 * (gg.loc[dd].mean() - bL))
        maxes.append(best)
    maxes = np.array(maxes)
    print(f"  permutation (sector rank shuffled WITHIN date, 300 reps) "
          f"max-of-grid null: median {np.median(maxes):+.3f}pp, 90th "
          f"{np.quantile(maxes,.9):+.3f}pp; observed grid max {obs_max:+.3f}pp "
          f"-> family-wise p = {float((maxes >= obs_max).mean()):.4f}")

    # ---------------- 5. CONCENTRATION -------------------------------------
    print("\n" + "=" * 78)
    print("5. CONCENTRATION")
    print("=" * 78)
    print(f"  {C['name'].nunique()} names supply {len(C)} name-days; top 8: "
          f"{C.groupby('name').size().sort_values(ascending=False).head(8).to_dict()}")
    print(f"  episodes: {cluster_note(pd.DatetimeIndex(gd.index), gd.values, k=3)}")
    byyr = pd.Series(gd.values, index=gd.index).groupby(gd.index.year).agg(
        ["size", "mean"])
    byyr["mean"] = (100 * byyr["mean"]).round(2)
    print("\n  by year (episode count, mean %):")
    print(byyr.to_string())
    dropbest = np.sort(gd.values)[:-3]
    print(f"  drop the 3 best episodes: {100*dropbest.mean():+.3f}% "
          f"(excess {100*(dropbest.mean()-bL):+.3f}pp) on N={len(dropbest)}")

    # ---------------- 6. DIAL ----------------------------------------------
    print("\n" + "=" * 78)
    print("6. DIAL (today ma10(63d) = 87.6)")
    print("=" * 78)
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    dial = frag["63d"].rolling(10).mean().reindex(idx).ffill()
    dv = dial.reindex(pd.DatetimeIndex(gd.index)).dropna()
    print(f"  {len(dv)} of {len(gd)} episodes carry a dial reading "
          f"(series starts {frag.index[0].date()})")
    if len(dv):
        print(f"  min {dv.min():.1f} p25 {dv.quantile(.25):.1f} median "
              f"{dv.median():.1f} p75 {dv.quantile(.75):.1f} MAX {dv.max():.1f}; "
              f">=70: {int((dv>=70).sum())}  >=85: {int((dv>=85).sum())}")
        for lo in (50, 70):
            sel = dv[dv >= lo].index
            v = pd.Series(gd.values, index=gd.index).reindex(sel).dropna()
            if len(v):
                print(f"  dial >= {lo}: N={len(v)} mean {100*v.mean():+.3f}% "
                      f"hit {100*(v>0).mean():.1f}% worst {100*v.min():.2f}%")

    # ---------------- 7. BOOK OVERLAP + EARNINGS ---------------------------
    print("\n" + "=" * 78)
    print("7. BOOK OVERLAP + EARNINGS on the narrowed cell")
    print("=" * 78)
    led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    led["Signal Date"] = pd.to_datetime(led["Signal Date"])
    key = set(zip(C["name"], C["date"]))
    led["k"] = list(zip(led["Ticker"], led["Signal Date"]))
    hit = led[led["k"].isin(key)]
    print(f"  ledger rows on the exact name-days: {len(hit)} of {len(key)} "
          f"trigger name-days")
    if len(hit):
        print(hit.groupby("Strategy").agg(n=("R_Multiple", "size"),
                                          avgR=("R_Multiple", "mean")).round(3).to_string())
    ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
    ec["date"] = pd.to_datetime(ec["date"])
    emap = {t: np.sort(g2["date"].values) for t, g2 in ec.groupby("ticker")}
    pos = pd.Series(range(len(idx)), index=idx)
    fl = []
    for nm, dt in zip(C["name"].values, C["date"].values):
        arr = emap.get(nm); p = pos.get(pd.Timestamp(dt))
        if arr is None or p is None or p + 1 + H >= len(idx):
            fl.append(False); continue
        lo = np.datetime64(idx[p + 1]); hi = np.datetime64(idx[p + 1 + H])
        fl.append(bool(((arr > lo) & (arr <= hi)).any()))
    C2 = C.assign(earn=fl)
    print(f"  earnings inside the hold: {int(C2.earn.sum())} of {len(C2)} "
          f"({100*C2.earn.mean():.1f}%)")
    if C2.earn.any():
        print(f"    with earnings: mean {100*C2[C2.earn]['ret'].mean():+.3f}%, "
              f"worst {100*C2[C2.earn]['ret'].min():.2f}%")
        print(f"    without:       mean {100*C2[~C2.earn]['ret'].mean():+.3f}%, "
              f"worst {100*C2[~C2.earn]['ret'].min():.2f}%")
    print("\n  LIVE: TJX next earnings per the calendar:",
          [str(pd.Timestamp(x).date()) for x in emap.get("TJX", [])
           if pd.Timestamp(x) > pd.Timestamp("2026-08-28")][:2])
    print(f"\n  runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
