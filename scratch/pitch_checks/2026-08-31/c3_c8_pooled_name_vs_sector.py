"""C8 -- POOLED: a name at 21d rank <= 2 while its own sector ETF is at r21 >= 75.

Trade LONG the name, MOC the next session (lag=1), h scanned 1..10.
Pooled over the whole mapped single-name universe, not claimed for TJX.

Hazards the brief requires handling:
 1. overlapping observations -> cluster by DATE, report clustered beside naive
 2. survivorship (master_prices holds today's survivors only) -> direction+size
 3. sector mapping coverage
 4. gate attribution: bare name r21<=2 vs joint (is the sector clause real?)
 5. book overlap against data/backtest_trades_full.parquet
 6. cost 10 bps single-name round trip, need >=5x
 7. era, concentration by date AND by name, sign test on declustered dates
 8. earnings hazard inside the hold
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
NAME_MAX = 2.0
SEC_MIN = 75.0
H_MAIN = 5
MIN_BARS = 500


def rank21(close: pd.Series) -> pd.Series:
    return pct_rank(close, 21)


def main():
    t0 = time.time()
    sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    mp["date"] = pd.to_datetime(mp["date"])
    have = set(mp["ticker"].unique())

    smap = {r.ticker: SECTOR_ETF[r.sector]
            for r in sm.itertuples()
            if r.sector in SECTOR_ETF and r.ticker in have}
    etfs = sorted(set(SECTOR_ETF.values()))
    names = sorted(t for t in smap if t not in etfs)

    print("=" * 78)
    print("C8  pooled: name r21 <= 2 while its sector ETF r21 >= 75")
    print("=" * 78)
    print(f"\n3. SECTOR MAPPING COVERAGE")
    print(f"  master_prices tickers: {len(have)}")
    print(f"  sector_map rows: {len(sm)}; mapped to a SPDR sector AND in cache: "
          f"{len(smap)}")
    print(f"  single names after removing the ETFs themselves: {len(names)}")
    cov = pd.Series([smap[n] for n in names]).value_counts()
    print("  per-ETF member counts:", cov.to_dict())
    unmapped = sorted(have - set(smap) - set(etfs))
    print(f"  UNMAPPED cache tickers (dropped): {len(unmapped)} "
          f"e.g. {unmapped[:12]}")

    wide = mp.pivot_table(index="date", columns="ticker", values="Close",
                          aggfunc="last").sort_index()
    keep = [n for n in names if wide[n].notna().sum() >= MIN_BARS]
    print(f"  names with >= {MIN_BARS} bars: {len(keep)}")

    print("\n  computing r21 for names + ETFs ...")
    etf_rk = {e: rank21(wide[e]) for e in etfs if e in wide.columns}
    name_rk = {}
    for i, n in enumerate(keep):
        name_rk[n] = rank21(wide[n])
        if i % 200 == 0:
            print(f"    {i}/{len(keep)}  ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.0f}s")

    idx = wide.index

    def fwd(n, h):
        c = wide[n].dropna()
        return (c.shift(-(1 + h)) / c.shift(-1) - 1.0).reindex(idx)

    # ------------------------------------------------------------------
    # build the pooled long frame for h=H_MAIN plus the gate variants
    # ------------------------------------------------------------------
    recs = []
    for n in keep:
        e = smap[n]
        if e not in etf_rk:
            continue
        nr = name_rk[n]
        er = etf_rk[e].reindex(idx)
        f = fwd(n, H_MAIN)
        m = nr.notna() & er.notna() & f.notna()
        if not m.any():
            continue
        sub = pd.DataFrame({"date": idx[m.values], "name": n, "etf": e,
                            "nr": nr[m].values, "er": er[m].values,
                            "ret": f[m].values})
        recs.append(sub)
    D = pd.concat(recs, ignore_index=True)
    print(f"\n  pooled name-day observations with a valid h={H_MAIN} return: "
          f"{len(D):,} over {D['date'].nunique():,} dates, {D['name'].nunique()} names")

    base_all = D["ret"].mean()
    print(f"  UNIVERSE BASE RATE (all mapped name-days, h={H_MAIN}, lag=1): "
          f"{100*base_all:+.3f}%")

    # ------------------------------------------------------------------
    # 4. GATE ATTRIBUTION
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. GATE ATTRIBUTION -- does the sector-strength clause filter?")
    print("-" * 78)

    def clustered(sub, label):
        """Date-clustered statistic: date means as the unit."""
        if len(sub) == 0:
            return {"label": label, "n": 0}
        g = sub.groupby("date")["ret"].mean()
        s = summarize(g.values, label)
        s["n_days"] = len(sub)
        s["n_dates"] = len(g)
        s["excess_pp"] = s["mean_pct"] - 100 * base_all
        naive = summarize(sub["ret"].values, "")
        s["naive_t"] = naive["t"]
        return s

    cells = {
        "(a) name r21<=2 ALONE": D[D.nr <= NAME_MAX],
        "(b) sector r21>=75 ALONE": D[D.er >= SEC_MIN],
        "(c) JOINT name<=2 & sector>=75": D[(D.nr <= NAME_MAX) & (D.er >= SEC_MIN)],
        "(d) ANTI name<=2 & sector<25": D[(D.nr <= NAME_MAX) & (D.er < 25)],
        "(e) name<=2 & sector 25-75": D[(D.nr <= NAME_MAX) & (D.er >= 25) & (D.er < SEC_MIN)],
    }
    rows = [clustered(v, k) for k, v in cells.items()]
    rows.append({**clustered(D, "CTRL all mapped name-days")})
    show(rows, f"gate attribution, h={H_MAIN}, DATE-CLUSTERED (naive_t shown for contrast)")
    a = rows[0]["mean_pct"]; c = rows[2]["mean_pct"]
    print(f"\n  sector clause is worth {c-a:+.3f}pp over the bare name cell "
          f"(threshold for 'not decoration' is ~+0.15pp)")
    print(f"  joint excess over the universe base = {rows[2]['excess_pp']:+.3f}pp")

    # dose response across sector-rank buckets
    print("\n  DOSE RESPONSE: name r21<=2, bucketed by its sector ETF r21")
    sub = D[D.nr <= NAME_MAX]
    br = []
    for lo, hi in [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]:
        s = sub[(sub.er >= lo) & (sub.er < hi)]
        r = clustered(s, f"sector r21 [{lo},{hi})")
        br.append(r)
    show(br, "sector-rank dose response (date-clustered)")

    # ------------------------------------------------------------------
    # 1. OVERLAP / CLUSTERING detail + horizon scan
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. OVERLAPPING OBSERVATIONS -- naive vs date-clustered vs declustered")
    print("-" * 78)
    J = cells["(c) JOINT name<=2 & sector>=75"]
    g = J.groupby("date")["ret"].mean()
    print(f"  joint: {len(J):,} name-days on {len(g):,} distinct dates "
          f"({len(J)/max(len(g),1):.2f} names per firing date)")
    print(f"  naive pooled t = {summarize(J['ret'].values,'')['t']:.2f}   "
          f"date-clustered t = {summarize(g.values,'')['t']:.2f}")
    dec = declusters(pd.DatetimeIndex(g.index), H_MAIN, idx)
    gd = g.loc[dec]
    sd = summarize(gd.values, f"declustered dates min_gap={H_MAIN}")
    wn = int((gd.values > 0).sum())
    print(f"  declustered by date (min_gap {H_MAIN}): N={sd['n']} mean "
          f"{sd['mean_pct']:+.3f}% hit {sd['hit']:.1f}% t {sd['t']:.2f} "
          f"sign p {sign_test(wn, sd['n']):.4f}")
    for gap in (10, 21):
        d2 = declusters(pd.DatetimeIndex(g.index), gap, idx)
        s2 = summarize(g.loc[d2].values, "")
        w2 = int((g.loc[d2].values > 0).sum())
        print(f"  min_gap {gap:2d}: N={s2['n']:3d} mean {s2['mean_pct']:+.3f}% "
              f"hit {s2['hit']:.1f}% sign p {sign_test(w2, s2['n']):.4f}")

    # date-clustered control on the SAME dates (what did the rest of the
    # universe do on those days?)
    firing = set(g.index)
    rest = D[D["date"].isin(firing)]
    gr = rest.groupby("date")["ret"].mean()
    print(f"\n  SAME-DATE control: on the {len(gr)} firing dates the whole "
          f"mapped universe paid {100*gr.mean():+.3f}% at h={H_MAIN}; the "
          f"trigger names paid {100*g.mean():+.3f}%  -> "
          f"same-date excess {100*(g.mean()-gr.reindex(g.index).mean()):+.3f}pp")

    print("\n  horizon scan h=1..10 (date-clustered, joint cell):")
    hs = []
    for h in range(1, 11):
        rr = []
        for n in keep:
            e = smap[n]
            if e not in etf_rk:
                continue
            nr = name_rk[n]; er = etf_rk[e].reindex(idx)
            m = (nr <= NAME_MAX) & (er >= SEC_MIN)
            if not m.any():
                continue
            f = fwd(n, h)
            mm = m & f.notna()
            if mm.any():
                rr.append(pd.DataFrame({"date": idx[mm.values],
                                        "ret": f[mm].values}))
        if not rr:
            continue
        JJ = pd.concat(rr)
        gg = JJ.groupby("date")["ret"].mean()
        # universe base at this horizon (sampled for speed)
        bb = []
        for n in keep[::5]:
            f = fwd(n, h)
            bb.append(f.dropna().values)
        base_h = np.concatenate(bb).mean()
        s = summarize(gg.values, f"h={h}")
        s["n_days"] = len(JJ)
        s["base_pct"] = 100 * base_h
        s["excess_pp"] = s["mean_pct"] - 100 * base_h
        hs.append(s)
    show(hs, "horizon scan, date-clustered")

    # ------------------------------------------------------------------
    # 2. SURVIVORSHIP
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. SURVIVORSHIP")
    print("-" * 78)
    last_bar = wide[keep].apply(lambda s: s.last_valid_index())
    panel_end = idx[-1]
    stopped = last_bar[last_bar < panel_end - pd.Timedelta(days=30)]
    print(f"  names whose series ENDS before the panel end: {len(stopped)} of "
          f"{len(keep)} ({100*len(stopped)/len(keep):.1f}%)")
    print(f"  master_prices is built from TODAY's universe files, so names "
          f"delisted out of the universe are absent entirely; the repo's own "
          f"ledger caveat puts that at 21 of 22 major 2020s delistings.")
    if len(stopped):
        sn = set(stopped.index)
        Jj = J[J.name.isin(sn)]
        Jo = J[~J.name.isin(sn)]
        show([clustered(Jj, f"trigger days on names that STOP (N={len(Jj)})"),
              clustered(Jo, f"trigger days on full survivors (N={len(Jo)})")],
             "partial-delisting proxy")
    print("  BIAS DIRECTION: positive. The cell buys the most washed-out name "
          "in a strong sector; names that kept falling to zero are the exact "
          "population the cache cannot contain, so the measured mean is an "
          "UPPER BOUND.")

    # ------------------------------------------------------------------
    # 5. BOOK OVERLAP
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. BOOK OVERLAP -- what did the systematic book do on these name-days?")
    print("-" * 78)
    led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    led["Signal Date"] = pd.to_datetime(led["Signal Date"])
    key = set(zip(J["name"], J["date"]))
    led["k"] = list(zip(led["Ticker"], led["Signal Date"]))
    hit = led[led["k"].isin(key)]
    print(f"  ledger trades whose (Ticker, Signal Date) is a C8 trigger: "
          f"{len(hit)} of {len(led)} ledger rows, covering "
          f"{hit['k'].nunique()} of {len(key):,} trigger name-days "
          f"({100*hit['k'].nunique()/max(len(key),1):.2f}%)")
    if len(hit):
        agg = hit.groupby("Strategy").agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
            totR=("R_Multiple", "sum"),
            pnl=("PnL_flat_750k", "sum")).sort_values("n", ascending=False)
        print(agg.round(3).to_string())
    # how enriched is the trigger population in book-eligible dip-buy shape?
    olv = led[led.Strategy.isin(["Oversold Low Volume", "LT Trend ST OS"])]
    olvk = set(olv["k"])
    print(f"  OLV + LT Trend ST OS specifically: "
          f"{len(key & olvk)} trigger name-days are also book signals")

    # ------------------------------------------------------------------
    # 6/7. COST, ERA, CONCENTRATION
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("6+7. COST, ERA, CONCENTRATION")
    print("-" * 78)
    edge_bps = 100 * (g.mean() - base_all) * 100
    print(f"  date-clustered mean {100*g.mean():+.3f}%, excess over universe "
          f"base {100*(g.mean()-base_all):+.3f}pp = {edge_bps:+.1f} bps")
    print(f"  single-name round trip ~10 bps -> {edge_bps/10:.1f}x cost "
          f"(need >= 5x, i.e. >= +0.50pp excess)")

    gg = pd.DataFrame({"ret": g})
    gg["yr"] = gg.index.year
    era = gg.assign(pre=gg.index < pd.Timestamp("2018-01-01"))
    show([summarize(era[era.pre]["ret"].values, "pre-2018 dates"),
          summarize(era[~era.pre]["ret"].values, "2018+ dates")],
         "era split (date-clustered)")
    byyr = gg.groupby("yr")["ret"].agg(["size", "mean"])
    print("\n  by year (date count, date-mean %):")
    print((byyr.assign(mean=(100 * byyr["mean"]).round(2))).to_string())

    cn = J.groupby("name").size().sort_values(ascending=False)
    top10 = cn.head(10)
    print(f"\n  CONCENTRATION BY NAME: {len(cn)} distinct names supply "
          f"{len(J):,} name-days; top 10 = {int(top10.sum()):,} "
          f"({100*top10.sum()/len(J):.1f}%)")
    print("   ", top10.to_dict())
    half = (cn.cumsum() <= len(J) / 2).sum() + 1
    print(f"  names needed to reach half the observations: {half}")
    cd = J.groupby("date").size().sort_values(ascending=False)
    print(f"  CONCENTRATION BY DATE: top 10 dates = {int(cd.head(10).sum()):,} "
          f"name-days ({100*cd.head(10).sum()/len(J):.1f}%): "
          f"{[str(d.date()) for d in cd.head(5).index]}")

    # ------------------------------------------------------------------
    # 8. EARNINGS HAZARD
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("8. EARNINGS INSIDE THE HOLD")
    print("-" * 78)
    ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
    ec["date"] = pd.to_datetime(ec["date"])
    emap = {t: np.sort(gg2["date"].values) for t, gg2 in ec.groupby("ticker")}
    pos = pd.Series(range(len(idx)), index=idx)
    flags = []
    for nm, dt in zip(J["name"].values, J["date"].values):
        arr = emap.get(nm)
        p = pos.get(pd.Timestamp(dt))
        if arr is None or p is None or p + 1 + H_MAIN >= len(idx):
            flags.append(False); continue
        lo = np.datetime64(idx[p + 1]); hi = np.datetime64(idx[p + 1 + H_MAIN])
        flags.append(bool(((arr > lo) & (arr <= hi)).any()))
    J = J.assign(earn=flags)
    print(f"  trigger name-days with an earnings print inside the hold: "
          f"{int(J.earn.sum()):,} of {len(J):,} "
          f"({100*J.earn.mean():.1f}%); earnings coverage for trigger names: "
          f"{100*np.mean([nm in emap for nm in J['name'].unique()]):.1f}%")
    show([clustered(J[J.earn], "earnings IN hold"),
          clustered(J[~J.earn], "earnings OUT")], "earnings split (date-clustered)")
    print(f"  worst single name-day with earnings in hold: "
          f"{100*J[J.earn]['ret'].min():.2f}%; without: "
          f"{100*J[~J.earn]['ret'].min():.2f}%")

    # ------------------------------------------------------------------
    # DIAL
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DIAL DISTRIBUTION of the trigger population (today ma10(63d) 87.6)")
    print("-" * 78)
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma = frag["63d"].rolling(10).mean().reindex(idx).ffill()
    tv = ma.reindex(pd.DatetimeIndex(g.index)).dropna()
    if len(tv):
        print(f"  N firing dates with a dial = {len(tv)} of {len(g)} "
              f"(dial starts {frag.index[0].date()}; pre-2026-07-02 rows are a "
              f"recompute vintage)")
        print(f"  min {tv.min():.1f} p25 {tv.quantile(.25):.1f} median "
              f"{tv.median():.1f} p75 {tv.quantile(.75):.1f} MAX {tv.max():.1f}; "
              f">=85: {int((tv>=85).sum())} dates")
        hi = tv[tv >= 70].index
        if len(hi):
            show([clustered(J[J["date"].isin(set(hi))], f"dial>=70 (N dates {len(hi)})")],
                 "high-dial slice")
    print(f"\n  total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
