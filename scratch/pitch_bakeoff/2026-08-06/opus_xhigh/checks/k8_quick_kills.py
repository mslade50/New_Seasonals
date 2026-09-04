"""K8 — Quick kills, verified with numbers so they can be journalled.

(i)  LONG AAPL / SHORT QQQ when AAPL 5d-return rank <= 5th pctile while QQQ's
     is >= 95th (today: AAPL 1.2, QQQ 100.0). Confirm the N, then broaden to
     any of AAPL MSFT NVDA AMZN GOOGL META vs QQQ at <=10 / >=90 and test
     whether "mega-cap laggard snaps back to its index" has edge at h3/h5/h10.
(ii) LONG SLV / SHORT GLD after a joint >2% up-day while SLV's 63d return
     trails GLD's by more than 6pp (today: -8.1pp). Confirm h3 N=50 avg +0.98
     t=2.09 all-days / episodes N=36 t=1.31, h5/h10 fading, and give the era
     split.

All data truncated to bars strictly before 2026-08-06.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

MEGA = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
HORIZONS = [3, 5, 10]


def pair_fwd(a: pd.Series, b: pd.Series, k: int) -> pd.Series:
    """Long a / short b, close-to-close over k sessions, in percentage points."""
    return C.fwd(a, k) - C.fwd(b, k)


def pair_fwd_open(da: pd.DataFrame, db: pd.DataFrame, k: int) -> pd.Series:
    return C.fwd_from_next_open(da, k) - C.fwd_from_next_open(db, k)


def episodes(dates, gap_td: int = 10) -> list[list[pd.Timestamp]]:
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) == 0:
        return []
    eps, cur = [], [dates[0]]
    for prev, d in zip(dates[:-1], dates[1:]):
        if (d - prev).days > gap_td * 1.6:
            eps.append(cur)
            cur = [d]
        else:
            cur.append(d)
    eps.append(cur)
    return eps


def ep_stats(tab: pd.DataFrame, mask: pd.Series, col: str, gap_td: int = 10) -> dict:
    d = tab.index[mask]
    eps = episodes(d, gap_td)
    vals = [tab.loc[e, col].mean(skipna=True) for e in eps]
    vals = [v for v in vals if np.isfinite(v)]
    return {"n_eps": len(vals), "ep_avg": round(float(np.mean(vals)), 3) if vals else np.nan,
            "ep_t": round(C.tstat(np.array(vals)), 2) if len(vals) > 1 else np.nan}


# ===================================================================== (i)
def part_i() -> None:
    print("=" * 78)
    print("(i) MEGA-CAP LAGGARD vs QQQ  —  long laggard / short QQQ")
    print("=" * 78)
    px = C.load(MEGA + ["QQQ"])
    qqq = px["QQQ"]
    qc = qqq["Close"]
    q_r5 = C.ret(qc, 5)
    q_rank = C.pct_rank(q_r5, 252)

    print(f"\n  QQQ 5d ret rank today: {q_rank.iloc[-1]:.1f} (5d ret {q_r5.iloc[-1]:+.2f}%)")
    for t in MEGA:
        if t not in px:
            continue
        s = px[t]["Close"]
        r = C.ret(s, 5)
        rk = C.pct_rank(r, 252)
        print(f"  {t:6s} 5d ret {r.iloc[-1]:+6.2f}%  rank {rk.iloc[-1]:5.1f}  "
              f"first bar {s.index.min().date()}")

    # ---- strict AAPL-only version ----
    print("\n  --- STRICT: AAPL rank <= 5 AND QQQ rank >= 95 ---")
    a = px["AAPL"]
    ac = a["Close"]
    tab = pd.DataFrame({"a_rank": C.pct_rank(C.ret(ac, 5), 252),
                        "q_rank": q_rank}).dropna()
    for k in HORIZONS:
        tab[f"p{k}"] = pair_fwd(ac, qc, k)
        tab[f"po{k}"] = pair_fwd_open(a, qqq, k)
    tab = tab.dropna(subset=["a_rank", "q_rank"])
    m = (tab["a_rank"] <= 5) & (tab["q_rank"] >= 95)
    print(f"  sample {tab.index.min().date()} .. {tab.index.max().date()}  n={len(tab)}")
    print(f"  STRICT N = {int(m.sum())}   dates: "
          f"{', '.join(str(d.date()) for d in tab.index[m])}")
    if m.sum():
        rows = []
        for k in HORIZONS:
            rows.append({"h": k, **C.describe("AAPL<=5 & QQQ>=95", tab.loc[m, f"p{k}"], tab[f"p{k}"])})
        C.show(rows)

    # relaxed AAPL-only, several grids
    print("\n  --- AAPL-only, threshold grid (long AAPL / short QQQ, close->close) ---")
    rows = []
    for lo, hi in [(5, 95), (10, 90), (15, 85), (20, 80), (25, 75)]:
        mm = (tab["a_rank"] <= lo) & (tab["q_rank"] >= hi)
        r = {"aapl<=": lo, "qqq>=": hi, "n": int(mm.sum()),
             **{kk: vv for kk, vv in ep_stats(tab, mm, "p5").items()}}
        for k in HORIZONS:
            x = tab.loc[mm, f"p{k}"].dropna()
            r[f"h{k}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
            r[f"h{k}_t"] = round(C.tstat(x), 2) if len(x) > 1 else np.nan
            r[f"h{k}_hit"] = round(float((x > 0).mean() * 100), 1) if len(x) else np.nan
        rows.append(r)
    C.show(rows)

    # ---- BROADENED: pooled across the 6 mega-caps ----
    print("\n  --- BROADENED: any of " + " ".join(MEGA) + " vs QQQ ---")
    for lo, hi in [(5, 95), (10, 90), (15, 85), (20, 80)]:
        recs = []
        for t in MEGA:
            if t not in px:
                continue
            d = px[t]
            s = d["Close"]
            rk = C.pct_rank(C.ret(s, 5), 252)
            sub = pd.DataFrame({"rank": rk, "q_rank": q_rank}).dropna()
            for k in HORIZONS:
                sub[f"p{k}"] = pair_fwd(s, qc, k)
                sub[f"po{k}"] = pair_fwd_open(d, qqq, k)
            mm = (sub["rank"] <= lo) & (sub["q_rank"] >= hi)
            for dt in sub.index[mm]:
                rec = {"date": dt, "ticker": t}
                for k in HORIZONS:
                    rec[f"p{k}"] = sub.loc[dt, f"p{k}"]
                    rec[f"po{k}"] = sub.loc[dt, f"po{k}"]
                recs.append(rec)
        pool = pd.DataFrame(recs)
        if pool.empty:
            print(f"   lo={lo} hi={hi}: N=0")
            continue
        pool = pool.sort_values("date").set_index("date")
        print(f"\n   [rank <= {lo} & QQQ >= {hi}]  N={len(pool)}  "
              f"{pool.index.min().date()} .. {pool.index.max().date()}  "
              f"tickers: {dict(pool['ticker'].value_counts())}")
        rows = []
        for k in HORIZONS:
            rows.append({"h": k, **C.describe("pooled (close->close)", pool[f"p{k}"])})
            rows.append({"h": k, **C.describe("pooled (MOO next)", pool[f"po{k}"])})
        C.show(rows)
        # day-level declustering: collapse to one obs per DATE (equal-weight basket)
        for k in HORIZONS:
            daily = pool.groupby(pool.index)[f"p{k}"].mean()
            dailyo = pool.groupby(pool.index)[f"po{k}"].mean()
            eps = episodes(daily.index, 10)
            epv = np.array([daily.loc[e].mean(skipna=True) for e in eps])
            epv = epv[np.isfinite(epv)]
            print(f"    h{k}: date-collapsed N={daily.notna().sum()} avg {daily.mean():+.3f} "
                  f"t {C.tstat(daily.to_numpy()):+.2f} | episodes N={len(epv)} "
                  f"avg {epv.mean():+.3f} t {C.tstat(epv):+.2f} | MOO-next date-collapsed "
                  f"avg {dailyo.mean():+.3f} t {C.tstat(dailyo.to_numpy()):+.2f}")
        # era split on h5
        print("    era split (h5, pooled close->close):")
        C.show(C.era_split(pool.index, pool["p5"].to_numpy(), cut="2018-01-01"))
        # per-year
        yr = pool.groupby(pool.index.year)["p5"]
        print("    per-year h5: " + "  ".join(
            f"{y}:n{int(n)}/{v:+.2f}" for (y, v), (_, n) in zip(yr.mean().items(), yr.size().items())))

    # ---- THE DECISIVE TEST: does the edge survive at TODAY'S gap magnitude? ----
    print("\n  --- GAP-MAGNITUDE CONDITIONING (the decisive test) ---")
    print("  Today's gap: AAPL 5d -8.04% vs QQQ 5d +8.40% = -16.44pp. If the loose")
    print("  cell's edge lives in SMALL gaps and dies/reverses in big ones, the")
    print("  broadened version does NOT rescue today's setup.")
    recs = []
    for t in MEGA:
        if t not in px:
            continue
        d = px[t]
        s = d["Close"]
        r5 = C.ret(s, 5)
        rk = C.pct_rank(r5, 252)
        sub = pd.DataFrame({"rank": rk, "q_rank": q_rank, "gap": r5 - q_r5}).dropna()
        for k in HORIZONS:
            sub[f"p{k}"] = pair_fwd(s, qc, k)
            sub[f"po{k}"] = pair_fwd_open(d, qqq, k)
        mm = (sub["rank"] <= 20) & (sub["q_rank"] >= 80)
        for dt in sub.index[mm]:
            rec = {"date": dt, "ticker": t, "gap": sub.loc[dt, "gap"]}
            for k in HORIZONS:
                rec[f"p{k}"] = sub.loc[dt, f"p{k}"]
                rec[f"po{k}"] = sub.loc[dt, f"po{k}"]
            recs.append(rec)
    pool = pd.DataFrame(recs).sort_values("date").set_index("date")
    print(f"\n  base cell (rank<=20 & QQQ>=80) N={len(pool)}, gap distribution:")
    print("   " + pool["gap"].describe(percentiles=[.05, .1, .25, .5, .75, .9]).round(2).to_string().replace("\n", "  "))
    print(f"   today's gap -16.44pp sits at the {100*(pool['gap'] < -16.44).mean():.1f}th "
          f"pctile of that cell (N below = {int((pool['gap'] < -16.44).sum())})")
    rows = []
    for lo, hi, tag in [(-999, -12, "gap < -12pp (today-like)"),
                        (-12, -8, "-12 .. -8pp"),
                        (-8, -5, "-8 .. -5pp"),
                        (-5, 999, "gap > -5pp (mild)")]:
        mm = (pool["gap"] >= lo) & (pool["gap"] < hi)
        sl = pool[mm]
        r = {"bucket": tag, "n": len(sl),
             "n_dates": int(sl.index.nunique())}
        for k in HORIZONS:
            x = sl[f"p{k}"].dropna()
            xo = sl[f"po{k}"].dropna()
            r[f"h{k}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
            r[f"h{k}_t"] = round(C.tstat(x.to_numpy()), 2) if len(x) > 1 else np.nan
            r[f"h{k}_hit"] = round(float((x > 0).mean() * 100), 1) if len(x) else np.nan
            r[f"h{k}_MOOavg"] = round(float(xo.mean()), 3) if len(xo) else np.nan
        rows.append(r)
    C.show(rows)
    big = pool[pool["gap"] < -12]
    if len(big):
        print("\n   the 'today-like' (gap < -12pp) observations in full:")
        det = big[["ticker", "gap", "p3", "p5", "p10", "po5"]].round(2)
        det.index = det.index.date
        print(det.to_string())
        print("\n   ERA SPLIT of the today-like bucket (this is the kill or the save):")
        rows = []
        for lo, hi, tag in [("2000-01-01", "2003-01-01", "2000-2002 dot-com bust"),
                            ("2003-01-01", "2012-01-01", "2003-2011"),
                            ("2012-01-01", "2018-01-01", "2012-2017"),
                            ("2018-01-01", "2027-01-01", "2018+ (modern)")]:
            sl = big[(big.index >= lo) & (big.index < hi)]
            r = {"era": tag, "n": len(sl), "n_dates": int(sl.index.nunique()),
                 "n_eps": len(episodes(sl.index.unique(), 10)) if len(sl) else 0}
            for k in HORIZONS:
                x = sl[f"p{k}"].dropna()
                xo = sl[f"po{k}"].dropna()
                r[f"h{k}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
                r[f"h{k}_t"] = round(C.tstat(x.to_numpy()), 2) if len(x) > 1 else np.nan
                r[f"h{k}_MOO"] = round(float(xo.mean()), 3) if len(xo) else np.nan
            rows.append(r)
        C.show(rows)
        print("\n   episode-collapsed (gap<-12pp, one obs per >10-session cluster):")
        eps = episodes(big.index.unique(), 10)
        print(f"    {len(eps)} episodes: " + ", ".join(str(e[0].date()) for e in eps))
        for k in HORIZONS:
            epv = np.array([big.loc[e, f"p{k}"].mean(skipna=True) for e in eps])
            epv = epv[np.isfinite(epv)]
            epo = np.array([big.loc[e, f"po{k}"].mean(skipna=True) for e in eps])
            epo = epo[np.isfinite(epo)]
            print(f"    h{k}: N_eps={len(epv)} avg {epv.mean():+.3f} t {C.tstat(epv):+.2f}"
                  f"  | MOO avg {epo.mean():+.3f} t {C.tstat(epo):+.2f}")


# ==================================================================== (ii)
def part_ii() -> None:
    print("\n\n" + "=" * 78)
    print("(ii) LONG SLV / SHORT GLD after a joint >2% up-day, SLV 63d lag > 6pp")
    print("=" * 78)
    px = C.load(["SLV", "GLD"])
    slv, gld = px["SLV"], px["GLD"]
    sc, gc = slv["Close"], gld["Close"]

    tab = pd.DataFrame({
        "slv_d": sc.pct_change() * 100,
        "gld_d": gc.pct_change() * 100,
        "lag63": C.ret(sc, 63) - C.ret(gc, 63),
    }).dropna()
    for k in HORIZONS + [21]:
        tab[f"p{k}"] = pair_fwd(sc, gc, k)
        tab[f"po{k}"] = pair_fwd_open(slv, gld, k)

    print(f"  sample {tab.index.min().date()} .. {tab.index.max().date()}  n={len(tab)}")
    print(f"  TODAY: SLV {tab['slv_d'].iloc[-1]:+.2f}%  GLD {tab['gld_d'].iloc[-1]:+.2f}%  "
          f"63d lag {tab['lag63'].iloc[-1]:+.2f}pp")

    m = (tab["slv_d"] > 2) & (tab["gld_d"] > 2) & (tab["lag63"] < -6)
    print(f"\n  TRIGGER N = {int(m.sum())}  ({int(m.sum())} signal days)")
    eps = episodes(tab.index[m], 10)
    print(f"  distinct episodes (>10 sessions apart): {len(eps)}")
    for i, e in enumerate(eps, 1):
        print(f"   ep{i:2d}  {e[0].date()} .. {e[-1].date()}  n={len(e)}  "
              f"h3 {tab.loc[e,'p3'].mean():+.2f}  h5 {tab.loc[e,'p5'].mean():+.2f}  "
              f"h10 {tab.loc[e,'p10'].mean():+.2f}")

    print("\n  All-days vs episode-collapsed:")
    rows = []
    for k in HORIZONS + [21]:
        rows.append({"h": k, **C.describe("trigger (close->close)", tab.loc[m, f"p{k}"], tab[f"p{k}"])})
        rows.append({"h": k, **C.describe("trigger (MOO next)", tab.loc[m, f"po{k}"], tab[f"po{k}"])})
        epv = [tab.loc[e, f"p{k}"].mean() for e in eps]
        rows.append({"h": k, **C.describe("episode-mean", epv, tab[f"p{k}"])})
        first = pd.DatetimeIndex([e[0] for e in eps])
        rows.append({"h": k, **C.describe("episode-first", tab.loc[first, f"p{k}"], tab[f"p{k}"])})
        rows.append({"h": k, **C.describe("BASELINE all days", tab[f"p{k}"])})
    C.show(rows)

    print("\n  Era split (close->close):")
    for k in HORIZONS:
        print(f"   h={k}")
        C.show(C.era_split(tab.index[m], tab.loc[m, f"p{k}"].to_numpy(), cut="2016-01-01"))

    print("\n  Per-year (h3 / h5, close->close):")
    sub = tab.loc[m]
    out = pd.DataFrame({
        "n": sub.groupby(sub.index.year).size(),
        "h3_avg": sub.groupby(sub.index.year)["p3"].mean().round(3),
        "h5_avg": sub.groupby(sub.index.year)["p5"].mean().round(3),
        "h10_avg": sub.groupby(sub.index.year)["p10"].mean().round(3),
    })
    print(out.to_string())

    print("\n  Leave-one-EPISODE-out on h3 (all-days basis):")
    rows = []
    full = tab.loc[m, "p3"].dropna()
    print(f"   full: n={len(full)} avg {full.mean():+.3f} t {C.tstat(full.to_numpy()):+.2f}")
    for i, e in enumerate(eps, 1):
        keep = tab.loc[m, "p3"].drop(index=[d for d in e], errors="ignore").dropna()
        rows.append({"drop_ep": f"{i}:{e[0].date()}", "n": len(keep),
                     "avg": round(float(keep.mean()), 3), "t": round(C.tstat(keep.to_numpy()), 2)})
    C.show(rows)

    print("\n  Reconciliation with the triage's declustering (_common.declusterize, gap_td=5):")
    d5 = tab.index[m]
    keep = C.declusterize(d5, gap_td=5)
    for k in HORIZONS:
        x = tab.loc[d5[keep], f"p{k}"].dropna()
        xo = tab.loc[d5[keep], f"po{k}"].dropna()
        print(f"   h{k}: declusterize-first N={len(x)} avg {x.mean():+.3f} t {C.tstat(x.to_numpy()):+.2f}"
              f"   | MOO-next N={len(xo)} avg {xo.mean():+.3f} t {C.tstat(xo.to_numpy()):+.2f}")

    print("\n  Threshold sensitivity (joint up-day %, 63d lag pp) on h3:")
    rows = []
    for up in [1.5, 2.0, 2.5]:
        for lag in [-4, -6, -8]:
            mm = (tab["slv_d"] > up) & (tab["gld_d"] > up) & (tab["lag63"] < lag)
            x = tab.loc[mm, "p3"].dropna()
            e2 = episodes(tab.index[mm], 10)
            epv = [tab.loc[ee, "p3"].mean() for ee in e2]
            rows.append({"up>": up, "lag<": lag, "n": int(mm.sum()),
                         "avg": round(float(x.mean()), 3) if len(x) else np.nan,
                         "t": round(C.tstat(x.to_numpy()), 2) if len(x) > 1 else np.nan,
                         "n_eps": len(e2),
                         "ep_avg": round(float(np.mean(epv)), 3) if epv else np.nan,
                         "ep_t": round(C.tstat(np.array(epv)), 2) if len(epv) > 1 else np.nan})
    C.show(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    part_i()
    part_ii()
