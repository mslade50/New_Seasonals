"""C4 -- two-member idiosyncratic shock inside an already-washed-out XLU.

Trigger: >=2 XLU members close <= -4% on the SAME session while XLU's own
21d return rank (trailing 252d) is in the bottom decile. Trade LONG XLU,
MOC the next session (lag=1), h scanned 1..10.

Mandatory tests in the brief, in order:
 1. gate attribution both ways (washout alone / shock alone / joint)
 2. overlap with the CLOSED 2026-08-25 XLU r21<=5 cell
 3. reference class across the nine SPDR sectors (Q, I2, FE, permutation)
 4. threshold neighbours
 5. era / midterm / concentration / sign test / decluster
 6. cost 4 bps one leg, need >=5x
 7. TLT loading + duration-neutral form
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

XLU_MEMBERS = ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "ED", "PEG",
               "PCG", "EIX", "ETR", "FE", "CMS", "DTE", "PPL", "PNW", "CNP"]

SECTOR_ETF = {
    "Utilities": "XLU", "Technology": "XLK", "Healthcare": "XLV",
    "Financial Services": "XLF", "Energy": "XLE", "Industrials": "XLI",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Basic Materials": "XLB",
}

H_MAIN = 5
DROP = -0.04
NMEM = 2
WASH = 10.0


def daily_rets(tickers):
    """Per-ticker 1-session return on the ticker's OWN valid sessions."""
    px = load_prices(tickers)
    out = {}
    for t, g in px.items():
        c = g["Close"].dropna()
        out[t] = (c / c.shift(1) - 1.0)
    return pd.DataFrame(out)


def shock_count(rets, thresh=DROP):
    return (rets <= thresh).sum(axis=1)


def alive_count(rets):
    return rets.notna().sum(axis=1)


def main():
    etfs = sorted(set(SECTOR_ETF.values())) + ["TLT", "SPY"]
    px = close_panel(etfs)
    r21 = pct_rank(px["XLU"], 21)
    mem = daily_rets(XLU_MEMBERS).reindex(px.index)
    nshock = shock_count(mem).reindex(px.index).fillna(0)
    nalive = alive_count(mem).reindex(px.index).fillna(0)

    print("=" * 78)
    print("C4  XLU two-member shock inside a bottom-decile 21d washout")
    print("=" * 78)
    print("\nmembership coverage (names with a bar), by year:")
    print(nalive.groupby(nalive.index.year).max().to_string())

    wash = (r21 <= WASH)
    shock = (nshock >= NMEM)
    joint = wash & shock

    # live instance sanity
    last = px.index[-1]
    print(f"\nlive read {last.date()}: XLU r21 = {r21.iloc[-1]:.1f}, "
          f"members <= -4% = {int(nshock.iloc[-1])} "
          f"({[c for c in mem.columns if mem[c].iloc[-1] <= DROP]}), "
          f"XLU 1d = {100*(px['XLU'].iloc[-1]/px['XLU'].iloc[-2]-1):.2f}%")

    # ---------------- 1. GATE ATTRIBUTION -----------------------------------
    print("\n" + "-" * 78)
    print("1. GATE ATTRIBUTION -- what does each clause buy? (episode level, "
          f"h={H_MAIN}, min_gap={H_MAIN})")
    print("-" * 78)
    ret = fwd_lag(px["XLU"], H_MAIN, lag=1)
    valid = ret.notna()
    base = ret[valid]
    rows = []
    for lbl, m in [("(a) washout ALONE  r21<=10", wash),
                   ("(b) 2-member shock ALONE", shock),
                   ("(c) JOINT cell", joint)]:
        d = px.index[m.values & valid.values]
        epi = declusters(d, H_MAIN, px.index)
        s = summarize(ret.loc[epi].values, lbl)
        s["n_days"] = len(d)
        s["excess_pp"] = s.get("mean_pct", np.nan) - 100 * base.mean()
        rows.append(s)
    rows.append({**summarize(base.values, "CTRL-b XLU all days"), "n_days": len(base),
                 "excess_pp": 0.0})
    show(rows, "gate attribution")
    a = rows[0]["mean_pct"]; b = rows[1]["mean_pct"]; c = rows[2]["mean_pct"]
    print(f"\n  join arithmetic: joint {c:+.3f}%  vs washout-alone {a:+.3f}%  "
          f"vs shock-alone {b:+.3f}%")
    print(f"  join adds {c-a:+.3f}pp over the washout parent and "
          f"{c-b:+.3f}pp over the shock parent")

    # ---------------- 2. OVERLAP WITH THE CLOSED CELL -----------------------
    print("\n" + "-" * 78)
    print("2. OVERLAP WITH THE CLOSED 2026-08-25 CELL (XLU r21<=5) and the")
    print("   registry's dead z10<=-1.5 / rank21<=5 utilities corpses")
    print("-" * 78)
    z10 = zscore(px["XLU"], 10)
    tlt21 = pct_rank(px["TLT"], 21)
    closed5 = (r21 <= 5)
    closed_z = (z10 <= -1.5)
    wl26 = (r21 <= 5) & (tlt21 < 25)
    jd = px.index[joint.values & valid.values]
    for lbl, m in [("XLU r21<=5 (the 08-25 closed cell)", closed5),
                   ("XLU z10<=-1.5 (dead expression #1)", closed_z),
                   ("watchlist 26 rescue (r21<=5 & TLT r21<25)", wl26),
                   ("EITHER dead utilities corpse", closed5 | closed_z)]:
        ov = int(m.reindex(jd, fill_value=False).sum())
        print(f"  {lbl:<45s} overlap {ov:3d} / {len(jd):3d} trigger days "
              f"= {100*ov/max(len(jd),1):5.1f}%")

    # ---------------- 3. REFERENCE CLASS, NINE SPDR SECTORS -----------------
    print("\n" + "-" * 78)
    print("3. REFERENCE CLASS: identical rule on the nine SPDR sectors")
    print("-" * 78)
    sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
    have = set(pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                               columns=["ticker"])["ticker"].unique())
    fam = {}
    for sec, etf in SECTOR_ETF.items():
        names = sorted(set(sm.loc[sm["sector"] == sec, "ticker"]) & have)
        fam[etf] = names
        print(f"  {etf}: {len(names)} mapped members")

    famrows, thetas, vars_, ns, epi_store = [], [], [], [], {}
    for etf, names in fam.items():
        if len(names) < 8:
            continue
        rr = daily_rets(names).reindex(px.index)
        nsh = (rr <= DROP).sum(axis=1)
        rk = pct_rank(px[etf], 21)
        m = (nsh >= NMEM) & (rk <= WASH)
        r = fwd_lag(px[etf], H_MAIN, lag=1)
        v = r.notna()
        d = px.index[m.values & v.values]
        if len(d) < 3:
            famrows.append({"etf": etf, "n_epi": len(d), "note": "too few"})
            continue
        epi = declusters(d, H_MAIN, px.index)
        vals = r.loc[epi].values
        bs = r[v].mean()
        exc = vals - bs           # excess over that ETF's own all-days drift
        epi_store[etf] = (epi, exc, r, v)
        s = summarize(vals, etf)
        famrows.append({"etf": etf, "n_days": len(d), "n_epi": len(epi),
                        "mean_pct": s["mean_pct"], "hit": s["hit"],
                        "base_pct": 100 * bs,
                        "excess_pp": s["mean_pct"] - 100 * bs, "t": s["t"]})
        thetas.append(exc.mean()); vars_.append(exc.var(ddof=1) / len(exc))
        ns.append(etf)
    show(famrows, f"family: >=2 members <= -4% & ETF r21<=10, h={H_MAIN}")

    th = np.array(thetas); vr = np.array(vars_)
    w = 1.0 / vr
    fe = float((w * th).sum() / w.sum())
    Q = float((w * (th - fe) ** 2).sum())
    dfree = len(th) - 1
    I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
    from scipy import stats as sps
    pQ = 1 - sps.chi2.cdf(Q, dfree)
    print(f"\n  fixed-effect common excess = {100*fe:+.3f}pp   "
          f"Cochran Q = {Q:.2f} on {dfree} df (p {pQ:.4f})   I^2 = {I2:.1f}%")
    print(f"  best member = {ns[int(np.argmax(th))]} at {100*max(th):+.3f}pp; "
          f"XLU = {100*th[ns.index('XLU')]:+.3f}pp "
          f"(rank {sorted(th, reverse=True).index(th[ns.index('XLU')])+1} of {len(th)})")

    # permutation max-of-9: random anchor dates, same episode counts
    rng = np.random.default_rng(42)
    NB = 4000
    maxes = np.zeros(NB)
    pools = {}
    for etf in ns:
        epi, exc, r, v = epi_store[etf]
        pools[etf] = (r[v].values - r[v].mean(), len(epi))
    for i in range(NB):
        best = -9e9
        for etf in ns:
            pool, k = pools[etf]
            draw = rng.choice(pool, size=k, replace=False).mean()
            best = max(best, draw)
        maxes[i] = best
    obs = float(max(th))
    pfw = float((maxes >= obs).mean())
    print(f"  permutation max-of-{len(ns)} null: median {100*np.median(maxes):+.3f}pp, "
          f"90th {100*np.quantile(maxes,0.9):+.3f}pp; observed best "
          f"{100*obs:+.3f}pp -> family-wise p = {pfw:.4f}")
    xlu_th = th[ns.index("XLU")]
    print(f"  XLU alone against the same null: p = "
          f"{float((maxes >= xlu_th).mean()):.4f}")

    # ---------------- 4/5/6. FULL BATTERY on the joint cell ----------------
    print("\n" + "-" * 78)
    print("4-6. FULL BATTERY on the joint cell")
    print("-" * 78)
    variants = {
        "member drop <= -3%": (shock_count(mem, -0.03) >= NMEM).reindex(px.index, fill_value=False) & wash,
        "member drop <= -5%": (shock_count(mem, -0.05) >= NMEM).reindex(px.index, fill_value=False) & wash,
        "n members >= 1": (nshock >= 1) & wash,
        "n members >= 3": (nshock >= 3) & wash,
        "washout <= 5 pct": shock & (r21 <= 5),
        "washout <= 20 pct": shock & (r21 <= 20),
        "no washout clause": shock,
    }
    battery(px, joint, [("XLU", 1.0)], H_MAIN,
            "C4 joint: >=2 XLU members <= -4% & XLU r21<=10  LONG XLU",
            cost_bps=4.0, variants=variants, min_gap=H_MAIN)

    # horizon scan
    jd_all = px.index[joint.values]
    print("\n  horizon scan h=1..10 (episodes, excess vs XLU all-days):")
    show(horizon_scan(px, jd_all, [("XLU", 1.0)], hs=tuple(range(1, 11))),
         "horizon scan")

    # midterm split + decluster ladder
    epi = declusters(px.index[joint.values & valid.values], H_MAIN, px.index)
    ev = ret.loc[epi].values
    mid = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(ev[mid], f"midterm (N={int(mid.sum())})"),
          summarize(ev[~mid], f"non-midterm (N={int((~mid).sum())})")],
         "midterm split (episodes)")
    print("\n  decluster ladder:")
    for g in (5, 10, 21):
        e2 = declusters(px.index[joint.values & valid.values], g, px.index)
        s = summarize(ret.loc[e2].values, f"min_gap={g}")
        wn = int((ret.loc[e2].values > 0).sum())
        print(f"   min_gap {g:2d}: N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
              f"hit {s['hit']:.1f}% sign p {sign_test(wn, s['n']):.4f}")

    # ---------------- 7. RATES CONTROL --------------------------------------
    print("\n" + "-" * 78)
    print("7. RATES CONTROL -- TLT loading of the trigger population")
    print("-" * 78)
    tr = fwd_lag(px["TLT"], H_MAIN, lag=1)
    xr = ret
    both = px.index[joint.values & xr.notna().values & tr.notna().values]
    epi2 = declusters(both, H_MAIN, px.index)
    x = tr.loc[epi2].values
    y = xr.loc[epi2].values
    if len(x) > 3:
        beta = np.polyfit(x, y, 1)
        alpha = y.mean() - beta[0] * x.mean()
        print(f"  TLT forward move on trigger episodes: {100*x.mean():+.3f}% "
              f"(TLT all-days {100*tr[tr.notna()].mean():+.3f}%)")
        print(f"  regression XLU_fwd ~ TLT_fwd: beta {beta[0]:+.3f}, "
              f"alpha {100*alpha:+.3f}% vs raw mean {100*y.mean():+.3f}%")
        dn = y - beta[0] * x
        show([summarize(y, "raw long XLU"), summarize(dn, "duration-neutral")],
             "duration-neutral form (episodes)")

    # dial distribution of the trigger population
    print("\n" + "-" * 78)
    print("DIAL DISTRIBUTION of the trigger population (today ma10(63d) = 87.6)")
    print("-" * 78)
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma = frag["63d"].rolling(10).mean()
    dd = ma.reindex(px.index).ffill()
    tv = dd.loc[epi].dropna()
    if len(tv):
        print(f"  N with a dial = {len(tv)} of {len(epi)} episodes "
              f"(dial history starts {frag.index[0].date()}; rows before "
              f"2026-07-02 are a recompute vintage)")
        print(f"  min {tv.min():.1f}  p25 {tv.quantile(.25):.1f}  median "
              f"{tv.median():.1f}  p75 {tv.quantile(.75):.1f}  MAX {tv.max():.1f}")
        print(f"  episodes at dial >= 85: {int((tv >= 85).sum())}")
    else:
        print("  NO trigger episode has a dial reading at all.")


if __name__ == "__main__":
    main()
