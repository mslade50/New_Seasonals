"""Flow-conditional sizing, part 1: does signal density forecast next-trade expectancy?

Usage: python flow_conditional_02_state_vs_edge.py [fills|candidates]
  fills       - flow counted from FILLED trades (ledger; biased toward fills)
  candidates  - flow counted from the engine's raw candidate signal-dates (staged, incl. unfilled)

Outputs scratch/ultracode_sizing_2026-09-02/flow_conditional_edge_<src>.json
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from flow_conditional_lib import (load_ledger, load_candidates, attach_flow, attach_open_legs, episodes,
                                  cluster_boot_diff, spearman, FAMILIES, FAMILY, OUT_DIR, NAV)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
SRC = sys.argv[1] if len(sys.argv) > 1 else "fills"
OUT: dict = {"source": SRC}

tr = load_ledger()
print(f"trades {len(tr)}  cap_scale<0.999: {(tr.cap_scale < 0.999).mean():.3%}  years {tr.year.min()}-{tr.year.max()}")
if SRC == "candidates":
    cand = load_candidates()
    assert cand is not None, "run flow_conditional_01_dump_candidates.py first"
    sig = cand.rename(columns={"signal_date": "signal_date", "strategy": "strategy"})
else:
    sig = tr.rename(columns={"Signal Date": "signal_date", "Strategy": "strategy"})[["signal_date", "strategy", "Ticker", "Tier"]]
tr = attach_flow(tr, sig)
tr = attach_open_legs(tr)
tr = tr[tr.year >= 2005].copy()           # 2003-04 = warm-up, few strategies live
tr["risk_wt"] = tr["Risk"] / NAV * 1e4     # bps deployed

# per-family episode ids on signal dates (for clustered SEs)
tr["ep"] = 0
for f in FAMILIES:
    m = tr.family == f
    tr.loc[m, "ep"] = episodes(tr.loc[m, "Signal Date"]) + 100000 * FAMILIES.index(f)

FLOW_VARS = ["s1", "s5", "s21", "s63", "f1", "f5", "f21", "f63", "b5", "b21", "nstrat1", "nstrat5", "open_s", "open_f", "open_b", "s21_rel", "f21_rel"]


def bucketize(x: pd.Series, q=(1 / 3, 2 / 3)) -> pd.Series:
    """Tercile buckets using quantiles of the DISTINCT-value-friendly rank (ties -> lower bucket)."""
    if x.nunique() < 3:
        return pd.Series(np.where(x > x.median(), "hi", "lo"), index=x.index)
    lo, hi = x.quantile(q[0]), x.quantile(q[1])
    return pd.Series(np.where(x <= lo, "lo", np.where(x <= hi, "mid", "hi")), index=x.index)


def cell_stats(g: pd.DataFrame) -> dict:
    return dict(N=int(len(g)), avgR=float(g.R.mean()), sdR=float(g.R.std()), win=float((g.R > 0).mean()),
                r_per_risk=float(g.PnL.sum() / g.Risk.sum()), pnl=float(g.PnL.sum()), risk_bps_mean=float(g.risk_wt.mean()),
                episodes=int(g.ep.nunique()), cap_scale_mean=float(g.cap_scale.mean()))


# ------------------------------------------------------------------ A. per-family and per-strategy buckets
print("\n=== A. avgR by flow tercile (within group), cluster-bootstrap t of hi-lo (episodes) ===")
rows = []
for level, col in [("family", "family"), ("strategy", "Strategy")]:
    for name, g in tr.groupby(col):
        if len(g) < 40:
            continue
        for v in FLOW_VARS:
            x = g[v]
            if x.isna().all():
                continue
            b = bucketize(x.fillna(x.median()))
            cells = {k: cell_stats(gg) for k, gg in g.groupby(b)}
            hi, lo = g[b == "hi"], g[b == "lo"]
            cb = cluster_boot_diff(hi.R.values, hi.ep.values, lo.R.values, lo.ep.values, n=600, seed=1)
            rows.append(dict(level=level, name=name, var=v, N=len(g), rho=spearman(x, g.R),
                             lo_N=cells.get("lo", {}).get("N"), lo_avgR=cells.get("lo", {}).get("avgR"), lo_rpr=cells.get("lo", {}).get("r_per_risk"),
                             mid_avgR=cells.get("mid", {}).get("avgR"),
                             hi_N=cells.get("hi", {}).get("N"), hi_avgR=cells.get("hi", {}).get("avgR"), hi_rpr=cells.get("hi", {}).get("r_per_risk"),
                             hi_sdR=cells.get("hi", {}).get("sdR"), lo_sdR=cells.get("lo", {}).get("sdR"),
                             hi_cap=cells.get("hi", {}).get("cap_scale_mean"), lo_cap=cells.get("lo", {}).get("cap_scale_mean"),
                             hi_ep=cells.get("hi", {}).get("episodes"), lo_ep=cells.get("lo", {}).get("episodes"),
                             diff=cb["diff"], t_cl=cb["t"], p_cl=cb["p"], thr_lo=float(x.quantile(1 / 3)), thr_hi=float(x.quantile(2 / 3))))
A = pd.DataFrame(rows)
fam_view = A[A.level == "family"].pivot(index="name", columns="var", values="t_cl").reindex(FAMILIES)
print("family-level cluster t (hi - lo avgR) by flow variable:\n", fam_view[FLOW_VARS].round(2).to_string())
print("\nfamily-level Spearman(flow, R):\n", A[A.level == "family"].pivot(index="name", columns="var", values="rho").reindex(FAMILIES)[FLOW_VARS].round(3).to_string())
strat_view = A[A.level == "strategy"].pivot(index="name", columns="var", values="t_cl")
print("\nstrategy-level cluster t (hi - lo avgR):\n", strat_view[FLOW_VARS].round(2).to_string())
OUT["buckets"] = A.round(4).to_dict("records")

# detail tables for the headline variables
for v in ["s1", "s5", "f5", "f21", "b5", "open_s"]:
    print(f"\n--- {v}: family cells (lo / mid / hi) ---")
    sub = A[(A.level == "family") & (A["var"] == v)].set_index("name").reindex(FAMILIES)
    print(sub[["N", "thr_lo", "thr_hi", "lo_N", "lo_avgR", "mid_avgR", "hi_N", "hi_avgR", "lo_rpr", "hi_rpr", "hi_cap", "diff", "t_cl", "p_cl", "hi_ep", "lo_ep"]].round(3).to_string())

# ------------------------------------------------------------------ B. year-by-year sign stability (LOYO-style)
print("\n=== B. per-year Spearman(flow, R) sign stability by family (years with >= 15 trades) ===")
stab = []
for f in FAMILIES:
    g = tr[tr.family == f]
    for v in FLOW_VARS:
        yrs = []
        for y, gy in g.groupby("year"):
            if len(gy) >= 15 and gy[v].nunique() > 1:
                yrs.append(spearman(gy[v], gy.R))
        yrs = [r for r in yrs if not np.isnan(r)]
        if len(yrs) >= 5:
            stab.append(dict(family=f, var=v, years=len(yrs), pos_share=float(np.mean(np.array(yrs) > 0)), mean_rho=float(np.mean(yrs)),
                             t_years=float(np.mean(yrs) / (np.std(yrs, ddof=1) / np.sqrt(len(yrs)))) if len(yrs) > 1 else np.nan))
B = pd.DataFrame(stab)
print(B.pivot(index="family", columns="var", values="pos_share").reindex(FAMILIES)[[v for v in FLOW_VARS if v in B["var"].values]].round(2).to_string())
print("\nmean per-year rho:\n", B.pivot(index="family", columns="var", values="mean_rho").reindex(FAMILIES)[[v for v in FLOW_VARS if v in B["var"].values]].round(3).to_string())
OUT["year_stability"] = B.round(4).to_dict("records")

# ------------------------------------------------------------------ C. same-day count: exact-count cells (the cap's own variable)
print("\n=== C. same-day strategy signal count (s1) exact cells, per strategy (2005+) ===")
rows = []
for s, g in tr.groupby("Strategy"):
    for lab, m in [("1", g.s1 == 1), ("2", g.s1 == 2), ("3-4", (g.s1 >= 3) & (g.s1 <= 4)), ("5-8", (g.s1 >= 5) & (g.s1 <= 8)), ("9+", g.s1 >= 9)]:
        gg = g[m]
        if len(gg) >= 8:
            rows.append(dict(strategy=s, s1=lab, **cell_stats(gg)))
C = pd.DataFrame(rows)
print(C.pivot(index="strategy", columns="s1", values="avgR")[["1", "2", "3-4", "5-8", "9+"]].round(2).to_string())
print("\nN:\n", C.pivot(index="strategy", columns="s1", values="N")[["1", "2", "3-4", "5-8", "9+"]].to_string())
print("\ncap_scale mean:\n", C.pivot(index="strategy", columns="s1", values="cap_scale_mean")[["1", "2", "3-4", "5-8", "9+"]].round(2).to_string())
OUT["same_day_cells"] = C.round(4).to_dict("records")

# family same-day count of strategies firing
print("\n=== C2. number of DISTINCT strategies firing on the signal day (nstrat1), by family ===")
rows = []
for f, g in tr.groupby("family"):
    for lab, m in [("1", g.nstrat1 <= 1), ("2", g.nstrat1 == 2), ("3", g.nstrat1 == 3), ("4+", g.nstrat1 >= 4)]:
        gg = g[m]
        if len(gg) >= 8:
            rows.append(dict(family=f, nstrat1=lab, **cell_stats(gg)))
C2 = pd.DataFrame(rows)
print(C2.pivot(index="family", columns="nstrat1", values="avgR").reindex(FAMILIES)[["1", "2", "3", "4+"]].round(2).to_string())
print(C2.pivot(index="family", columns="nstrat1", values="N").reindex(FAMILIES)[["1", "2", "3", "4+"]].to_string())
OUT["nstrat_cells"] = C2.round(4).to_dict("records")

# ------------------------------------------------------------------ D. era split for the headline family results
print("\n=== D. era split (2005-15 vs 2016-26) of hi-vs-lo for s5 / f5 / f21 / open_s by family ===")
rows = []
for f in FAMILIES:
    for v in ["s1", "s5", "f5", "f21", "b5", "open_s"]:
        g = tr[tr.family == f]
        b = bucketize(g[v].fillna(g[v].median()))
        for era, me in [("2005-15", g.year <= 2015), ("2016-26", g.year >= 2016)]:
            hi, lo = g[(b == "hi") & me], g[(b == "lo") & me]
            if len(hi) >= 10 and len(lo) >= 10:
                cb = cluster_boot_diff(hi.R.values, hi.ep.values, lo.R.values, lo.ep.values, n=400, seed=2)
                rows.append(dict(family=f, var=v, era=era, hi_N=len(hi), lo_N=len(lo), hi_avgR=hi.R.mean(), lo_avgR=lo.R.mean(), diff=cb["diff"], t_cl=cb["t"]))
D = pd.DataFrame(rows)
print(D.round(3).to_string(index=False))
OUT["era_split"] = D.round(4).to_dict("records")

tr.to_parquet(OUT_DIR / f"flow_trades_{SRC}.parquet", index=False)
json.dump(OUT, open(OUT_DIR / f"flow_conditional_edge_{SRC}.json", "w"), indent=1, default=float)
print(f"\nwrote flow_conditional_edge_{SRC}.json and flow_trades_{SRC}.parquet")
