"""Cycle/macro regime conditioning, part 1: per-strategy avgR / sdR / N /
clustered t / LOYO at the trade level, and daily-basis Sharpe by regime,
for every regime axis in cycle_macro_lib.REGIME_COLS. Classifies each
(strategy, regime, bucket) cell as edge-moving, variance-moving, both or
neither. Writes cycle_macro_01_tables.json beside this file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import (HERE, NAV, REGIME_COLS, attach_trade_regimes, build_regimes, cluster_t, daily_sharpe,
                             episode_ids, jsonable, load_daily, load_ledger, loyo, welch_t)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {"meta": {}}

led = load_ledger()
R = build_regimes()
T = attach_trade_regimes(led, R)
T = T[T["Signal Date"] >= "2003-01-01"].copy()
strat, tot = load_daily()
OUT["meta"] = dict(trades=int(len(T)), ledger_span=[T["Signal Date"].min(), T["Signal Date"].max()],
                   daily_span=[strat.index.min(), strat.index.max()],
                   dial_vintage="current-weights; rows before 2026-07-02 are the recompute vintage; dial window 2016-07+",
                   trade_regime_timing="signal-date close (P/C lag-1)", daily_regime_timing="lag-1 session")

# regime day shares (for base rates)
share = {c: R[c].value_counts(normalize=True).round(3).to_dict() for c in REGIME_COLS}
OUT["regime_day_share"] = share
print("regime day shares (all SPY sessions 2000+):")
for c in REGIME_COLS:
    print(f"  {c:10s} {share[c]}")

# ---------------------------------------------------------------- trade-level tables
strats = ["ALL"] + sorted(T["Strat2"].unique())
rows = []
R_lag = R.shift(1)  # daily-series regime: yesterday's close
for col in REGIME_COLS:
    ep_all = {}
    for b in sorted(T[col].dropna().unique()):
        if b == "nan":
            continue
        ep_all[b] = episode_ids(R[col] == b, gap=21)
    for s in strats:
        d = T if s == "ALL" else T[T["Strat2"] == s]
        d = d[d[col] != "nan"]
        if len(d) < 20:
            continue
        base_mu, base_sd = d["R_Multiple"].mean(), d["R_Multiple"].std()
        for b in sorted(d[col].unique()):
            m = (d[col] == b)
            g, rest = d[m], d[~m]
            if len(g) < 8 or len(rest) < 5:
                continue
            # cluster: in-regime trades by regime episode, out-of-regime by signal month
            ep = ep_all[b].reindex(d["Signal Date"]).values
            cl = np.where(m.values, "E" + pd.Series(ep).astype(str), "M" + d["ym"].values)
            beta, tcl, G = cluster_t(d["R_Multiple"].values, m.values.astype(float), cl)
            lo = loyo(d, m)
            row = dict(regime=col, bucket=b, strategy=s, N=int(len(g)), N_rest=int(len(rest)), share=float(len(g) / len(d)),
                       avgR=float(g["R_Multiple"].mean()), sdR=float(g["R_Multiple"].std()), win=float((g["R_Multiple"] > 0).mean()),
                       avgR_rest=float(rest["R_Multiple"].mean()) if len(rest) else np.nan,
                       sdR_rest=float(rest["R_Multiple"].std()) if len(rest) > 1 else np.nan,
                       diff=float(beta), t_cluster=float(tcl), n_clusters=int(G), t_welch=welch_t(g["R_Multiple"], rest["R_Multiple"]),
                       sd_ratio=float(g["R_Multiple"].std() / base_sd) if base_sd > 0 else np.nan,
                       kelly_rel=float((g["R_Multiple"].mean() / g["R_Multiple"].var()) / (base_mu / base_sd**2)) if base_mu > 0 and g["R_Multiple"].var() > 0 else np.nan,
                       pnl_flat=float(g["PnL_flat_750k"].sum()), risk_flat=float(g["Risk_flat_750k"].sum()),
                       pnl_per_risk=float(g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum()) if g["Risk_flat_750k"].sum() > 0 else np.nan,
                       size_mult=float(g["Size_Mult"].mean()), years=int(g["yr"].nunique()),
                       first=g["Signal Date"].min(), last=g["Signal Date"].max(), **lo)
            rows.append(row)
TR = pd.DataFrame(rows)
# classification
def classify(r):
    edge = abs(r["t_cluster"]) >= 1.5 and abs(r["diff"]) >= 0.15 and r["N"] >= 15
    var = (r["sd_ratio"] >= 1.25 or r["sd_ratio"] <= 0.8) and r["N"] >= 15
    if edge and var:
        return "edge+variance"
    if edge:
        return "edge"
    if var:
        return "variance"
    return "neither"
TR["class"] = TR.apply(classify, axis=1)
OUT["trade_table"] = jsonable(TR.to_dict("records"))

print("\n=== BOOK (all strategies) by regime: avgR / sdR / N / clustered t / LOYO sign years ===")
bk = TR[TR.strategy == "ALL"][["regime", "bucket", "N", "share", "avgR", "sdR", "win", "diff", "t_cluster", "n_clusters", "sd_ratio", "kelly_rel", "yr_pos", "yr_neg", "class"]]
print(bk.to_string(index=False))

print("\n=== per-strategy cells with |t_cluster| >= 1.5 and N >= 15 (edge candidates) ===")
cand = TR[(TR.strategy != "ALL") & (TR["t_cluster"].abs() >= 1.5) & (TR.N >= 15)].sort_values("t_cluster")
print(cand[["regime", "bucket", "strategy", "N", "share", "avgR", "avgR_rest", "sdR", "diff", "t_cluster", "n_clusters", "sd_ratio", "yr_pos", "yr_neg", "loyo_min", "loyo_max", "size_mult", "class"]].to_string(index=False))

print("\n=== per-strategy cells where VARIANCE moves (sd_ratio >= 1.25 or <= 0.8, N >= 15) but edge does not ===")
vr = TR[(TR.strategy != "ALL") & (TR["class"] == "variance")].sort_values("sd_ratio")
print(vr[["regime", "bucket", "strategy", "N", "avgR", "avgR_rest", "sdR", "sdR_rest", "sd_ratio", "diff", "t_cluster"]].to_string(index=False))

# ---------------------------------------------------------------- daily-basis Sharpe per strategy by regime (lag-1)
print("\n=== daily-basis (active days) Sharpe per strategy by lag-1 regime ===")
drows = []
W = strat[strat.index >= "2003-01-01"]
for col in REGIME_COLS:
    lab = R_lag[col].reindex(W.index)
    for s in list(W.columns) + ["BOOK"]:
        x = tot.reindex(W.index) if s == "BOOK" else W[s]
        for b in sorted(lab.dropna().unique()):
            if b == "nan":
                continue
            xx = x[(lab == b).values]
            if s != "BOOK":
                xx = xx[xx != 0]
            ds = daily_sharpe(xx)
            if ds["days"] >= 20:
                drows.append(dict(regime=col, bucket=b, strategy=s, **ds))
DS = pd.DataFrame(drows)
OUT["daily_table"] = jsonable(DS.to_dict("records"))
piv = DS[DS.strategy == "BOOK"].pivot_table(index=["regime", "bucket"], values=["days", "sharpe", "mean_bps", "sd_bps"])
print(piv.round(2).to_string())
# per-strategy pivot for the main axes
for col in ["cycle", "vix_lvl", "vix_ts", "spy_200", "rv21", "spy_dd", "pc_fear", "credit21"]:
    p = DS[(DS.regime == col) & (DS.strategy != "BOOK")].pivot(index="strategy", columns="bucket", values="sharpe")
    print(f"\n-- daily Sharpe by {col} --\n" + p.round(2).to_string())

# ---------------------------------------------------------------- book-level variance vs edge by regime (daily, lag-1)
print("\n=== book daily sd and mean by regime (lag-1), 2003+ ===")
brows = []
for col in REGIME_COLS:
    lab = R_lag[col].reindex(tot.index)
    for b in sorted(lab.dropna().unique()):
        if b == "nan":
            continue
        x = tot[(lab == b).values]
        if len(x) < 40:
            continue
        brows.append(dict(regime=col, bucket=b, days=len(x), mean_bps=x.mean() * 1e4, sd_bps=x.std() * 1e4, sharpe=x.mean() / x.std() * np.sqrt(252),
                          sd_ratio=x.std() / tot.std(), mean_ratio=x.mean() / tot.mean(), worst=x.min() * 100,
                          cvar5_bps=x[x <= x.quantile(.05)].mean() * 1e4))
BK = pd.DataFrame(brows)
print(BK.round(3).to_string(index=False))
OUT["book_daily_by_regime"] = jsonable(BK.to_dict("records"))

json.dump(jsonable(OUT), open(HERE / "cycle_macro_01_tables.json", "w"), indent=1)
print("\nwrote", HERE / "cycle_macro_01_tables.json")
