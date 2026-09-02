"""Cycle/macro regime conditioning, part 2: validation.
(A) presidential-cycle effect with YEAR as the cluster (6 midterm years vs 18)
    per strategy, plus SPY base rates by cycle year in-sample;
(B) walk-forward per-strategy regime multiplier tables (Kelly-relative,
    shrunk, clipped) fit on trades before year Y and applied to year Y,
    2010-2026, vs a book-wide (pooled) table on the same axis;
(C) daily-series overlays that size UP with vol (the mirror image of the dead
    VIX-inverse overlay), equal-vol Sharpe + LOYO.
Writes cycle_macro_02_validation.json beside this file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import (HERE, NAV, attach_trade_regimes, build_regimes, jsonable, load_daily, load_ledger, load_prices,
                             welch_t)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
OUT: dict = {}
led = load_ledger()
R = build_regimes()
T = attach_trade_regimes(led, R)
strat, tot = load_daily()
px = load_prices()

# ---------------------------------------------------------------- A. cycle with year clusters
print("=== A. presidential cycle: year-level avgR (each year one observation) ===")
T["cyc"] = T["cycle"]
yr_all = T.groupby(["yr", "cyc"])["R_Multiple"].agg(["mean", "size"]).reset_index()
print(yr_all.pivot(index="yr", columns="cyc", values="mean").round(2).to_string())
def year_cluster(d, label):
    y = d.groupby("yr").agg(avgR=("R_Multiple", "mean"), N=("R_Multiple", "size"), cyc=("cyc", "first"))
    y = y[y.N >= 8]
    mid, oth = y[y.cyc == "midterm"], y[y.cyc != "midterm"]
    if len(mid) < 3 or len(oth) < 3:
        return None
    # weighted (by N) and unweighted year means
    return dict(label=label, n_mid_years=int(len(mid)), n_other_years=int(len(oth)), mid_mean=float(mid.avgR.mean()), other_mean=float(oth.avgR.mean()),
                t_year=welch_t(mid.avgR, oth.avgR), mid_years_below_other_mean=int((mid.avgR < oth.avgR.mean()).sum()),
                mid_by_year={int(k): round(float(v), 3) for k, v in mid.avgR.items()}, mid_N_by_year={int(k): int(v) for k, v in mid.N.items()},
                other_by_year={int(k): round(float(v), 3) for k, v in oth.avgR.items()})
rowsA = [year_cluster(T, "BOOK")]
for s, d in T.groupby("Strat2"):
    r = year_cluster(d, s)
    if r:
        rowsA.append(r)
A = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rowsA])
print(A.round(3).to_string(index=False))
OUT["cycle_year_cluster"] = jsonable(rowsA)
# H1 vs H2 of midterm years, year-clustered
mid = T[T.cycle == "midterm"]
h = mid.groupby(["yr", mid["Signal Date"].dt.month.le(6)])["R_Multiple"].mean().unstack()
h.columns = ["H2", "H1"]
print("\nmidterm years, H1 vs H2 avgR by year:\n", h.round(2).to_string())
OUT["midterm_h1_h2"] = jsonable(h.round(3).reset_index().to_dict("records"))
# SPY base rates by cycle year in the sample
spy = px["SPY"].dropna()
ann = spy.resample("YE").last().pct_change().dropna()
ann.index = ann.index.year
cyc = pd.Series([("election", "post_election", "midterm", "pre_election")[y % 4] for y in ann.index], index=ann.index)
spy_cyc = ann.groupby(cyc).agg(["mean", "median", "min", "count"])
print("\nSPY calendar-year return by cycle year (2001+):\n", (spy_cyc * [100, 100, 100, 1]).round(1).to_string())
OUT["spy_by_cycle_year"] = jsonable(spy_cyc.round(4).reset_index().to_dict("records"))
# midterm intra-year drawdown (peak to trough) vs others
def max_dd(s):
    return float((s / s.cummax() - 1).min())
ydd = spy.groupby(spy.index.year).apply(max_dd)
ydd = ydd[ydd.index >= 2001]
print("SPY intra-year maxDD by cycle year:", (ydd.groupby(cyc.reindex(ydd.index)).mean() * 100).round(1).to_dict())
OUT["spy_intra_year_dd_by_cycle"] = jsonable((ydd.groupby(cyc.reindex(ydd.index)).mean()).round(4).to_dict())
# per-strategy trade count by cycle (signal flow)
flow = T.groupby(["Strat2", "cycle"]).size().unstack().fillna(0)
flow_yr = flow.div(pd.Series({c: T[T.cycle == c]["yr"].nunique() for c in flow.columns}), axis=1)
print("\ntrades per year by cycle year (signal flow):\n", flow_yr.round(1).to_string())
OUT["signal_flow_by_cycle"] = jsonable(flow_yr.round(2).reset_index().to_dict("records"))

# ---------------------------------------------------------------- B. walk-forward regime multiplier tables
print("\n=== B. walk-forward per-strategy regime multipliers (fit < Y, apply Y), 2010-2026 ===")
N0, LO, HI = 50, 0.5, 1.5
AXES = ["cycle", "cycle_half", "vix_lvl", "rv21", "spy_dd", "vix_ts", "credit21", "tnx_chg63", "tnx_lvl", "pc_fear", "vol_ratio", "spy_200", "mom12_1", "dial"]
T["exit_d"] = T["Exit Date"]

def fit_table(tr: pd.DataFrame, axis: str, by_strategy: bool) -> dict:
    tab = {}
    groups = tr.groupby("Strat2") if by_strategy else [("ALL", tr)]
    for s, g in groups:
        mu, va = g["R_Multiple"].mean(), g["R_Multiple"].var()
        if mu <= 0 or len(g) < 30:
            continue
        base = mu / va
        for b, gb in g.groupby(axis):
            if b == "nan" or len(gb) < 8 or gb["R_Multiple"].var() == 0:
                continue
            k = (gb["R_Multiple"].mean() / gb["R_Multiple"].var()) / base
            m = (len(gb) * k + N0 * 1.0) / (len(gb) + N0)
            tab[(s, b)] = float(np.clip(m, LO, HI))
    return tab

def evaluate(mult: pd.Series, d: pd.DataFrame) -> dict:
    pnl = d["PnL_flat_750k"] * mult; risk = d["Risk_flat_750k"] * mult
    base_pnl, base_risk = d["PnL_flat_750k"], d["Risk_flat_750k"]
    scale = base_risk.sum() / risk.sum()            # risk-matched
    daily = (pnl * scale).groupby(d["exit_d"]).sum().reindex(pd.bdate_range(d["exit_d"].min(), d["exit_d"].max())).fillna(0) / NAV
    bdaily = base_pnl.groupby(d["exit_d"]).sum().reindex(daily.index).fillna(0) / NAV
    def st(x):
        eq = x.cumsum(); dd = (eq - eq.cummax()).min()
        return dict(ann=x.mean() * 252 * 100, vol=x.std() * np.sqrt(252) * 100, sharpe=x.mean() / x.std() * np.sqrt(252), maxdd=dd * 100)
    a, b = st(daily), st(bdaily)
    return dict(pnl=float(pnl.sum()), pnl_riskmatched=float(pnl.sum() * scale), base_pnl=float(base_pnl.sum()), risk_ratio=float(risk.sum() / base_risk.sum()),
                pnl_per_risk=float(pnl.sum() / risk.sum()), base_pnl_per_risk=float(base_pnl.sum() / base_risk.sum()),
                sharpe=a["sharpe"], base_sharpe=b["sharpe"], maxdd=a["maxdd"], base_maxdd=b["maxdd"], ann=a["ann"], base_ann=b["ann"],
                share_scaled=float((mult != 1).mean()), mean_mult=float(mult.mean()))

wf_rows, wf_yearly = [], {}
for axis in AXES:
    for mode in ["per_strategy", "book"]:
        mults, years = [], []
        start = 2019 if axis == "dial" else 2010
        for Y in range(start, 2027):
            tr = T[(T["yr"] < Y) & (T[axis] != "nan")]
            te = T[(T["yr"] == Y)]
            if axis == "dial":
                tr = tr[tr["yr"] >= 2016]
            tab = fit_table(tr, axis, mode == "per_strategy")
            if mode == "per_strategy":
                m = [tab.get((s, b), 1.0) for s, b in zip(te["Strat2"], te[axis])]
            else:
                m = [tab.get(("ALL", b), 1.0) for b in te[axis]]
            mults.append(pd.Series(m, index=te.index)); years.append(te)
        mult = pd.concat(mults); d = pd.concat(years)
        ev = evaluate(mult, d)
        # per-year PnL per risk vs base
        yr_better = 0; yrs = 0; ylist = []
        for Y, g in d.groupby("yr"):
            mm = mult[g.index]
            ppr = (g["PnL_flat_750k"] * mm).sum() / (g["Risk_flat_750k"] * mm).sum(); bppr = g["PnL_flat_750k"].sum() / g["Risk_flat_750k"].sum()
            yrs += 1; yr_better += int(ppr > bppr); ylist.append(dict(year=int(Y), ppr=ppr, base_ppr=bppr))
        wf_rows.append(dict(axis=axis, mode=mode, years=yrs, years_better_ppr=yr_better, **ev))
        wf_yearly[f"{axis}|{mode}"] = ylist
WF = pd.DataFrame(wf_rows)
WF["d_ppr"] = WF["pnl_per_risk"] - WF["base_pnl_per_risk"]; WF["d_sharpe"] = WF["sharpe"] - WF["base_sharpe"]; WF["d_maxdd"] = WF["maxdd"] - WF["base_maxdd"]
WF["d_pnl_rm"] = WF["pnl_riskmatched"] - WF["base_pnl"]
print(WF[["axis", "mode", "years", "years_better_ppr", "base_pnl_per_risk", "pnl_per_risk", "d_ppr", "base_sharpe", "sharpe", "d_sharpe", "base_maxdd", "maxdd", "d_maxdd", "d_pnl_rm", "share_scaled", "mean_mult"]].round(3).to_string(index=False))
OUT["walk_forward"] = jsonable(WF.round(4).to_dict("records"))
OUT["walk_forward_yearly"] = jsonable(wf_yearly)

# the tables as fitted through 2025 (what would ship) for the axes that survive
print("\n=== per-strategy tables fit on all trades through 2025 (shrunk N0=50, clip [0.5,1.5]) ===")
final_tabs = {}
tr = T[(T["yr"] <= 2025)]
for axis in AXES:
    tab = fit_table(tr[tr[axis] != "nan"] if axis != "dial" else tr[(tr[axis] != "nan") & (tr.yr >= 2016)], axis, True)
    df = pd.Series(tab).unstack()
    final_tabs[axis] = jsonable({f"{s}": {b: round(v, 3) for b, v in row.dropna().items()} for s, row in df.iterrows()})
    if axis in ["cycle", "vix_lvl", "rv21", "spy_dd", "vix_ts", "credit21"]:
        print(f"-- {axis} --\n{df.round(2).to_string()}")
OUT["tables_through_2025"] = final_tabs

# ---------------------------------------------------------------- C. daily-series vol-direct overlays
print("\n=== C. daily overlays sizing UP with vol (lag-1), equal-vol Sharpe, 2003+ and 2016-07+ ===")
def metrics(r):
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(ann=r.mean() * 252 * 100, vol=r.std() * np.sqrt(252) * 100, sharpe=r.mean() / r.std() * np.sqrt(252), maxdd=dd * 100, worst=r.min() * 100)
D = pd.DataFrame({"ret": tot}).join(R[["vix_val", "rv21_val", "spy_dd_val", "vix_ts_val", "dial_val"]].shift(1)).dropna(subset=["ret", "vix_val"])
ov_rows = []
for win_lab, W in [("2003+", D), ("2016-07+", D[D.index >= "2016-07-20"])]:
    W = W.copy(); base = W["ret"]
    med_vix = W["vix_val"].median(); med_rv = W["rv21_val"].median()
    sims = {"baseline": pd.Series(1.0, index=W.index),
            "vix_direct_a0.5": ((W["vix_val"] / med_vix) ** 0.5).clip(0.5, 1.5),
            "vix_direct_a1": (W["vix_val"] / med_vix).clip(0.5, 1.5),
            "vix_inverse_a1": (med_vix / W["vix_val"]).clip(0.5, 1.5),
            "rv_direct_a0.5": ((W["rv21_val"] / med_rv) ** 0.5).clip(0.5, 1.5),
            "vix_step_lt15_0.75": np.where(W["vix_val"] < 15, 0.75, 1.0),
            "vix_step_lt15_0.75_gt20_1.25": np.where(W["vix_val"] < 15, 0.75, np.where(W["vix_val"] > 20, 1.25, 1.0)),
            "dd_step_3to10_1.25": np.where((W["spy_dd_val"] <= -0.03) & (W["spy_dd_val"] > -0.10), 1.25, 1.0),
            "backwardation_1.25": np.where(W["vix_ts_val"] > 1.0, 1.25, 1.0)}
    for k, m in sims.items():
        m = pd.Series(np.asarray(m, float), index=W.index).fillna(1.0)
        r = base * m; r_ev = r * (base.std() / r.std())
        raw, ev = metrics(r), metrics(r_ev)
        # LOYO: years where equal-vol-within-year Sharpe beats baseline
        yb, yn = 0, 0
        for y, g in W.groupby(W.index.year):
            if len(g) < 60:
                continue
            rr = g["ret"] * m[g.index]
            yn += 1; yb += int(rr.mean() / rr.std() > g["ret"].mean() / g["ret"].std())
        ov_rows.append(dict(window=win_lab, overlay=k, sharpe_raw=raw["sharpe"], sharpe_ev=ev["sharpe"], ann_ev=ev["ann"], maxdd_ev=ev["maxdd"], worst_ev=ev["worst"],
                            years_better=yb, years=yn, mean_mult=float(m.mean())))
OV = pd.DataFrame(ov_rows); print(OV.round(3).to_string(index=False))
OUT["daily_overlays"] = jsonable(OV.round(4).to_dict("records"))

json.dump(jsonable(OUT), open(HERE / "cycle_macro_02_validation.json", "w"), indent=1)
print("\nwrote", HERE / "cycle_macro_02_validation.json")
