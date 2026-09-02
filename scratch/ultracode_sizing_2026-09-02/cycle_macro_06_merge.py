"""Merge parts 01-05 into cycle_macro_results.json with a computed summary
block (headline numbers pulled from the part files, plus two small extra
cells: book-ex-OVS midterm year-cluster t, and the calm-tape share of book
open risk)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_macro_lib import HERE, attach_trade_regimes, build_regimes, jsonable, load_ledger, welch_t

parts = {k: json.load(open(HERE / f"cycle_macro_{k}.json")) for k in ["01_tables", "02_validation", "03_overlap", "04_rules", "05_single_wf"]}
led = load_ledger(); R = build_regimes(); T = attach_trade_regimes(led, R)

# book ex-OVS, midterm, year cluster
ex = T[~T["Strat2"].str.startswith("OVS")]
y = ex.groupby("yr").agg(avgR=("R_Multiple", "mean"), N=("R_Multiple", "size"), cyc=("cycle", "first")); y = y[y.N >= 8]
mid, oth = y[y.cyc == "midterm"], y[y.cyc != "midterm"]
exovs = dict(mid_years=int(len(mid)), oth_years=int(len(oth)), mid_avgR=float(mid.avgR.mean()), oth_avgR=float(oth.avgR.mean()), t_year=welch_t(mid.avgR, oth.avgR),
             mid_years_below=int((mid.avgR < oth.avgR.mean()).sum()), mid_by_year={int(k): round(float(v), 3) for k, v in mid.avgR.items()})
print("book ex-OVS midterm year-cluster:", exovs)

# calm-tape share of open risk-days
tt = pd.DataFrame(parts["05_single_wf"]["open_risk_by_regime"]["rv21"]).set_index("rv21")
print(tt)

tr = pd.DataFrame(parts["01_tables"]["trade_table"])
book = tr[tr.strategy == "ALL"].set_index(["regime", "bucket"])
def cell(reg, b, cols=("N", "avgR", "avgR_rest", "sdR", "sd_ratio", "diff", "t_cluster", "yr_pos", "yr_neg", "kelly_rel")):
    r = book.loc[(reg, b)]
    return {c: (None if pd.isna(r[c]) else float(r[c])) for c in cols}
bd = pd.DataFrame(parts["01_tables"]["book_daily_by_regime"]).set_index(["regime", "bucket"])
def dcell(reg, b):
    r = bd.loc[(reg, b)]
    return {c: float(r[c]) for c in ["days", "mean_bps", "sd_bps", "sharpe", "sd_ratio", "mean_ratio"]}

summary = {
    "book_cycle": {"trade": {k: cell("cycle", k) for k in ["election", "midterm", "post_election", "pre_election"]},
                   "daily": {k: dcell("cycle", k) for k in ["election", "midterm", "post_election", "pre_election"]},
                   "year_cluster": parts["02_validation"]["cycle_year_cluster"][0], "ex_ovs_year_cluster": exovs,
                   "spy_by_cycle_year_2001plus": parts["02_validation"]["spy_by_cycle_year"], "spy_intra_year_dd": parts["02_validation"]["spy_intra_year_dd_by_cycle"]},
    "book_vol_axes": {"trade": {f"{a}|{b}": cell(a, b) for a, b in [("vix_lvl", "<15"), ("vix_lvl", "20-30"), ("vix_lvl", "30+"), ("rv21", "<12"), ("rv21", "20-30"), ("spy_dd", "<3"), ("spy_dd", "3-10"), ("spy_dd", "bear>20"), ("vix_ts", "backwardation"), ("credit21", "widening"), ("tnx_chg63", "falling"), ("pc_fear", "fear_on"), ("pc_fear", "complacent"), ("mom12_1", "neg"), ("spy_200", "below")]},
                      "daily": {f"{a}|{b}": dcell(a, b) for a, b in [("vix_lvl", "<15"), ("vix_lvl", "20-30"), ("vix_lvl", "30+"), ("rv21", "<12"), ("rv21", "20-30"), ("spy_dd", "<3"), ("spy_dd", "3-10"), ("spy_dd", "bear>20"), ("vix_ts", "backwardation"), ("credit21", "widening"), ("pc_fear", "fear_on"), ("mom12_1", "neg"), ("spy_200", "below")]}},
    "open_risk_by_regime": parts["05_single_wf"]["open_risk_by_regime"],
    "regime_vs_dial": parts["03_overlap"]["regime_vs_dial_days"],
    "pit_dial_cells": parts["03_overlap"]["pit_cells"],
    "pit_meta": parts["03_overlap"].get("pit_meta"),
    "walk_forward_axes": parts["02_validation"]["walk_forward"],
    "family_walk_forward": parts["04_rules"]["family_walk_forward"],
    "single_strategy_walk_forward": [{k: v for k, v in r.items() if k != "yearly"} for r in parts["05_single_wf"]["single_wf"]],
    "fixed_rules": parts["04_rules"]["fixed_rules"],
    "daily_overlays": parts["02_validation"]["daily_overlays"],
    "lt_trend_pit50": parts["04_rules"]["lt_trend_pit50"],
    "ovs_cycle": parts["03_overlap"]["ovs_cycle"],
}
out = {"meta": parts["01_tables"]["meta"], "summary": summary, "parts": parts}
json.dump(jsonable(out), open(HERE / "cycle_macro_results.json", "w"), indent=1)
print("wrote", HERE / "cycle_macro_results.json")
