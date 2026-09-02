"""Aggregate the flow-conditional study outputs into flow_conditional_results.json (the deliverable)."""
from __future__ import annotations
import json
import pandas as pd
from flow_conditional_lib import OUT_DIR, FAMILIES, FAMILY

J = lambda n: json.load(open(OUT_DIR / n))
edge = J("flow_conditional_edge_candidates.json"); edge_f = J("flow_conditional_edge_fills.json")
sleeve = J("flow_conditional_sleeve_candidates.json"); wf = J("flow_conditional_walkforward_candidates.json")
ctl = J("flow_conditional_controls.json"); fx = J("flow_conditional_fixed_rules.json")

B = pd.DataFrame(edge["buckets"]); Bf = pd.DataFrame(edge_f["buckets"])
fam = B[B.level == "family"]
keep = ["s1", "s5", "s21", "f1", "f5", "f21", "b5", "nstrat1", "open_s", "f21_rel"]
fam_tbl = {f: {v: {k: r[k] for k in ["N", "rho", "lo_N", "lo_avgR", "hi_N", "hi_avgR", "lo_rpr", "hi_rpr", "hi_cap", "diff", "t_cl", "p_cl", "hi_ep", "lo_ep", "thr_lo", "thr_hi"]}
               for v, r in ((r["var"], r) for _, r in fam[fam.name == f].iterrows()) if v in keep} for f in FAMILIES}
Y = pd.DataFrame(edge["year_stability"])
stab = {f: {r["var"]: dict(years=r["years"], pos_share=r["pos_share"], mean_rho=r["mean_rho"]) for _, r in Y[Y.family == f].iterrows() if r["var"] in keep} for f in FAMILIES}
S = pd.DataFrame(sleeve["forecast"])
sl = {s: {r["var"]: {k: r[k] for k in ["N", "rho_vol", "rho_mean", "rho_sharpe", "partial_flow_on_vol_given_trailvol", "lo_flow_fwd_vol", "hi_flow_fwd_vol", "lo_flow_fwd_mean", "hi_flow_fwd_mean", "lo_flow_fwd_sharpe", "hi_flow_fwd_sharpe"]}
          for _, r in S[S.sleeve == s].iterrows() if r["var"] in ("flow5", "flow21")} for s in FAMILIES + ["book"]}
W = pd.DataFrame(wf["walkforward"])
wfk = ["var", "mode", "N", "base_pnl", "eq_pnl", "eq_risk_ratio", "base_sharpe", "eq_sharpe", "base_maxdd", "eq_maxdd", "base_pnl_per_risk", "eq_pnl_per_risk", "years_better", "years", "worst_year_rel", "nocap_pnl"]
wft = {f: [dict((k, r[k]) for k in wfk) for _, r in W[(W.family == f) & (W["var"].isin(["s1", "s5", "f5", "f21", "f21_rel", "nstrat1", "b5"]))].iterrows()] for f in FAMILIES}
C = pd.DataFrame(ctl["rank_regression"]); Cw = pd.DataFrame(ctl["within_control"])
FX = pd.DataFrame(fx["fixed_rules"])

out = dict(
    topic="signal-flow-conditional sizing",
    data=dict(ledger="data/backtest_trades_full.parquet (vintage in schema metadata; trades collapsed over OVS tranches -> 3,483 trades 2003-2026, 3,360 from 2005)",
              flow_source="engine raw candidate signal-dates re-generated this session via generate_candidates_fast (scratch/ultracode_sizing_2026-09-02/flow_candidates.parquet, 24,669 candidates 2003-01..2026-09, 15 strategies x liquid/overflow) -- STAGED signals incl. unfilled; a fills-based flow was run as a cross-check (edge_fills.json)",
              daily_mtm="dist/data/strategy_daily.json (per-strategy daily MTM, flat $750k; Aug-2026 build) for the sleeve-vol tests; per-trade MTM rebuilt from master_prices closes (reconciled to booked PnL) for the replays",
              dial="data/rd2_fragility.parquet 10d-MA of 63d, lag-1; rows before 2026-07-02 are the recompute vintage (current weights, not PIT)",
              families=FAMILY),
    conventions="flow = trailing candidate counts INCLUSIVE of the signal day (same-day count is known at the close, like same_day_signal_derate); s=strategy, f=family, b=book, nstrat1 = distinct strategies firing that day, open_s = open filled legs of the strategy entered before the signal day; terciles within family; cluster t = episode-cluster bootstrap (episode = signal dates <= 5 td apart); walk-forward = expanding window, annual re-fit, test 2010-2026, mults = 1 + 0.5*(rpr_bucket/rpr_all - 1) clipped [0.5,1.5], eq-risk = rule scaled so training risk deployed matches baseline, per-strategy 250 bps cap RE-APPLIED after the rule using the row's recovered cap scale (placed total = cap/scale on bound days)",
    family_flow_cells_candidates=fam_tbl,
    per_year_sign_stability=stab,
    fills_vs_candidates_family_t={f: {v: dict(cand=float(fam[(fam.name == f) & (fam["var"] == v)].t_cl.iloc[0]), fills=float(Bf[(Bf.level == "family") & (Bf.name == f) & (Bf["var"] == v)].t_cl.iloc[0])) for v in ["s1", "s5", "f5", "f21", "b5", "open_s"]} for f in FAMILIES},
    same_day_cells=edge["same_day_cells"], nstrat_cells=edge["nstrat_cells"], era_split=edge["era_split"],
    sleeve_forecast=sl, sleeve_per_year=sleeve["per_year"], sleeve_kelly_by_flow=sleeve["kelly_by_flow"],
    controls=dict(rank_regression=C.to_dict("records"), within_control=Cw.to_dict("records"), flow_vs_controls=ctl["flow_vs_controls"], olv_rung_vs_flow=ctl["olv_rung_vs_flow"]),
    walkforward=wft, walkforward_last_fit=wf["walkforward_last_fit"], walkforward_years=wf["walkforward_years"], book_walkforward_f5=wf["book_f5"],
    cap_by_flow=wf["cap_by_flow"], cap_by_strategy=wf["cap_by_strategy"], cap_relief_unfitted=wf["cap_relief"],
    fixed_rules=fx["fixed_rules"], fixed_rules_book=fx["book"], cap_absorbs_share_of_upsize=fx.get("cap_absorbs_share_of_upsize"),
)
json.dump(out, open(OUT_DIR / "flow_conditional_results.json", "w"), indent=1, default=float)
print("wrote flow_conditional_results.json")
