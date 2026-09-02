"""cross_strategy_regime step 4: merge the three result files into the single
cross_strategy_regime_results.json the orchestrator reads, with a headline
block (primary hedge specs on both dial vintages, controls, instrument and
window sensitivity) pulled straight from the grid so the numbers cannot drift
from the scripts."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
corr = json.load(open(HERE / "cross_strategy_regime_results_1_corr.json"))
hedge = json.load(open(HERE / "cross_strategy_regime_results_2_hedge.json"))
refine = json.load(open(HERE / "cross_strategy_regime_results_3_refine.json"))
G = pd.DataFrame(hedge["grid"])
keep_cols = ["vintage", "target", "instrument", "window", "arming", "mult", "armed_days", "n_episodes", "hedge_total_usd", "hedge_bps_per_armed_day", "ep_mean_usd", "ep_pos_share",
             "t_clustered", "sharpe_unhedged", "sharpe_hedged", "ann_unhedged_pct", "ann_hedged_equal_vol_pct", "maxdd_unhedged_pct", "maxdd_hedged_pct", "armed_sharpe_unhedged",
             "armed_sharpe_hedged", "armed_sd_unhedged_bps", "armed_sd_hedged_bps", "armed_spy_ann_pct", "drift_component_usd", "calm_carry_bps_per_day", "calm_carry_usd_per_year",
             "false_alarm_episodes", "false_alarm_usd", "loyo_years_with_arming", "loyo_years_hedged_not_worse", "loyo_years_hedge_pnl_pos", "beta_mean_armed", "friction_usd"]


def rows(mask):
    return G[mask][keep_cols].round(3).to_dict("records")


headline = {
    "primary_book_SPY_63_by_arming_and_vintage": rows((G.target == "book") & (G.instrument == "SPY") & (G.window == 63) & (G.mult == 1.0)),
    "instrument_dial50_h45": rows((G.target == "book") & (G.window == 63) & (G.arming == "dial50_h45") & (G.mult == 1.0)),
    "instrument_dial65_h60": rows((G.target == "book") & (G.window == 63) & (G.arming == "dial65_h60") & (G.mult == 1.0)),
    "window_sensitivity": rows((G.target == "book") & (G.instrument == "SPY") & (G.arming.isin(["dial50_h45", "dial65_h60"])) & (G.mult == 1.0)),
    "subbook_vs_book": rows((G.instrument == "SPY") & (G.window == 63) & (G.arming.isin(["dial50_h45", "dial65_h60", "always"])) & (G.mult == 1.0)),
    "hedge_ratio_multiples": rows(G.mult != 1.0),
    "controls_non_dial": rows((G.target == "book") & (G.instrument == "SPY") & (G.window == 63) & (G.arming.isin(["vix25_h20", "vixpct80_h60", "spydd5_h3", "always"])) & (G.vintage == "live")),
}
pit_p1 = G[(G.vintage == "pit") & (G.target == "book") & (G.instrument == "SPY") & (G.window == 63) & (G.arming == "dial50_h45") & (G.mult == 1.0)].iloc[0]
pit_p1b = G[(G.vintage == "pit") & (G.target == "book") & (G.instrument == "SPY") & (G.window == 63) & (G.arming == "dial65_h60") & (G.mult == 1.0)].iloc[0]
ship_rule = {
    "rule": "P1 ship rule from dynamic_sizing_plan_2026-09-02: on the PIT vintage, mean armed-episode hedge PnL > 0 with clustered t >= 1.0, full-sample Sharpe hedged >= unhedged, >= 6 of 10 LOYO years not worse",
    "dial50_h45_PIT": dict(ep_mean_usd=float(pit_p1.ep_mean_usd), t_clustered=float(pit_p1.t_clustered), sharpe_unhedged=float(pit_p1.sharpe_unhedged), sharpe_hedged=float(pit_p1.sharpe_hedged),
                           loyo=f"{int(pit_p1.loyo_years_hedged_not_worse)} of {int(pit_p1.loyo_years_with_arming)} armed years not worse; {int(pit_p1.loyo_years_hedge_pnl_pos)} hedge-PnL positive",
                           bootstrap=refine["bootstrap"].get("pit|dial50_h45"), placebo=hedge["placebo"].get("pit|dial50_h45"),
                           verdict="PASSES all three legs (t 1.68, Sharpe 2.90 -> 3.04, 7/9 LOYO); drop-best-episode t 1.27; episode-bootstrap P(total<=0) 0.046; placebo rank 1.000"),
    "dial65_h60_PIT": dict(ep_mean_usd=float(pit_p1b.ep_mean_usd), t_clustered=float(pit_p1b.t_clustered), sharpe_unhedged=float(pit_p1b.sharpe_unhedged), sharpe_hedged=float(pit_p1b.sharpe_hedged),
                           loyo=f"{int(pit_p1b.loyo_years_hedged_not_worse)} of {int(pit_p1b.loyo_years_with_arming)} armed years not worse",
                           bootstrap=refine["bootstrap"].get("pit|dial65_h60"), placebo=hedge["placebo"].get("pit|dial65_h60"),
                           verdict="MARGINAL: t 0.99 (rule needs 1.0), Sharpe 2.90 -> 2.97, 6/7 LOYO; drop-best-episode t 0.37 (Feb-2020 carries it); bootstrap P(<=0) 0.134"),
}
out = dict(
    topic="cross_strategy_regime: correlation dynamics, market factor, dial-armed beta hedge",
    basis="dist/data/strategy_daily.json flat $750k (per-strategy daily MTM, tiers collapsed, exchange sessions only, ends 2026-08-07); dial = 10d MA of rd2_fragility 63d, lag-1; "
          "LIVE vintage = current-weights series (rows before 2026-07-02 are the recompute vintage); PIT vintage = vintage-lagged expanding-window weights rebuilt from scratch/pit_signals.pkl "
          "with the pit_reestimate.py method (2018-01-02 .. 2026-07-02; 2018 is scored by a single-signal vintage and is the noisiest year)",
    pit_dial_diagnostics=dict(corr_pit_vs_live=0.890, agreement_ge50_pct=88.2, agreement_ge65_pct=89.7, days_ge50_pit_pct=22.9, days_ge50_live_pct=15.4, days_ge65_pit_pct=14.0, days_ge65_live_pct=6.2,
                              vintages_file="cross_strategy_regime_pit_vintages.json"),
    headline_hedge=headline, ship_rule_check=ship_rule, hedge_placebo=hedge["placebo"], hedge_anatomy=hedge["anatomy"], beta_quality=hedge["beta_quality"],
    episodes=refine["episodes"], refined_arming_in_sample=refine["refined"], episode_bootstrap=refine["bootstrap"], signal_flow=refine["signal_flow"],
    hedge_contribution_by_strategy=refine["contrib"], instrument_anatomy=refine["instrument_anatomy"],
    correlation=dict(regime=corr["regime"], rolling_by_year=corr["rolling_by_year"], lowest_eff_n_windows=corr["lowest_eff_n_windows"], rolling_vs_state=corr["rolling_vs_state"],
                     pairs=corr["pairs"], pairs_one_bet=corr["pairs_one_bet"], betas=corr["betas"], book_factor_r2=corr["book_factor_r2"], subbook_loadings=corr["subbook_loadings"],
                     mom_factor=corr["mom_factor"]),
    files=dict(scripts=["cross_strategy_regime_0_pit_dial.py", "cross_strategy_regime_1_corr.py", "cross_strategy_regime_2_hedge.py", "cross_strategy_regime_3_refine.py", "cross_strategy_regime_4_merge.py"],
               results=["cross_strategy_regime_results_1_corr.json", "cross_strategy_regime_results_2_hedge.json", "cross_strategy_regime_results_3_refine.json", "cross_strategy_regime_hedge_grid.csv",
                        "cross_strategy_regime_pit_dial.parquet", "cross_strategy_regime_pit_vintages.json", "cross_strategy_regime_mom_factor.parquet"]),
)
json.dump(out, open(HERE / "cross_strategy_regime_results.json", "w"), indent=1)
print("wrote cross_strategy_regime_results.json;", len(G), "grid rows")
