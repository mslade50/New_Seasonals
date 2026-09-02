"""Merge parts 1-4 into unconstrained_growth_results.json with the
recommendation block.  m = multiple of current sizing; GRM = 1.5 m."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
G = json.load(open(HERE / "unconstrained_growth_01_growth.json"))
C = json.load(open(HERE / "unconstrained_growth_01b_capaware.json"))
M = json.load(open(HERE / "unconstrained_growth_02_margin.json"))
MR = json.load(open(HERE / "unconstrained_growth_02b_margin_refine.json"))
L = json.load(open(HERE / "unconstrained_growth_03_liquidity.json"))
K = json.load(open(HERE / "unconstrained_growth_04_compounding.json"))
GRM = 1.5

def bt(win, h, tag, m):
    return G["bootstrap"][f"{win}|h{h:g}"][tag][f"{m:g}"]

growth_table = {}
for win in ["2003+", "2016+", "2021+"]:
    growth_table[win] = {"base": G["analytic"][win]["base"], "by_haircut": {}}
    for h in ["0", "0.25", "0.5"]:
        a = G["analytic"][win]["haircut"][h]
        growth_table[win]["by_haircut"][h] = {k: a[k] for k in ["m_star", "grm_star", "g_star", "g_at_current", "m_quarter", "g_quarter", "m_half", "g_half", "m_3q", "g_3q", "quad_m_star", "first_ruin_m", "curve"]}
fat_tail = {
    "gpd_tail_no_haircut": {w: {k: G["parametric"][w]["gpd"][k] for k in ["xi_lower", "xi_upper", "m_star", "g_star", "g_at_current"]} for w in ["2003+", "2016+"]},
    "gpd_tail_with_haircut": K["gpd_haircut"],
    "student_t_mixture": {w: {k: C["t_mixture"][w][k] for k in ["p_flat", "df", "scale", "m_star_trunc", "g_star_trunc"]} | {"annual_p_single_day_ruin_at_m": {m: C["t_mixture"][w]["curve"][m]["p_ruin_year"] for m in ["1", "1.5", "2", "3", "5", "8", "12"]}} for w in ["2003+", "2016+"]},
    "bootstrap_growth_max": {key: max(((float(m), v["growth_mean"]) for m, v in G["bootstrap"][key]["3y"].items() if v["growth_mean"] is not None), key=lambda x: x[1]) for key in G["bootstrap"]},
}
dd_table = {}
for win in ["2016+", "2003+"]:
    for h in [0, 0.5]:
        for tag in ["1y", "3y"]:
            dd_table[f"{win}|h{h:g}|{tag}"] = {m: {k: bt(win, h, tag, float(m))[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30", "p_dd_gt_40", "p_dd_gt_50", "flat_dd_median", "flat_dd_p95", "p_flat_dd_gt_20", "growth_mean", "growth_p05", "p_end_below_start", "recover_days_median", "recover_days_mean", "recover_days_p95", "p_unrecovered_at_horizon", "longest_underwater_median", "longest_underwater_p95", "p_ruin"]}
                                              for m in ["1", "1.25", "1.5", "2", "2.5", "3", "4", "5", "8", "12"]}
cap = {w: {m: {k: C["cap_aware"][w][m][k] for k in ["g_linear", "g_cap_fixed", "g_cap_scaled", "eff_mult_fixed_cap", "ann_vol_fixed", "maxdd_fixed"]} for m in ["1", "1.5", "2", "3", "4", "5", "8", "12", "20"]} for w in ["2016+", "2003+"]}
cap["signal_day_risk_bps_at_1x"] = C["signal_day_risk_bps_at_1x"]
margin = {
    "gross_notional_pct_nav_at_1x": M["gross_notional_pct_nav_at_1x"], "top_notional_days": M["top_notional_days"][:5],
    "top_decile_notional_owner_share_2016": M["top_decile_notional_owner_share_2016"],
    "strategy_open_notional_p95_pct_nav": M["strategy_open_notional_p95_pct_nav"],
    "single_ticker_concentration_at_1x": MR["single_ticker_concentration_at_1x"],
    "feasibility_by_scenario": MR["feasibility"], "tail_composition_top1pct_days": MR["tail_composition_top1pct_days"],
    "ruin_bootstrap": {w: {tag: {sc: {m: M["ruin_bootstrap"][w][tag][sc][m] for m in ["1", "1.25", "1.5", "2", "2.5", "3", "4"]} for sc in ["pm_15", "pm_15_conc30", "pm_25_conc30", "pm_30_stress"]} for tag in ["1y", "3y"]} for w in ["2016+", "2003+"]},
    "gross_tail_pnl_share": M["gross_tail_pnl_share"], "hedge_note": MR["hedge_note"],
}
liq = {"k_impact": L["k_impact"], "impact_by_m": {m: {k: L["impact_by_m"][m][k] for k in ["drag_pct_nav_yr", "gross_pct_nav_yr", "drag_share_of_gross", "share_trades_part_over_5", "share_trades_part_over_25"]} for m in ["1", "1.5", "2", "3", "5", "8", "12", "20"]},
       "m_zero_edge_by_strategy": L["m_zero_edge_by_strategy"], "book_m_zero_edge": L["book_m_zero_edge"], "participation_at_1x_top": L["participation_at_1x"][:8], "sensitivity": L["sensitivity"], "adv_cap_10pct": {m: L["adv_cap_10pct"][m]["effective_mult"] for m in ["2", "3", "5", "8", "12", "20"]}}
comp = {k: v for k, v in K["policies"].items()}

# ------------------------------------------------------------ recommendation logic (numbers cited from the tables above)
rec = {
    "growth_optimum_unconstrained": "m* 15.5 (2003+) / 19.25 (2016+) on the raw empirical curve = GRM 23-29; 9-11.75 (GRM 13.5-17.6) at a 50% mean haircut; 6.25-9 (GRM 9.4-13.5) once the tails are GPD-extrapolated, where the optimum is set by the single-day ruin boundary (one simulated -11% to -15% day in 2M draws ruins at m 6.5-9.25) rather than by curvature, and is haircut-insensitive; the flat-day/active-day Student-t mixture gives 11-14 with an annual single-day-ruin probability of 0.3-0.7% at m 1-1.5 rising to 6% at m 5. Every version sits far above anything the account can carry.",
    "binding_boundary": "Portfolio margin, not appetite and not liquidity. At IBKR-style PM 15% (3x ETFs 45%) with no concentration add-on the historical max requirement day (2016-06-14, gross 418% NAV, all long, WCDS+Monday Dip+MonFri cluster) reaches 100% of the flat $750k at m=1.60 (GRM 2.39); the p99 day at m=2.46 (GRM 3.69); on the live ~$632k primary NLV the max day binds at m=1.34 (GRM 2.01). With a 30% concentration add-on on non-broad tickers above 25% NAV the max day already needs 89% of NAV at m=1 (binds at m=1.12, GRM 1.68). Under Reg T the book is already infeasible on its tail days (m 0.48).",
    "secondary_boundary": "The per-strategy 250 bps/day cap: realised multiple saturates at ~4.6x however large m is (eff 1.93 at m=2, 2.68 at m=3, 3.69 at m=5), and it clips the cluster days that carry the edge, so cap-fixed growth is 60.5% vs 63.7% linear at m=2 and 82% vs 93% at m=3 (2016+).",
    "liquidity": "Not binding: sqrt-impact drag at k=1 is 2.8% NAV/yr (8% of gross) at m=1, 8.0%/yr (11.5%) at m=2, 14.7%/yr (14%) at m=3; the book's edge halves only at m~58. Only the overflow-tier ATR Extended Gap Up (p90 participation 2.7% ADV, edge halves at m=12) and St OS Sznl / LT Trend overflow (m 16-19) are liquidity-sensitive.",
    "recommended_grm": 2.25,
    "recommended_m": 1.5,
    "recommended_grm_rationale": "GRM 2.25 (m=1.5) is the largest step that stays inside the plain-PM boundary on the historical max day (m 1.60) with a small cushion, and inside the p99 day on the live NLV (m 2.46 x 632/750 = 2.07). It is unconstrained by drawdown: growth theory would go 5-10x further, the account cannot. A step to GRM 3.0 (m=2) is feasible ONLY if the gross-notional tail is controlled first (SPY/QQQ/DIA legs of the dip-buy cluster on index futures at ~5% margin, or a margin-feasibility guard in order_staging), because at m=2 the joint bootstrap puts P(requirement > equity within 3y) at 14% (2016+) to 23% (2003+) on plain PM and 33-48% with a 15% cushion.",
    "dd_accepted_at_recommendation": {
        "2016+_no_haircut": {"1y": {k: bt("2016+", 0, "1y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}, "3y": {k: bt("2016+", 0, "3y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}},
        "2016+_50pct_haircut": {"1y": {k: bt("2016+", 0.5, "1y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}, "3y": {k: bt("2016+", 0.5, "3y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}},
        "2003+_no_haircut": {"1y": {k: bt("2003+", 0, "1y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}, "3y": {k: bt("2003+", 0, "3y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}},
        "2003+_50pct_haircut": {"1y": {k: bt("2003+", 0.5, "1y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}, "3y": {k: bt("2003+", 0.5, "3y", 1.5)[k] for k in ["maxdd_median", "maxdd_p95", "p_dd_gt_10", "p_dd_gt_20", "p_dd_gt_30"]}},
        "actual_path_2003_2026_compounded": G["historical"]["2003+"]["1.5"], "actual_path_2016_2026_compounded": G["historical"]["2016+"]["1.5"],
    },
    "growth_at_recommendation": {"2016+": {h: G["analytic"]["2016+"]["haircut"][h]["curve"]["1.5"] for h in ["0", "0.25", "0.5"]}, "2003+": {h: G["analytic"]["2003+"]["haircut"][h]["curve"]["1.5"] for h in ["0", "0.25", "0.5"]}},
    "infeasibility_multiples": {
        "margin_pm15_max_day_flat750k": MR["feasibility"]["pm_15"]["m_at_max"], "margin_pm15_p99_day_flat750k": MR["feasibility"]["pm_15"]["m_at_p99"],
        "margin_pm15_max_day_live_nlv": MR["feasibility"]["pm_15"]["m_at_max_live_nlv"], "margin_pm15_conc30_nonbroad_max_day": MR["feasibility"]["pm_15_conc30_nonbroad"]["m_at_max"],
        "margin_pm30_stress_max_day": MR["feasibility"]["pm_30_stress_all"]["m_at_max"], "regT_max_day": MR["feasibility"]["regT_50"]["m_at_max"],
        "single_day_ruin_2016": G["analytic"]["2016+"]["haircut"]["0"]["first_ruin_m"], "single_day_ruin_2003": G["analytic"]["2003+"]["haircut"]["0"]["first_ruin_m"],
        "liquidity_half_edge_book": L["book_m_zero_edge"]["m_half_edge"], "cap250_effective_multiple_ceiling": C["cap_aware"]["2016+"]["20"]["eff_mult_fixed_cap"],
    },
    "compounding": "Switching the base to live NAV is growth-positive in expectation (10y median terminal 2016+ no haircut: comp far above flat at m=1) and carries no extra %-of-peak drawdown, but two facts bind: (i) the live primary NLV (~$632k) is BELOW the $750k constant, so the book is already running ~1.19x the risk a NAV basis would give (effective GRM ~1.78 in live terms) and a switch today CUTS risk 16%; (ii) sequencing: the flat basis de-levers automatically as NAV grows (effective m at the median 10y flat path end ~0.2-0.3), which is why flat maxDD-in-dollars stays bounded while compounded dollar drawdowns scale with NAV. A half-compounding base (0.5 x 750k + 0.5 x NAV, quarterly) captures most of the median gain with a dollar-DD tail between the two; see compounding_policies.",
}

OUT = {"topic": "unconstrained growth optimum and its real boundaries", "asof": "2026-09-02", "basis": "flat $750k; ledger gha:33608560596 (2026-09-02) for notional/liquidity; strategy_daily.json (2026-08-07 vintage) for the daily series; m = multiple of current sizing, GRM = 1.5 m",
       "growth_curves": growth_table, "fat_tail": fat_tail, "cap_aware": cap, "drawdown_distribution": dd_table, "fractional_kelly_theory": G["fractional_kelly_theory"],
       "historical_path": G["historical"], "margin": margin, "liquidity": liq, "compounding_policies": comp, "recommendation": rec}
json.dump(OUT, open(HERE / "unconstrained_growth_results.json", "w"), indent=1, default=float)
print("wrote", HERE / "unconstrained_growth_results.json")
print(json.dumps(rec["dd_accepted_at_recommendation"], indent=1)[:3000])
print(json.dumps(rec["infeasibility_multiples"], indent=1))
print(json.dumps(fat_tail["gpd_tail_with_haircut"], indent=1))
print(json.dumps(fat_tail["bootstrap_growth_max"], indent=1))
