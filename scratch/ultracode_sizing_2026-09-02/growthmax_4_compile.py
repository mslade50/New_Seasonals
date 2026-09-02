"""Growth-maximizer lens, part 4: merge growthmax_1/2/3 JSONs into
growthmax_results.json with a headline block (the numbers the plan cites).
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
m = json.load(open(HERE / "growthmax_1_margin_tiered.json"))
g = json.load(open(HERE / "growthmax_2_growth_dd_acf.json"))
a = json.load(open(HERE / "growthmax_3_alloc_keep.json"))

sc = m["scenarios"]
head = {
    "lens": "growth-maximizer: maximize E[log growth] subject to margin feasibility and liquidity only",
    "binding_boundary_tims_tiered": {
        "scenario": "tims_pm_cheapshort (broad idx 8%, small-cap idx 10%, single/sector 15%, 3x 45%, cheap-short per-share floors)",
        "max_day": sc["tims_pm_cheapshort"]["max_date"], "req_pct_nav_max": sc["tims_pm_cheapshort"]["max"], "req_pct_nav_p99": sc["tims_pm_cheapshort"]["p99"],
        "m_max_750k": sc["tims_pm_cheapshort"]["m_max_750"], "grm_max_750k": sc["tims_pm_cheapshort"]["grm_max_750"],
        "m_max_live632k": sc["tims_pm_cheapshort"]["m_max_live"], "grm_max_live632k": sc["tims_pm_cheapshort"]["grm_max_live"],
        "m_p99_live632k": sc["tims_pm_cheapshort"]["m_p99_live"], "grm_p99_live632k": sc["tims_pm_cheapshort"]["grm_p99_live"],
        "m_max_750k_15pct_cushion": sc["tims_pm_cheapshort"]["m_max_750_cushion15"],
        "flat15_reference_m_max_750k": sc["flat15_pm"]["m_max_750"],
    },
    "if_ibkr_applies_rules_based_3x_margin": {"scenario": "tims_lev_rules (3x at 75% long / 90% short)", "max_day": sc["tims_lev_rules"]["max_date"],
                                              "req_pct_nav_max_at_m1": sc["tims_lev_rules"]["max"], "m_max_750k": sc["tims_lev_rules"]["m_max_750"], "m_p99_live": sc["tims_lev_rules"]["m_p99_live"]},
    "concentration_addon_scenarios": {k: dict(max=sc[k]["max"], max_date=sc[k]["max_date"], m_max_750k=sc[k]["m_max_750"]) for k in ["tims_pm_cheapshort_conc25", "tims_pm_cheapshort_conc50"]},
    "futures_rerouting_of_broad_index_legs": {"m_max_750k": sc["tims_pm_futures_broad"]["m_max_750"], "m_p99_live": sc["tims_pm_futures_broad"]["m_p99_live"],
                                             "note": "does not move the max day (2023-02-03 is a 3x-short cluster); lifts the p99 day only"},
    "equity_shock_30pct": m["equity_shock_30pct_loss_pct_nav"],
    "guard_days_per_year_req_over_70pct_nav": m["guard_days_per_year_over_70pct"],
    "guard_days_per_year_req_over_100pct_nav": m["guard_days_per_year_over_100pct"],
    "book_margin_per_dollar_risk_tims": m["book_margin_per_risk_tims"],
    "growth_at_40pct_haircut": {w: {k: v for k, v in g["growth"][w]["hc0.4"].items()} for w in g["growth"]},
    "drawdown_1y_40pct_haircut": {w: {mm: g["drawdown"][w]["1y_hc0.4"][mm] for mm in ["1", "1.5", "1.75", "2", "2.5", "3"]} for w in g["drawdown"]},
    "drawdown_3y_40pct_haircut": {w: {mm: g["drawdown"][w]["3y_hc0.4"][mm] for mm in ["1.5", "2", "2.5", "3"]} for w in g["drawdown"]},
    "actual_path": g["actual_path"],
    "acf": {w: {k: v for k, v in g["acf"][w].items() if k != "lags"} | {"lag1_5": g["acf"][w]["lags"][:5]} for w in g["acf"]},
    "kelly_fraction_40pct_haircut": {w: {k: v for k, v in g["kelly_fraction"][w].items() if k.startswith("hc0.4")} for w in g["kelly_fraction"]},
    "allocation_walk_forward": a["walk_forward"]["summary"],
    "ship_multipliers": a["ship_multipliers"],
    "margin_shadow_price": a["margin_shadow_price"], "equal_weight_margin_pct_nav": a["equal_weight_margin_pct_nav"],
}
out = {"headline": head, "margin": m, "growth_dd_acf": g, "allocation": a}
json.dump(out, open(HERE / "growthmax_results.json", "w"), indent=1, default=float)
print(json.dumps(head["equity_shock_30pct"], indent=1))
print("guard >70%:", head["guard_days_per_year_req_over_70pct_nav"])
print("acf:", json.dumps(head["acf"], indent=1)[:1500])
print("wrote growthmax_results.json")
