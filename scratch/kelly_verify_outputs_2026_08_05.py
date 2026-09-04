"""Integrity checks for the Kelly research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scratch"
STAMP = "2026-08-05"
OUT = S / f"kelly_verification_{STAMP}.json"


def close(a, b, tol=1e-7):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def main():
    replay = json.loads((S / f"kelly_current_replay_summary_{STAMP}.json").read_text())
    analytic = json.loads((S / f"kelly_phase3_analytic_summary_{STAMP}.json").read_text())
    engine = json.loads((S / f"kelly_engine_replay_results_{STAMP}.json").read_text())
    trades = pd.read_parquet(S / f"kelly_current_trades_{STAMP}.parquet")
    signals = pd.read_parquet(S / f"kelly_current_signals_{STAMP}.parquet")
    daily = pd.read_parquet(S / f"kelly_current_daily_components_{STAMP}.parquet")
    alloc = pd.read_csv(S / f"kelly_allocations_{STAMP}.csv")
    scenarios = pd.read_csv(S / f"kelly_scenario_multipliers_{STAMP}.csv", index_col=0)
    curve = pd.read_csv(S / f"kelly_growth_drawdown_curve_{STAMP}.csv")
    comp = pd.read_csv(S / f"kelly_engine_replay_components_{STAMP}.csv")

    checks = {}
    daily_total = float(daily.drop(columns="date").to_numpy().sum())
    checks["daily_equals_replay_total"] = close(daily_total, replay["metrics"]["total_pnl"])
    checks["signal_pnl_equals_replay_total"] = close(signals["PnL"].sum(), replay["metrics"]["total_pnl"])
    checks["trade_rows_match"] = len(trades) == replay["engine_row_count"] == 4778
    checks["collapsed_signals_match"] = len(signals) == replay["collapsed_signal_count"] == 3585
    checks["components_match"] = set(alloc["Component"]) == set(replay["components"])
    checks["fifteen_strategies_current"] = replay["liquid_strategy_count"] == 15
    checks["baseline_replay_exact"] = close(
        engine["baseline"]["total_pnl"], replay["metrics"]["total_pnl"]
    )

    budget = alloc.set_index("Component")["Annual_Risk_Budget"]
    primary = scenarios.loc["Primary_LOYO_FullCov_EqualRisk"].reindex(budget.index)
    checks["primary_budget_equality"] = close(float((primary * budget).sum()), float(budget.sum()), 1e-6)
    checks["primary_nonnegative"] = bool((primary >= -1e-10).all())
    for pilot in analytic["pilots_frozen"]:
        checks[f"pilot_fixed_{pilot}"] = close(primary[pilot], 1.0)

    ovs_budget = budget[["Overbot Vol Spike P1", "Overbot Vol Spike P2"]]
    budget_2018 = alloc.set_index("Component")["Annual_Risk_Budget_2018"]
    for name in [x for x in scenarios.index if x.startswith("OVS_Internal") and "LiquidOnly" not in x]:
        row = scenarios.loc[name]
        use_budget = budget_2018[ovs_budget.index] if "2018" in name else ovs_budget
        checks[f"ovs_budget_{name}"] = close(
            float((row[use_budget.index] * use_budget).sum()), float(use_budget.sum()), 1e-6
        )
    checks["ovs_split_sign_reversal"] = bool(
        scenarios.loc["OVS_Internal_LOYO_FixedRisk", "Overbot Vol Spike P1"] > 1
        and scenarios.loc["OVS_Internal_LOYO_FixedRisk", "Overbot Vol Spike P2"] < 1e-8
        and scenarios.loc["OVS_Internal_LiquidOnly_FixedRisk", "Overbot Vol Spike P1"] < 1e-8
        and scenarios.loc["OVS_Internal_LiquidOnly_FixedRisk", "Overbot Vol Spike P2"] > 1
    )

    p_current = analytic["bootstrap"]["current_book"]["p_dd_gt_20pct"]
    p_quarter = float(curve.loc[np.isclose(curve["c"], 0.25), "P_DD_Worse_20pct"].iloc[0])
    p_half = float(curve.loc[np.isclose(curve["c"], 0.50), "P_DD_Worse_20pct"].iloc[0])
    checks["current_passes_drawdown_gate"] = p_current < 0.05
    checks["quarter_fails_drawdown_gate"] = p_quarter >= 0.05
    checks["half_fails_drawdown_gate"] = p_half >= 0.05
    checks["bisection_on_gate"] = abs(
        analytic["bootstrap"]["probability_at_bisection_c"] - 0.05
    ) <= 0.001

    checks["engine_delta_components_reconcile"] = close(
        comp["Delta_PnL"].sum(), engine["delta"]["total_pnl"]
    )
    checks["engine_proposal_near_equal_risk"] = abs(
        engine["delta"]["annual_filled_risk_fraction"]
    ) < 0.001
    checks["engine_proposal_same_maxdd"] = close(
        engine["baseline"]["max_dd"], engine["proposal_metrics"]["max_dd"]
    )

    failed = [k for k, v in checks.items() if not v]
    report = {
        "study_date": STAMP,
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": failed,
        "headline_values": {
            "current_bootstrap_p_dd20": p_current,
            "quarter_kelly_bootstrap_p_dd20": p_quarter,
            "half_kelly_bootstrap_p_dd20": p_half,
            "drawdown_constrained_c": analytic["bootstrap"]["first_crossing_c_bisection"],
            "engine_delta_total_pnl": engine["delta"]["total_pnl"],
            "engine_delta_annual_pnl": engine["delta"]["annual_pnl"],
            "engine_delta_sharpe": engine["delta"]["sharpe"],
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
