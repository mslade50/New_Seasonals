"""Compile the seasonality_flow_* study outputs into seasonality_flow_results.json."""
from __future__ import annotations
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
parts = {k: json.load(open(HERE / f"seasonality_flow_{k}.json")) for k in ["cells", "daily", "walkforward", "flow", "checks", "family"]}
cells = parts["cells"]["cells"]
book_month = [c for c in cells if c["strategy"] == "BOOK" and c["dim"] == "month"]
daily_book_month = [c for c in parts["daily"]["cells"] if c["strategy"] == "BOOK" and c["dim"] == "month"]
summary = dict(
    scripts=[str(HERE / f) for f in ["seasonality_flow_common.py", "seasonality_flow_1_cells.py", "seasonality_flow_2_daily.py", "seasonality_flow_3_walkforward.py",
                                     "seasonality_flow_4_flow.py", "seasonality_flow_5_checks.py", "seasonality_flow_6_family.py", "seasonality_flow_compile.py"]],
    basis="flat $750k ledger data/backtest_trades_full.parquet (4,696 rows, 3,483 fills after OVS tranche collapse), dist/data/trade_mtm.json + strategy_daily.json (2003-01..2026-08-07)",
    n_tests=parts["cells"]["n_tests"],
    headline=[
        "No (strategy x calendar) cell survives BH-FDR 10% on the year-paired test across 1,000+ trade-level cells; the two episode-clustered 'survivors' are tiny-complement artefacts (holiday none vs 2-trade pre/post cells).",
        "Fitted calendar multipliers (strategy x month/quarter/half/earnings/opex/TOM/dow) are at or below baseline out of sample: walk-forward strat x month dSharpe +0.04 (t 1.0), LOYO -0.06 with maxDD 2.6 pts worse; even in-sample dSharpe -0.03.",
        "Book daily variance IS seasonal (May-Oct sd 63 vs Nov-Apr 83 bps/day, year-paired t=-3.19, sd lower in 19/24 summers) but the mean moves with it (7.1 vs 11.6 bps), so mu/sigma^2 is flat (median yearly Kelly ratio 1.03). Mechanism: open risk (Spearman 0.77 across months), elasticity 0.51 = legs add in quadrature. Nothing to size.",
        "The one directionally consistent pattern: dip-buy family weaker May-Oct (avgR 0.30 vs 0.49, 6/6 strategies, episode t=-2.12, year t=-1.79, LOYO all negative) matching the Bouman-Jacobsen base rate; non-family control shows the opposite sign. But the gap only turned negative after 2013, the summer cell still earns R/risk 0.30, and a 0.75x cut costs 3% PnL for zero maxDD change. September alone is the zero-edge month (dip-buy R/risk 0.03, N=70).",
        "Signal FLOW state beats the calendar: OVS days with >=5 signals avgR 0.65 vs 0.23 (N=451, 38 days, q=0.002, LOYO diff never below +0.33, 10/11 years positive); single-signal OVS days 0.19. Book-level: signals after a quiet 21-session stretch are worse (low tercile 0.34 vs 0.53, q=0.09). A fitted flow multiplier still fails OOS (LOYO -2.6% PnL).",
        "Earnings season is BETTER, not worse, for every strategy after the existing blackouts (book avgR 0.56 vs 0.43, p=0.023 nominal, q=0.27; single-stock 7/7 same sign, LOYO [+0.06,+0.15]); the base-rate 'cut single stocks into earnings' prior loses 7% PnL. Existing OVS blackout / OLV override already remove the bad cell.",
        "External priors that do not transfer: Sep 0.5x (-4.5% PnL, 1/24 years better, Sep positive 15/23 years), May-Oct 0.75x (-11% PnL, no maxDD change), opex-week / TOM tilts (n.s. both ways), holiday adjacency (n.s.).",
        "January is the highest-variance month (sd 96 bps, worst day -4.9%, 9 drawdown troughs) but episodic: only 8/24 Januaries exceed 1.3x the rest-of-year sd, median ratio 0.99. Not sizeable.",
        "Friday MTM is weak (book 3.6 vs 10.3 bps/day, 21/24 years, t=-3.2; OVS 0.1 vs 1.6, OLV 0.0 vs 1.3) but half of it is structural zeros (MonFri/Monday Dip flat on Fridays) and sizing keys on signal date where Friday signals are fine (avgR 0.56). Not a sizing lever; an exit/hedge-timing question at most.",
    ],
    book_month_trade=[{k: c[k] for k in ["cell", "N", "avgR", "sdR", "R_per_risk", "flow_ratio", "sum_pnl", "t_year", "p_year", "q_p_year_fam_dim"]} for c in book_month],
    book_month_daily=[{k: c[k] for k in ["cell", "days", "mean_bps", "sd_bps", "sharpe", "sd_ratio", "kelly_ratio", "worst_day_pct", "t_year_mean", "p_year_mean", "t_year_sd", "p_year_sd", "levene_p"]} for c in daily_book_month],
)
out = dict(summary=summary, cells=parts["cells"], daily=parts["daily"], walkforward=parts["walkforward"], flow=parts["flow"], checks=parts["checks"], family=parts["family"])
(HERE / "seasonality_flow_results.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / "seasonality_flow_results.json", (HERE / "seasonality_flow_results.json").stat().st_size // 1024, "KB")
