"""Orchestration helpers for local, research-only intraday experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from .capital import CapitalReuseConfig, apply_capital_feasibility
from .data import normalize_frame_map
from .eligibility import EligibilityConfig, calculate_eligibility
from .simulator import simulate_fixed_time_signals_audited
from .templates import (
    GapFirstHourConfig,
    IntradayShockConfig,
    generate_gap_first_hour_signals,
    generate_intraday_shock_signals,
)


@dataclass
class IntradayResearchResult:
    eligibility: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    execution_rejections: pd.DataFrame | None = None
    capital_audit: pd.DataFrame | None = None
    capital_feasible_trades: pd.DataFrame | None = None
    capital_rejections: pd.DataFrame | None = None
    capital_summary: pd.DataFrame | None = None


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "template_id",
                "n_trades",
                "n_days",
                "n_tickers",
                "mean_gross_return",
                "mean_net_return",
                "median_net_return",
                "win_rate_net",
            ]
        )
    rows: list[dict] = []
    for template_id, group in trades.groupby("template_id", sort=True):
        rows.append(
            {
                "template_id": template_id,
                "n_trades": len(group),
                "n_days": int(group["day_cluster"].nunique()),
                "n_tickers": int(group["ticker"].nunique()),
                "mean_gross_return": float(group["gross_return"].mean()),
                "mean_net_return": float(group["net_return"].mean()),
                "median_net_return": float(group["net_return"].median()),
                "win_rate_net": float(group["net_return"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_intraday_research(
    frames: Mapping[str, pd.DataFrame],
    metadata: pd.DataFrame | None = None,
    *,
    market_ticker: str = "SPY",
    candidates: Iterable[str] | None = None,
    templates: Iterable[str] = ("gap_first_hour", "intraday_shock"),
    eligibility_config: EligibilityConfig | None = None,
    gap_config: GapFirstHourConfig | None = None,
    shock_config: IntradayShockConfig | None = None,
    round_trip_cost_bps: float = 5.0,
    capital_config: CapitalReuseConfig | None = None,
    capital_priority_column: str | None = None,
) -> IntradayResearchResult:
    """Run selected templates without touching strategy or execution state."""

    eligibility_config = eligibility_config or EligibilityConfig()
    gap_config = gap_config or GapFirstHourConfig()
    shock_config = shock_config or IntradayShockConfig()
    normalized = normalize_frame_map(frames)
    eligibility = calculate_eligibility(
        normalized,
        eligibility_config,
        calendar_ticker=market_ticker,
        frames_are_normalized=True,
    )
    selected = set(templates)
    unknown = selected.difference({"gap_first_hour", "intraday_shock"})
    if unknown:
        raise ValueError(f"unknown intraday template(s): {sorted(unknown)}")

    signal_frames: list[pd.DataFrame] = []
    if "gap_first_hour" in selected:
        signal_frames.append(
            generate_gap_first_hour_signals(
                normalized,
                metadata,
                market_ticker=market_ticker,
                candidates=candidates,
                eligibility=eligibility,
                eligibility_config=eligibility_config,
                config=gap_config,
                frames_are_normalized=True,
            )
        )
    if "intraday_shock" in selected:
        signal_frames.append(
            generate_intraday_shock_signals(
                normalized,
                metadata,
                market_ticker=market_ticker,
                candidates=candidates,
                eligibility=eligibility,
                eligibility_config=eligibility_config,
                config=shock_config,
                frames_are_normalized=True,
            )
        )
    nonempty = [frame for frame in signal_frames if not frame.empty]
    signals = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    simulation = simulate_fixed_time_signals_audited(
        signals,
        normalized,
        round_trip_cost_bps=round_trip_cost_bps,
        frames_are_normalized=True,
    )
    trades = simulation.trades
    capital_audit = None
    capital_feasible_trades = None
    capital_rejections = None
    capital_summary = None
    if capital_config is not None:
        capital_result = apply_capital_feasibility(
            trades,
            capital_config,
            priority_column=capital_priority_column,
        )
        capital_audit = capital_result.audit
        capital_feasible_trades = capital_result.feasible_trades
        capital_rejections = capital_result.rejected_trades
        capital_summary = summarize_trades(capital_feasible_trades)

    return IntradayResearchResult(
        eligibility=eligibility,
        signals=signals,
        trades=trades,
        summary=summarize_trades(trades),
        execution_rejections=simulation.execution_rejections,
        capital_audit=capital_audit,
        capital_feasible_trades=capital_feasible_trades,
        capital_rejections=capital_rejections,
        capital_summary=capital_summary,
    )
