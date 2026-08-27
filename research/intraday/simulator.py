"""Explicit event-clock execution simulator for intraday research signals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .data import BAR_DELTA, LookaheadError, normalize_frame_map

REQUIRED_SIGNAL_COLUMNS = (
    "template_id",
    "ticker",
    "sector",
    "trade_date",
    "side",
    "decision_ts",
    "feature_bar_ts",
    "feature_available_ts",
    "entry_bar_ts",
    "entry_ts",
    "exit_bar_ts",
    "exit_ts",
)


class MissingExecutionBarError(ValueError):
    """Raised by the strict wrapper when scheduled execution bars are absent."""


@dataclass
class FixedTimeSimulationResult:
    trades: pd.DataFrame
    execution_rejections: pd.DataFrame


def _validate_signal_clock(signals: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in REQUIRED_SIGNAL_COLUMNS if column not in signals.columns
    ]
    if missing:
        raise ValueError(f"signals missing event-clock columns: {missing}")
    work = signals.copy()
    for column in (
        "trade_date",
        "decision_ts",
        "feature_bar_ts",
        "feature_available_ts",
        "entry_bar_ts",
        "entry_ts",
        "exit_bar_ts",
        "exit_ts",
    ):
        work[column] = pd.to_datetime(work[column], errors="coerce")
    if work[list(REQUIRED_SIGNAL_COLUMNS[3:])].isna().any().any():
        raise ValueError("signals contain missing event-clock values")
    if not work["side"].isin((-1, 1)).all():
        raise ValueError("signal side must be -1 or +1")
    if work["feature_available_ts"].ne(work["feature_bar_ts"] + BAR_DELTA).any():
        raise LookaheadError("feature availability must equal feature bar close time")
    if work["feature_available_ts"].gt(work["decision_ts"]).any():
        row = work.loc[work["feature_available_ts"].gt(work["decision_ts"])].iloc[0]
        raise LookaheadError(
            f"{row['ticker']} uses a feature available after its decision timestamp"
        )
    if work["feature_available_ts"].ge(work["entry_ts"]).any():
        raise LookaheadError("entry must occur strictly after feature availability")
    if work["decision_ts"].ge(work["entry_ts"]).any():
        raise LookaheadError("entry must occur strictly after its signal decision")
    if work["entry_ts"].ne(work["entry_bar_ts"]).any():
        raise ValueError("entries must execute at the explicitly named bar open")
    if work["exit_ts"].ne(work["exit_bar_ts"] + BAR_DELTA).any():
        raise ValueError("exits must execute at the explicitly named 15m bar close")
    if work["exit_ts"].le(work["entry_ts"]).any():
        raise ValueError("exit must occur after entry")
    trade_dates = work["trade_date"].dt.normalize()
    for column in (
        "decision_ts",
        "feature_bar_ts",
        "feature_available_ts",
        "entry_bar_ts",
        "entry_ts",
        "exit_bar_ts",
        "exit_ts",
    ):
        if not work[column].dt.normalize().eq(trade_dates).all():
            raise LookaheadError(f"{column} must belong to trade_date")
    return work


def simulate_fixed_time_signals_audited(
    signals: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    *,
    round_trip_cost_bps: float = 5.0,
    frames_are_normalized: bool = False,
) -> FixedTimeSimulationResult:
    """Execute at named bar opens/closes and subtract a round-trip cost proxy.

    There is intentionally no stop, target, partial-fill, spread, quote, queue,
    borrow, or capital-sizing model.  ``round_trip_cost_bps`` is deducted once
    from gross return and should be stress-tested rather than optimized.
    """

    if not isfinite(round_trip_cost_bps) or round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be finite and non-negative")
    if signals.empty:
        empty = signals.copy()
        return FixedTimeSimulationResult(empty, empty.copy())
    work = _validate_signal_clock(signals)
    bars = dict(frames) if frames_are_normalized else normalize_frame_map(frames)
    indexed = {
        ticker: frame.set_index("ts", drop=False) for ticker, frame in bars.items()
    }
    output: list[dict] = []
    rejections: list[dict] = []

    for record in work.to_dict("records"):
        ticker = str(record["ticker"]).upper()
        if ticker not in indexed:
            raise KeyError(f"missing bars for signal ticker {ticker}")
        ticker_bars = indexed[ticker]
        entry_bar_ts = pd.Timestamp(record["entry_bar_ts"])
        scheduled_exit_bar_ts = pd.Timestamp(record["exit_bar_ts"])
        scheduled_exit_ts = pd.Timestamp(record["exit_ts"])
        if entry_bar_ts not in ticker_bars.index:
            rejected = dict(record)
            rejected.update(
                {
                    "execution_status": "missing_scheduled_entry_bar",
                    "execution_rejection_reason": f"missing entry bar {entry_bar_ts}",
                }
            )
            rejections.append(rejected)
            continue
        if scheduled_exit_bar_ts not in ticker_bars.index:
            rejected = dict(record)
            rejected.update(
                {
                    "execution_status": "missing_scheduled_exit_bar",
                    "execution_rejection_reason": (
                        f"missing exit bar {scheduled_exit_bar_ts}; no retrospective fallback"
                    ),
                }
            )
            rejections.append(rejected)
            continue

        exit_bar_ts = scheduled_exit_bar_ts

        entry_bar = ticker_bars.loc[entry_bar_ts]
        exit_bar = ticker_bars.loc[exit_bar_ts]
        entry_price = float(entry_bar["open"])
        exit_price = float(exit_bar["close"])
        side = int(record["side"])
        gross_return = side * (exit_price / entry_price - 1.0)
        net_return = gross_return - round_trip_cost_bps / 10_000.0
        result = dict(record)
        result.update(
            {
                "ticker": ticker,
                "scheduled_exit_bar_ts": scheduled_exit_bar_ts,
                "scheduled_exit_ts": scheduled_exit_ts,
                "exit_bar_ts": exit_bar_ts,
                "exit_ts": exit_bar_ts + BAR_DELTA,
                "execution_status": "executed",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "round_trip_cost_bps": float(round_trip_cost_bps),
                "net_return": net_return,
                "holding_minutes": int(
                    (exit_bar_ts + BAR_DELTA - pd.Timestamp(record["entry_ts"]))
                    / pd.Timedelta(minutes=1)
                ),
                "day_cluster": pd.Timestamp(record["trade_date"]).normalize(),
                "ticker_cluster": ticker,
                "sector_cluster": str(record["sector"]),
            }
        )
        output.append(result)
    trades = pd.DataFrame(output)
    if not trades.empty:
        trades = trades.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    rejected_frame = pd.DataFrame(rejections)
    if not rejected_frame.empty:
        rejected_frame = rejected_frame.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    return FixedTimeSimulationResult(trades, rejected_frame)


def simulate_fixed_time_signals(
    signals: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    *,
    round_trip_cost_bps: float = 5.0,
    frames_are_normalized: bool = False,
) -> pd.DataFrame:
    """Strict convenience wrapper that fails if an execution bar is missing."""

    result = simulate_fixed_time_signals_audited(
        signals,
        frames,
        round_trip_cost_bps=round_trip_cost_bps,
        frames_are_normalized=frames_are_normalized,
    )
    if not result.execution_rejections.empty:
        statuses = (
            result.execution_rejections["execution_status"].value_counts().to_dict()
        )
        raise MissingExecutionBarError(f"scheduled execution bars missing: {statuses}")
    return result.trades
