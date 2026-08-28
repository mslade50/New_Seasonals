"""Research-only streaming experiment for the preregistered gap-reversal v1.

The module reads explicit local raw 15-minute parquets and writes only beneath
the repository's ignored ``artifacts`` directory.  It has no network, R2,
broker, scanner, staging, scheduling, or promotion dependency.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .data import BAR_DELTA, normalize_bars
from .diagnostics import (
    CAPACITY_SLOTS,
    DEFAULT_COST_GRID_BPS,
    PRIMARY_COST_BPS,
    annual_diagnostics,
    day_cluster_statistics,
    leave_one_year_out_diagnostics,
    make_daily_returns,
    materialize_cost_grid,
    rolling_five_year_train_one_year_test,
    side_diagnostics,
    validate_cost_grid,
)
from .eligibility import EligibilityConfig
from .streaming import (
    FULL_SESSION_LAST_MINUTE,
    SESSION_OPEN_MINUTE,
    RawPriceDiscontinuityConfig,
    _calculate_streaming_eligibility,
    _cost_grid_summary,
    _eligibility_summary,
    _expected_sessions,
    _frame_coverage_row,
    _market_status,
    _missing_coverage_row,
    _nearest_common_factor,
    _parquet_path,
    _sha256_file,
    reduce_bars_to_daily,
)

GAP_DOWN_LONG_TEMPLATE_ID: Final[str] = "gap_down_open_minus_025atr_long_v1"
GAP_UP_SHORT_TEMPLATE_ID: Final[str] = "gap_up_open_plus_075atr_short_v1"
GAP_REVERSAL_TEMPLATE_IDS: Final[tuple[str, str]] = (
    GAP_DOWN_LONG_TEMPLATE_ID,
    GAP_UP_SHORT_TEMPLATE_ID,
)
COMBINED_DIAGNOSTIC_ID: Final[str] = "gap_reversal_combined_diagnostic_v1"
ATR_SESSIONS: Final[int] = 14
LONG_LIMIT_ATR: Final[float] = 0.25
SHORT_LIMIT_ATR: Final[float] = 0.75
ACTIVATION_MINUTE: Final[int] = 9 * 60 + 45
LAST_ENTRY_MINUTE: Final[int] = 15 * 60 + 30
PRIMARY_CAPACITY_SLOTS: Final[int] = 3
MATERIAL_GAP_THRESHOLDS_ATR: Final[tuple[float, ...]] = (0.0, 0.25, 0.50, 1.00)

SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "template_id",
    "ticker",
    "sector",
    "trade_date",
    "side",
    "decision_ts",
    "activation_ts",
    "last_entry_bar_ts",
    "exit_bar_ts",
    "exit_ts",
    "prior_close_1545",
    "open_0930",
    "gap_return",
    "atr_14_lagged",
    "gap_atr",
    "limit_atr_multiple",
    "limit_price",
    "signal_strength",
    "price_proxy",
    "median_dollar_volume",
    "data_completeness",
)


@dataclass
class GapReversalResearchResult:
    data_dir: Path
    requested_tickers: tuple[str, ...]
    loaded_candidate_tickers: tuple[str, ...]
    expected_sessions: pd.DatetimeIndex
    full_sessions: pd.DatetimeIndex
    primary_sessions: pd.DatetimeIndex
    signals: pd.DataFrame
    signal_rejections: pd.DataFrame
    input_rejections: pd.DataFrame
    signal_generation_audit: pd.DataFrame
    eligibility_summary: pd.DataFrame
    trades: pd.DataFrame
    execution_rejections: pd.DataFrame
    opening_bar_sensitivity_trades: pd.DataFrame
    opening_bar_sensitivity_rejections: pd.DataFrame
    cost_grid_trades: pd.DataFrame
    cost_grid_summary: pd.DataFrame
    conditional_daily_returns: pd.DataFrame
    conditional_day_cluster_stats: pd.DataFrame
    slot_daily_returns: pd.DataFrame
    slot_summary: pd.DataFrame
    primary_daily_returns: pd.DataFrame
    primary_day_cluster_stats: pd.DataFrame
    annual_stats: pd.DataFrame
    leave_one_year_out: pd.DataFrame
    rolling_diagnostics: pd.DataFrame
    side_summary: pd.DataFrame
    material_gap_views: pd.DataFrame
    opening_bar_sensitivity_summary: pd.DataFrame
    opening_bar_sensitivity_stats: pd.DataFrame
    combined_diagnostic: pd.DataFrame
    primary_selected_candidates: pd.DataFrame
    primary_selected_fills: pd.DataFrame
    selected_outcome_summary: pd.DataFrame
    ticker_summary: pd.DataFrame
    sector_summary: pd.DataFrame
    coverage_audit: pd.DataFrame
    market_calendar_audit: pd.DataFrame
    eligibility_config: EligibilityConfig
    discontinuity_config: RawPriceDiscontinuityConfig
    cost_grid_bps: tuple[float, ...]
    bootstrap_reps: int
    metadata_fingerprint: str
    universe_fingerprint: str
    source_provenance: dict[str, object]


def _minute_number(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def _exact_series(work: pd.DataFrame, minute: int, column: str) -> pd.Series:
    selected = work.loc[work["minute"].eq(minute), ["trade_date", column]]
    return selected.set_index("trade_date")[column]


def calculate_lagged_atr(
    bars: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex,
    *,
    atr_sessions: int = ATR_SESSIONS,
    discontinuity_config: RawPriceDiscontinuityConfig | None = None,
    frames_are_normalized: bool = False,
) -> pd.DataFrame:
    """Return exact daily gap inputs and a T-1-shifted raw ATR.

    A true range is valid only for an exact full or observed early-close tape
    with positive volume in every bar and a valid prior session close. Missing
    expected sessions therefore invalidate, rather than shorten, a 14-session
    ATR window.
    """

    if atr_sessions < 1:
        raise ValueError("atr_sessions must be positive")
    discontinuity_config = discontinuity_config or RawPriceDiscontinuityConfig()
    work = bars.copy() if frames_are_normalized else normalize_bars(bars)
    work["trade_date"] = work["ts"].dt.normalize()
    work["minute"] = _minute_number(work["ts"])
    daily = reduce_bars_to_daily(work).reindex(expected_sessions).copy()
    grouped = work.groupby("trade_date", sort=True, observed=True)
    daily["daily_high"] = grouped["high"].max()
    daily["daily_low"] = grouped["low"].min()
    daily["open_0930_volume"] = _exact_series(work, SESSION_OPEN_MINUTE, "volume")
    daily["prior_close_1545"] = daily["close_1545"].shift(1).where(
        daily["volume_1545"].shift(1).gt(0)
    )
    valid_tape = (
        (daily["is_exact_full_session"].eq(True)
         | daily["is_exact_observed_early_close"].eq(True))
        & daily["zero_volume_bars"].eq(0)
        & daily["valid_session_close"].notna()
    )
    previous_valid_close = daily["valid_session_close"].shift(1)
    true_range = pd.concat(
        [
            daily["daily_high"] - daily["daily_low"],
            (daily["daily_high"] - previous_valid_close).abs(),
            (daily["daily_low"] - previous_valid_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    daily["daily_true_range"] = true_range.where(valid_tape & previous_valid_close.notna())
    discontinuity, _ = _nearest_common_factor(
        daily["open_0930"] / previous_valid_close,
        discontinuity_config,
    )
    daily["raw_price_discontinuity"] = discontinuity
    # A raw split day is not merely an untradeable current gap: letting its
    # cross-basis range enter ATR contaminates limits for the following 14
    # sessions. Mask the source TR so rolling min_periods keeps ATR unavailable
    # until the complete lookback is again on one raw-price basis.
    daily["daily_true_range"] = daily["daily_true_range"].where(~discontinuity)
    daily["atr_14_lagged"] = daily["daily_true_range"].shift(1).rolling(
        atr_sessions, min_periods=atr_sessions
    ).mean()
    required = work["minute"].between(SESSION_OPEN_MINUTE, FULL_SESSION_LAST_MINUTE)
    required_work = work.loc[required]
    daily["required_bar_count"] = required_work.groupby(
        "trade_date", observed=True
    ).size()
    daily["required_min_volume"] = required_work.groupby(
        "trade_date", observed=True
    )["volume"].min()
    daily["current_execution_tape_ok"] = (
        daily["is_exact_full_session"].eq(True)
        & daily["required_bar_count"].eq(26)
        & daily["required_min_volume"].gt(0)
        & daily["open_0930"].notna()
        & daily["open_0930_volume"].gt(0)
    )
    daily.index.name = "trade_date"
    return daily


def _normalize_metadata(
    candidates: tuple[str, ...], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise ValueError("non-empty sector metadata is required")
    work = metadata.copy()
    work.columns = [str(column).strip().lower() for column in work.columns]
    if "ticker" not in work.columns or "sector" not in work.columns:
        raise ValueError("sector metadata must contain ticker and sector columns")
    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    if work["ticker"].duplicated().any():
        duplicate = work.loc[work["ticker"].duplicated(), "ticker"].iloc[0]
        raise ValueError(f"duplicate sector metadata for {duplicate}")
    work = work.set_index("ticker").reindex(candidates)
    sectors = work["sector"].where(work["sector"].notna(), "").astype(str).str.strip()
    normalized = pd.DataFrame({"ticker": candidates, "sector": sectors.to_numpy()})
    exclusions = {
        row.ticker: "missing_sector_metadata"
        for row in normalized.itertuples(index=False)
        if not row.sector
    }
    return normalized, exclusions


def _clock_index(index: pd.Index, minute: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).normalize() + pd.Timedelta(minutes=minute)


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def _signal_time_quality_mask(
    daily: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.Series:
    """Return the universe-observable gate using only 09:30-known inputs."""

    aligned_eligibility = eligibility.reindex(daily.index)
    return (
        aligned_eligibility["eligible"].fillna(False)
        & daily["prior_close_1545"].notna()
        & daily["prior_close_1545"].gt(0)
        & daily["atr_14_lagged"].notna()
        & daily["atr_14_lagged"].gt(0)
        & daily["open_0930"].notna()
        & daily["open_0930"].gt(0)
    )


def _build_ticker_signals(
    *,
    ticker: str,
    sector: str,
    daily: pd.DataFrame,
    eligibility: pd.DataFrame,
    market_status: pd.Series,
    discontinuity_config: RawPriceDiscontinuityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    index = daily.index
    aligned_eligibility = eligibility.reindex(index)
    full_market = market_status.reindex(index).eq("full_session")
    eligible = aligned_eligibility["eligible"].fillna(False)
    prior_ok = daily["prior_close_1545"].notna() & daily["prior_close_1545"].gt(0)
    atr_ok = daily["atr_14_lagged"].notna() & daily["atr_14_lagged"].gt(0)
    tape_ok = daily["current_execution_tape_ok"].fillna(False)
    open_ok = daily["open_0930"].notna() & daily["open_0930"].gt(0)
    # Candidate rank is frozen from information available when the official
    # open is known. Current-session bar completeness and eventual volume are
    # execution outcomes, never candidate gates: otherwise a lower-ranked name
    # with a clean future tape can silently substitute for a selected name whose
    # later tape is missing or invalid.
    signal_time_quality = _signal_time_quality_mask(daily, eligibility)
    canonical_signal_quality = signal_time_quality & full_market
    ratio = daily["open_0930"] / daily["prior_close_1545"]
    gap_return = ratio - 1.0
    gap_atr = (daily["open_0930"] - daily["prior_close_1545"]) / daily[
        "atr_14_lagged"
    ]
    discontinuity, nearest_factor = _nearest_common_factor(
        ratio, discontinuity_config
    )

    rejection_rows: list[dict[str, object]] = []
    for day in index[eligible & ~canonical_signal_quality]:
        reasons: list[str] = []
        if not bool(full_market.loc[day]):
            reasons.append(str(market_status.reindex(index).loc[day]))
        if not bool(prior_ok.loc[day]):
            reasons.append("missing_valid_prior_scheduled_1545_close")
        if not bool(atr_ok.loc[day]):
            reasons.append("missing_valid_lagged_atr14")
        if not bool(open_ok.loc[day]):
            reasons.append("missing_valid_official_open")
        rejection_rows.append(
            {
                "ticker": ticker,
                "trade_date": day,
                "input_rejection_reasons": "|".join(dict.fromkeys(reasons)),
            }
        )

    signal_records: list[dict[str, object]] = []
    signal_rejections: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    definitions = (
        (GAP_DOWN_LONG_TEMPLATE_ID, -1, 1, LONG_LIMIT_ATR),
        (GAP_UP_SHORT_TEMPLATE_ID, 1, -1, SHORT_LIMIT_ATR),
    )
    for template_id, gap_sign, side, limit_multiple in definitions:
        signed = gap_return.lt(0) if gap_sign < 0 else gap_return.gt(0)
        candidates = canonical_signal_quality & signed
        filtered = candidates & discontinuity
        selected = candidates & ~discontinuity
        for day in index[filtered]:
            signal_rejections.append(
                {
                    "template_id": template_id,
                    "ticker": ticker,
                    "trade_date": day,
                    "signal_rejection_reason": "raw_price_common_split_factor",
                    "open_prior_close_ratio": float(ratio.loc[day]),
                    "nearest_common_factor": nearest_factor.loc[day],
                }
            )
        for day in index[selected]:
            open_price = float(daily.loc[day, "open_0930"])
            atr = float(daily.loc[day, "atr_14_lagged"])
            limit_price = open_price - limit_multiple * atr if side > 0 else open_price + limit_multiple * atr
            if not np.isfinite(limit_price) or limit_price <= 0:
                rejection_rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": day,
                        "input_rejection_reasons": "non_positive_limit_price",
                    }
                )
                continue
            signal_records.append(
                {
                    "template_id": template_id,
                    "ticker": ticker,
                    "sector": sector,
                    "trade_date": day,
                    "side": side,
                    "decision_ts": _clock_index([day], SESSION_OPEN_MINUTE)[0],
                    "activation_ts": _clock_index([day], ACTIVATION_MINUTE)[0],
                    "last_entry_bar_ts": _clock_index([day], LAST_ENTRY_MINUTE)[0],
                    "exit_bar_ts": _clock_index([day], FULL_SESSION_LAST_MINUTE)[0],
                    "exit_ts": _clock_index([day], FULL_SESSION_LAST_MINUTE)[0] + BAR_DELTA,
                    "prior_close_1545": float(daily.loc[day, "prior_close_1545"]),
                    "open_0930": open_price,
                    "gap_return": float(gap_return.loc[day]),
                    "atr_14_lagged": atr,
                    "gap_atr": float(gap_atr.loc[day]),
                    "limit_atr_multiple": limit_multiple,
                    "limit_price": limit_price,
                    "signal_strength": abs(float(gap_atr.loc[day])),
                    "price_proxy": aligned_eligibility.loc[day, "price_proxy"],
                    "median_dollar_volume": aligned_eligibility.loc[
                        day, "median_dollar_volume"
                    ],
                    "data_completeness": aligned_eligibility.loc[
                        day, "data_completeness"
                    ],
                }
            )
        audit_rows.append(
            {
                "template_id": template_id,
                "ticker": ticker,
                "n_expected_sessions": len(index),
                "n_eligible_sessions": int(eligible.sum()),
                "n_signal_time_quality_valid_sessions": int(
                    signal_time_quality.sum()
                ),
                "n_strictly_signed_gaps": int(candidates.sum()),
                "n_signed_gaps_with_later_execution_tape_failure": int(
                    (candidates & ~tape_ok).sum()
                ),
                "n_raw_discontinuity_filtered": int(filtered.sum()),
                "n_signals": sum(
                    record["template_id"] == template_id for record in signal_records
                ),
            }
        )
    signals = pd.DataFrame(signal_records).reindex(columns=SIGNAL_COLUMNS)
    if not signals.empty:
        signals = signals.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    return (
        signals,
        pd.DataFrame(signal_rejections),
        pd.DataFrame(rejection_rows),
        audit_rows,
    )


def simulate_gap_reversal_signals(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    opening_bar_touch: bool = False,
    frames_are_normalized: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate exact-limit first touches and the scheduled 15:45 exit."""

    if signals.empty:
        trade_columns = [*SIGNAL_COLUMNS, "entry_bar_ts", "entry_price", "fill_type", "exit_price", "gross_return", "round_trip_cost_bps", "net_return"]
        reject_columns = [*SIGNAL_COLUMNS, "execution_status", "execution_rejection_reason"]
        return pd.DataFrame(columns=trade_columns), pd.DataFrame(columns=reject_columns)
    if signals["ticker"].nunique() != 1:
        raise ValueError("simulate_gap_reversal_signals accepts one ticker at a time")
    work = bars.copy() if frames_are_normalized else normalize_bars(bars)
    work["trade_date"] = work["ts"].dt.normalize()
    work["minute"] = _minute_number(work["ts"])
    by_day = {day: group.set_index("minute") for day, group in work.groupby("trade_date")}
    trades: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    required_minutes = list(range(SESSION_OPEN_MINUTE, FULL_SESSION_LAST_MINUTE + 1, 15))
    entry_minutes = list(range(ACTIVATION_MINUTE, LAST_ENTRY_MINUTE + 1, 15))
    for signal in signals.to_dict("records"):
        day = pd.Timestamp(signal["trade_date"]).normalize()
        session = by_day.get(day)
        reason = ""
        if session is None or any(minute not in session.index for minute in required_minutes):
            reason = "missing_required_execution_bar"
        elif (session.loc[required_minutes, "volume"] <= 0).any():
            reason = "zero_volume_required_execution_bar"
        if reason:
            rejections.append(
                {**signal, "execution_status": reason, "execution_rejection_reason": reason}
            )
            continue
        side = int(signal["side"])
        limit_price = float(signal["limit_price"])
        fill_minute: int | None = None
        fill_type = ""
        if opening_bar_touch:
            opening = session.loc[SESSION_OPEN_MINUTE]
            opening_touched = (
                float(opening["low"]) <= limit_price
                if side > 0
                else float(opening["high"]) >= limit_price
            )
            if opening_touched:
                fill_minute = SESSION_OPEN_MINUTE
                fill_type = "optimistic_opening_bar_touch_exact_limit"
        if fill_minute is None:
            for minute in entry_minutes:
                bar = session.loc[minute]
                touched = (
                    float(bar["low"]) <= limit_price
                    if side > 0
                    else float(bar["high"]) >= limit_price
                )
                if touched:
                    fill_minute = minute
                    fill_type = (
                        "activation_open_through_limit_exact_limit"
                        if minute == ACTIVATION_MINUTE
                        and (
                            (side > 0 and float(bar["open"]) <= limit_price)
                            or (side < 0 and float(bar["open"]) >= limit_price)
                        )
                        else "first_touch_exact_limit"
                    )
                    break
        if fill_minute is None:
            reason = "limit_not_touched_before_1545"
            rejections.append(
                {**signal, "execution_status": reason, "execution_rejection_reason": reason}
            )
            continue
        exit_price = float(session.loc[FULL_SESSION_LAST_MINUTE, "close"])
        gross_return = side * (exit_price / limit_price - 1.0)
        trades.append(
            {
                **signal,
                "entry_bar_ts": day + pd.Timedelta(minutes=fill_minute),
                "entry_price": limit_price,
                "fill_type": fill_type,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "round_trip_cost_bps": PRIMARY_COST_BPS,
                "net_return": gross_return - PRIMARY_COST_BPS / 10_000.0,
            }
        )
    trade_frame = pd.DataFrame(trades)
    rejection_frame = pd.DataFrame(rejections)
    if not trade_frame.empty:
        trade_frame = trade_frame.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    if not rejection_frame.empty:
        rejection_frame = rejection_frame.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    return trade_frame, rejection_frame


def _merge_candidates_and_fills(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    execution_rejections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fill_keys = ["template_id", "ticker", "trade_date"]
    execution_columns = [
        "entry_bar_ts",
        "entry_price",
        "fill_type",
        "exit_price",
        "gross_return",
    ]
    available = [column for column in execution_columns if column in trades.columns]
    fills = (
        trades[fill_keys + available].copy()
        if not trades.empty
        else pd.DataFrame(columns=fill_keys + execution_columns)
    )
    candidates = signals.merge(fills, on=fill_keys, how="left", validate="one_to_one")
    candidates["filled"] = candidates["gross_return"].notna()
    if execution_rejections is not None:
        rejection_columns = [
            column
            for column in ("execution_status", "execution_rejection_reason")
            if column in execution_rejections.columns
        ]
        rejected = (
            execution_rejections[fill_keys + rejection_columns].copy()
            if not execution_rejections.empty
            else pd.DataFrame(columns=fill_keys + rejection_columns)
        )
        candidates = candidates.merge(
            rejected, on=fill_keys, how="left", validate="one_to_one"
        )
    return candidates


def select_candidate_slots(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    capacity_slots: int = PRIMARY_CAPACITY_SLOTS,
    min_abs_gap_atr: float = 0.0,
    cost_bps: float = PRIMARY_COST_BPS,
    eligible_sessions: pd.DatetimeIndex | None = None,
    execution_rejections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the actual candidates reserved before fills for one slot view.

    Ranking is performed independently by arm and day using information fixed
    at 09:30.  The result retains selected-but-unfilled candidates so an audit
    can prove that a lower-ranked eventual fill never substitutes for them.
    """

    if capacity_slots < 1:
        raise ValueError("capacity_slots must be positive")
    if not np.isfinite(min_abs_gap_atr) or min_abs_gap_atr < 0:
        raise ValueError("min_abs_gap_atr must be finite and non-negative")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if signals.empty:
        return pd.DataFrame(
            columns=[
                *SIGNAL_COLUMNS,
                "candidate_count",
                "selected_rank",
                "capacity_slots",
                "min_abs_gap_atr",
                "filled",
                "gross_return",
                "cost_bps",
                "net_return",
                "slot_return_contribution",
            ]
        )
    candidates = _merge_candidates_and_fills(
        signals, trades, execution_rejections=execution_rejections
    )
    if eligible_sessions is not None:
        allowed = pd.DatetimeIndex(eligible_sessions).normalize()
        candidates = candidates.loc[
            pd.to_datetime(candidates["trade_date"]).dt.normalize().isin(allowed)
        ]
    eligible = candidates.loc[
        candidates["signal_strength"].ge(min_abs_gap_atr)
    ]
    selected_frames: list[pd.DataFrame] = []
    for _, group in eligible.groupby(
        ["template_id", "trade_date"], sort=True, observed=True
    ):
        ranked = group.sort_values(
            ["signal_strength", "ticker"], ascending=[False, True]
        ).copy()
        selected = ranked.head(capacity_slots).copy()
        selected["candidate_count"] = len(group)
        selected["selected_rank"] = np.arange(1, len(selected) + 1, dtype=int)
        selected_frames.append(selected)
    if not selected_frames:
        return pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True)
    selected["capacity_slots"] = capacity_slots
    selected["min_abs_gap_atr"] = float(min_abs_gap_atr)
    selected["cost_bps"] = float(cost_bps)
    selected["net_return"] = np.where(
        selected["filled"],
        selected["gross_return"] - cost_bps / 10_000.0,
        np.nan,
    )
    selected["slot_return_contribution"] = np.where(
        selected["filled"], selected["net_return"] / capacity_slots, 0.0
    )
    if "execution_status" not in selected:
        selected["execution_status"] = pd.NA
    if "execution_rejection_reason" not in selected:
        selected["execution_rejection_reason"] = pd.NA
    selected["selected_outcome_status"] = np.where(
        selected["filled"], "filled", selected["execution_status"]
    )
    selected["later_tape_quality_failure"] = selected[
        "execution_status"
    ].isin(
        [
            "missing_required_execution_bar",
            "zero_volume_required_execution_bar",
        ]
    )
    return selected.sort_values(
        ["trade_date", "template_id", "selected_rank"], ignore_index=True
    )


def selected_slot_concentration_summaries(
    selected_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize only filled members of the preregistered selected slots."""

    columns = [
        "template_id",
        "concentration_population",
        "capacity_slots",
        "cost_bps",
        "group",
        "n_trades",
        "n_days",
        "mean_net_return",
        "median_net_return",
        "win_rate_net",
        "share_of_template_trades",
        "endpoint_contribution_sum",
        "absolute_endpoint_contribution_sum",
        "share_of_template_endpoint_return",
        "share_of_template_absolute_endpoint_contribution",
    ]
    if selected_candidates.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()
    required = {
        "template_id",
        "ticker",
        "sector",
        "trade_date",
        "filled",
        "net_return",
        "slot_return_contribution",
        "capacity_slots",
        "cost_bps",
    }
    missing = required.difference(selected_candidates.columns)
    if missing:
        raise ValueError(f"selected candidates missing concentration columns: {sorted(missing)}")
    filled = selected_candidates.loc[selected_candidates["filled"]].copy()
    if filled.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()
    if filled["capacity_slots"].nunique() != 1 or filled["cost_bps"].nunique() != 1:
        raise ValueError("concentration requires exactly one slot count and cost case")
    totals = filled.groupby("template_id", observed=True).size()
    return_totals = filled.groupby("template_id", observed=True)[
        "slot_return_contribution"
    ].sum()
    absolute_totals = (
        filled.assign(
            absolute_endpoint_contribution=filled["slot_return_contribution"].abs()
        )
        .groupby("template_id", observed=True)["absolute_endpoint_contribution"]
        .sum()
    )

    def summarize(group_column: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (template_id, label), group in filled.groupby(
            ["template_id", group_column], sort=True, observed=True, dropna=False
        ):
            returns = group["net_return"].astype(float)
            contribution = float(group["slot_return_contribution"].sum())
            absolute = float(group["slot_return_contribution"].abs().sum())
            template_return = float(return_totals[template_id])
            template_absolute = float(absolute_totals[template_id])
            rows.append(
                {
                    "template_id": template_id,
                    "concentration_population": "prefill_ranked_top3_filled_slots_10bps",
                    "capacity_slots": int(group["capacity_slots"].iloc[0]),
                    "cost_bps": float(group["cost_bps"].iloc[0]),
                    "group": str(label),
                    "n_trades": len(group),
                    "n_days": int(pd.to_datetime(group["trade_date"]).nunique()),
                    "mean_net_return": float(returns.mean()),
                    "median_net_return": float(returns.median()),
                    "win_rate_net": float(returns.gt(0).mean()),
                    "share_of_template_trades": float(len(group) / totals[template_id]),
                    "endpoint_contribution_sum": contribution,
                    "absolute_endpoint_contribution_sum": absolute,
                    "share_of_template_endpoint_return": (
                        contribution / template_return
                        if not np.isclose(template_return, 0.0)
                        else np.nan
                    ),
                    "share_of_template_absolute_endpoint_contribution": (
                        absolute / template_absolute
                        if not np.isclose(template_absolute, 0.0)
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    return summarize("ticker"), summarize("sector")


def _selected_outcome_summary(selected_candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "template_id",
        "n_selected_candidates",
        "n_fills",
        "n_no_touch",
        "n_later_tape_quality_failures",
        "later_tape_quality_failure_rate",
        "advance_blocked_by_selected_tape_quality",
    ]
    rows: list[dict[str, object]] = []
    for template_id in GAP_REVERSAL_TEMPLATE_IDS:
        group = selected_candidates.loc[
            selected_candidates["template_id"].eq(template_id)
        ]
        tape_failures = int(group.get("later_tape_quality_failure", pd.Series(dtype=bool)).sum())
        rows.append(
            {
                "template_id": template_id,
                "n_selected_candidates": len(group),
                "n_fills": int(group.get("filled", pd.Series(dtype=bool)).sum()),
                "n_no_touch": int(
                    group.get("execution_status", pd.Series(dtype=object))
                    .eq("limit_not_touched_before_1545")
                    .sum()
                ),
                "n_later_tape_quality_failures": tape_failures,
                "later_tape_quality_failure_rate": (
                    tape_failures / len(group) if len(group) else 0.0
                ),
                # Fail closed: any selected observation with unknowable outcome
                # prevents an advance label. The table still exposes the rate so
                # a later protocol can preregister a different materiality rule.
                "advance_blocked_by_selected_tape_quality": tape_failures > 0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def candidate_slot_portfolios(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    full_sessions: pd.DatetimeIndex,
    *,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    slots: tuple[int, ...] = CAPACITY_SLOTS,
    gap_thresholds_atr: tuple[float, ...] = MATERIAL_GAP_THRESHOLDS_ATR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pre-fill-ranked fixed-slot portfolios with unused slots in cash."""

    costs = validate_cost_grid(tuple(cost_grid_bps))
    if any(slot < 1 for slot in slots):
        raise ValueError("slots must be positive")
    thresholds = tuple(sorted({float(value) for value in gap_thresholds_atr}))
    if any(not np.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("gap thresholds must be finite and non-negative")
    columns = [
        "template_id",
        "cost_bps",
        "trade_date",
        "min_abs_gap_atr",
        "capacity_slots",
        "candidate_count",
        "slots_reserved",
        "fills",
        "unused_slots",
        "slot_portfolio_return",
    ]
    summary_columns = [
        "template_id",
        "cost_bps",
        "min_abs_gap_atr",
        "capacity_slots",
        "n_full_sessions",
        "n_candidate_days",
        "n_candidates_selected",
        "n_fills",
        "fill_rate_selected",
        "mean_candidate_day_return",
        "mean_full_session_return",
        "annualized_return_arithmetic",
        "annualized_volatility",
        "annualized_sharpe",
        "compound_return",
        "max_drawdown",
    ]
    full_index = pd.DatetimeIndex(full_sessions).normalize().sort_values().unique()
    if full_index.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=summary_columns)
    candidates = (
        _merge_candidates_and_fills(signals, trades)
        if not signals.empty
        else pd.DataFrame(columns=[*SIGNAL_COLUMNS, "gross_return", "filled"])
    )
    if not candidates.empty:
        candidates = candidates.loc[
            pd.to_datetime(candidates["trade_date"]).dt.normalize().isin(full_index)
        ]
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        eligible = candidates.loc[candidates["signal_strength"].ge(threshold)]
        for (template_id, trade_date), group in eligible.groupby(
            ["template_id", "trade_date"], sort=True, observed=True
        ):
            ranked = group.sort_values(
                ["signal_strength", "ticker"], ascending=[False, True]
            )
            for slot_count in slots:
                selected = ranked.head(slot_count)
                for cost_bps in costs:
                    filled = selected.loc[selected["filled"]]
                    slot_return = float(
                        (filled["gross_return"] - cost_bps / 10_000.0).sum()
                        / slot_count
                    )
                    rows.append(
                        {
                            "template_id": template_id,
                            "cost_bps": cost_bps,
                            "trade_date": pd.Timestamp(trade_date).normalize(),
                            "min_abs_gap_atr": threshold,
                            "capacity_slots": slot_count,
                            "candidate_count": len(group),
                            "slots_reserved": len(selected),
                            "fills": int(selected["filled"].sum()),
                            "unused_slots": slot_count - int(selected["filled"].sum()),
                            "slot_portfolio_return": slot_return,
                        }
                    )
    observed_daily = pd.DataFrame(rows, columns=columns)
    grid = pd.MultiIndex.from_product(
        [
            GAP_REVERSAL_TEMPLATE_IDS,
            costs,
            full_index,
            thresholds,
            slots,
        ],
        names=[
            "template_id",
            "cost_bps",
            "trade_date",
            "min_abs_gap_atr",
            "capacity_slots",
        ],
    ).to_frame(index=False)
    daily = grid.merge(
        observed_daily,
        on=[
            "template_id",
            "cost_bps",
            "trade_date",
            "min_abs_gap_atr",
            "capacity_slots",
        ],
        how="left",
        validate="one_to_one",
    )
    for column in ("candidate_count", "slots_reserved", "fills"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0).astype(int)
    unused_slots = pd.to_numeric(daily["unused_slots"], errors="coerce")
    daily["unused_slots"] = unused_slots.where(
        unused_slots.notna(), daily["capacity_slots"]
    ).astype(int)
    daily["slot_portfolio_return"] = pd.to_numeric(
        daily["slot_portfolio_return"], errors="coerce"
    ).fillna(0.0)
    daily = daily[columns].sort_values(
        [
            "cost_bps",
            "min_abs_gap_atr",
            "capacity_slots",
            "trade_date",
            "template_id",
        ],
        ignore_index=True,
    )
    summaries: list[dict[str, object]] = []
    for keys, group in daily.groupby(
        ["template_id", "cost_bps", "min_abs_gap_atr", "capacity_slots"],
        sort=True,
        observed=True,
    ):
        template_id, cost_bps, threshold, slot_count = keys
        all_sessions = group.set_index("trade_date")["slot_portfolio_return"].reindex(
            full_index, fill_value=0.0
        )
        candidate_days = group.loc[group["candidate_count"].gt(0)]
        mean_full = float(all_sessions.mean())
        volatility = float(all_sessions.std(ddof=1)) if len(all_sessions) > 1 else np.nan
        wealth = (1.0 + all_sessions).cumprod()
        max_drawdown = float((wealth / wealth.cummax() - 1.0).min()) if len(wealth) else np.nan
        selected_count = int(group["slots_reserved"].sum())
        fill_count = int(group["fills"].sum())
        summaries.append(
            {
                "template_id": template_id,
                "cost_bps": float(cost_bps),
                "min_abs_gap_atr": float(threshold),
                "capacity_slots": int(slot_count),
                "n_full_sessions": len(full_index),
                "n_candidate_days": int(candidate_days["trade_date"].nunique()),
                "n_candidates_selected": selected_count,
                "n_fills": fill_count,
                "fill_rate_selected": fill_count / selected_count if selected_count else np.nan,
                "mean_candidate_day_return": (
                    float(candidate_days["slot_portfolio_return"].mean())
                    if not candidate_days.empty
                    else np.nan
                ),
                "mean_full_session_return": mean_full,
                "annualized_return_arithmetic": mean_full * 252.0,
                "annualized_volatility": volatility * sqrt(252.0),
                "annualized_sharpe": (
                    mean_full / volatility * sqrt(252.0)
                    if np.isfinite(volatility) and volatility > 0
                    else np.nan
                ),
                "compound_return": float((1.0 + all_sessions).prod() - 1.0),
                "max_drawdown": max_drawdown,
            }
        )
    return daily, pd.DataFrame(summaries, columns=summary_columns)


def _primary_daily(slot_daily: pd.DataFrame) -> pd.DataFrame:
    selected = slot_daily.loc[
        slot_daily["min_abs_gap_atr"].eq(0.0)
        & slot_daily["capacity_slots"].eq(PRIMARY_CAPACITY_SLOTS)
    ].copy()
    selected = selected.rename(
        columns={"fills": "n_trades", "slot_portfolio_return": "daily_equal_notional_return"}
    )
    return selected[
        ["template_id", "cost_bps", "trade_date", "n_trades", "daily_equal_notional_return"]
    ]


def _combined_diagnostic(primary_daily: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "template_id",
        "cost_bps",
        "n_union_candidate_days",
        "mean_daily_return_fixed_half_per_arm",
        "median_daily_return_fixed_half_per_arm",
        "win_rate",
    ]
    if primary_daily.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for cost_bps, group in primary_daily.groupby("cost_bps", sort=True):
        pivot = group.pivot(index="trade_date", columns="template_id", values="daily_equal_notional_return")
        for template_id in GAP_REVERSAL_TEMPLATE_IDS:
            if template_id not in pivot:
                pivot[template_id] = 0.0
        combined = pivot[list(GAP_REVERSAL_TEMPLATE_IDS)].fillna(0.0).mean(axis=1)
        rows.append(
            {
                "template_id": COMBINED_DIAGNOSTIC_ID,
                "cost_bps": float(cost_bps),
                "n_union_candidate_days": len(combined),
                "mean_daily_return_fixed_half_per_arm": float(combined.mean()),
                "median_daily_return_fixed_half_per_arm": float(combined.median()),
                "win_rate": float(combined.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_gap_reversal_research(
    data_dir: str | Path,
    metadata: pd.DataFrame,
    candidates: Iterable[str],
    *,
    eligibility_config: EligibilityConfig | None = None,
    discontinuity_config: RawPriceDiscontinuityConfig | None = None,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    bootstrap_reps: int = 2_000,
    source_provenance: dict[str, object] | None = None,
) -> GapReversalResearchResult:
    """Run the frozen v1 experiment from explicit local parquet files."""

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"intraday data directory does not exist: {root}")
    requested = tuple(sorted({str(ticker).upper().strip() for ticker in candidates}))
    if not requested or any(not ticker for ticker in requested):
        raise ValueError("explicit non-empty candidate universe is required")
    costs = validate_cost_grid(tuple(cost_grid_bps))
    eligibility_config = eligibility_config or EligibilityConfig()
    discontinuity_config = discontinuity_config or RawPriceDiscontinuityConfig()
    normalized_metadata, exclusions = _normalize_metadata(requested, metadata)
    exclusions.update({ticker: "candidate_is_market_proxy" for ticker in requested if ticker == "SPY"})
    metadata_fingerprint = sha256(
        normalized_metadata.sort_values("ticker").to_csv(index=False).encode("utf-8")
    ).hexdigest()
    universe_fingerprint = sha256(("\n".join(requested) + "\n").encode("utf-8")).hexdigest()

    market_path = _parquet_path(root, "SPY")
    if not market_path.is_file():
        raise FileNotFoundError(f"required market proxy parquet is missing: {market_path}")
    market_bars = normalize_bars(pd.read_parquet(market_path), ticker="SPY")
    market_daily = reduce_bars_to_daily(market_bars)
    market_hash = _sha256_file(market_path)
    expected_sessions = _expected_sessions(market_daily)
    market_status = _market_status(market_daily, expected_sessions)
    full_sessions = pd.DatetimeIndex(market_status.index[market_status.eq("full_session")])
    observed_counts: Counter[pd.Timestamp] = Counter(
        pd.DatetimeIndex(market_daily.index).normalize().intersection(expected_sessions)
    )
    observable_universe_counts: Counter[pd.Timestamp] = Counter()
    coverage_rows = [
        _frame_coverage_row(
            ticker="SPY",
            role="market_proxy",
            path=market_path,
            input_hash=market_hash,
            daily=market_daily,
            expected_sessions=expected_sessions,
            discontinuity_config=discontinuity_config,
        )
    ]
    del market_bars

    metadata_by_ticker = normalized_metadata.set_index("ticker")["sector"].to_dict()
    signal_frames: list[pd.DataFrame] = []
    signal_rejection_frames: list[pd.DataFrame] = []
    input_rejection_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    execution_rejection_frames: list[pd.DataFrame] = []
    sensitivity_trade_frames: list[pd.DataFrame] = []
    sensitivity_rejection_frames: list[pd.DataFrame] = []
    loaded_candidates: list[str] = []

    for ticker in requested:
        path = _parquet_path(root, ticker)
        if ticker in exclusions or not path.is_file():
            reason = exclusions.get(ticker, "missing_candidate_file")
            exclusions[ticker] = reason
            coverage_rows.append(
                _missing_coverage_row(ticker, "candidate", path, reason, len(expected_sessions))
            )
            continue
        bars = normalize_bars(pd.read_parquet(path), ticker=ticker)
        observed_daily = reduce_bars_to_daily(bars)
        daily = calculate_lagged_atr(
            bars,
            expected_sessions,
            discontinuity_config=discontinuity_config,
            frames_are_normalized=True,
        )
        input_hash = _sha256_file(path)
        coverage_rows.append(
            _frame_coverage_row(
                ticker=ticker,
                role="candidate",
                path=path,
                input_hash=input_hash,
                daily=observed_daily,
                expected_sessions=expected_sessions,
                discontinuity_config=discontinuity_config,
            )
        )
        for day in pd.DatetimeIndex(observed_daily.index):
            if day in expected_sessions:
                observed_counts[pd.Timestamp(day)] += 1
        loaded_candidates.append(ticker)
        eligibility = _calculate_streaming_eligibility(
            daily, expected_sessions, eligibility_config
        )
        eligibility_rows.append(_eligibility_summary(ticker, eligibility, eligibility_config))
        signal_time_quality = _signal_time_quality_mask(daily, eligibility)
        for day in pd.DatetimeIndex(daily.index[signal_time_quality]):
            observable_universe_counts[pd.Timestamp(day)] += 1
        signals, signal_rejections, input_rejections, candidate_audit = _build_ticker_signals(
            ticker=ticker,
            sector=str(metadata_by_ticker[ticker]),
            daily=daily,
            eligibility=eligibility,
            market_status=market_status,
            discontinuity_config=discontinuity_config,
        )
        signal_frames.append(signals)
        if not signal_rejections.empty:
            signal_rejection_frames.append(signal_rejections)
        if not input_rejections.empty:
            input_rejection_frames.append(input_rejections)
        audit_rows.extend(candidate_audit)
        trades, execution_rejections = simulate_gap_reversal_signals(
            signals, bars, frames_are_normalized=True
        )
        sensitivity_trades, sensitivity_rejections = simulate_gap_reversal_signals(
            signals, bars, opening_bar_touch=True, frames_are_normalized=True
        )
        if not trades.empty:
            trade_frames.append(trades)
        if not execution_rejections.empty:
            execution_rejection_frames.append(execution_rejections)
        if not sensitivity_trades.empty:
            sensitivity_trade_frames.append(sensitivity_trades)
        if not sensitivity_rejections.empty:
            sensitivity_rejection_frames.append(sensitivity_rejections)
        del bars, daily, observed_daily, eligibility

    if not loaded_candidates:
        raise ValueError(f"no candidate could be evaluated: {dict(Counter(exclusions.values()))}")

    def concat_or_empty(frames: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(frames, ignore_index=True).sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )

    signals = concat_or_empty(signal_frames, SIGNAL_COLUMNS)
    signal_rejections = concat_or_empty(
        signal_rejection_frames,
        ["template_id", "ticker", "trade_date", "signal_rejection_reason"],
    )
    input_rejections = (
        pd.concat(input_rejection_frames, ignore_index=True).sort_values(
            ["trade_date", "ticker"], ignore_index=True
        )
        if input_rejection_frames
        else pd.DataFrame(columns=["ticker", "trade_date", "input_rejection_reasons"])
    )
    trade_columns = [*SIGNAL_COLUMNS, "entry_bar_ts", "entry_price", "fill_type", "exit_price", "gross_return", "round_trip_cost_bps", "net_return"]
    reject_columns = [*SIGNAL_COLUMNS, "execution_status", "execution_rejection_reason"]
    trades = concat_or_empty(trade_frames, trade_columns)
    execution_rejections = concat_or_empty(execution_rejection_frames, reject_columns)
    sensitivity_trades = concat_or_empty(sensitivity_trade_frames, trade_columns)
    sensitivity_rejections = concat_or_empty(sensitivity_rejection_frames, reject_columns)

    primary_sessions = pd.DatetimeIndex(
        [
            day
            for day in full_sessions
            if observable_universe_counts[pd.Timestamp(day)] > 0
        ]
    )
    if primary_sessions.empty:
        raise ValueError(
            "no canonical SPY full session has an observable signal-time universe"
        )

    # Make the canonical primary cohort explicit on every return-producing
    # surface, including the secondary conditional-on-fill diagnostics. This is
    # redundant with the common full-market signal gate by design and prevents
    # later refactors from leaking incomplete/noncanonical dates into outputs.
    def in_primary_cohort(frame: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(frame["trade_date"]).dt.normalize().isin(
            primary_sessions
        )

    signals = signals.loc[in_primary_cohort(signals)].reset_index(drop=True)
    trades = trades.loc[in_primary_cohort(trades)].reset_index(drop=True)
    execution_rejections = execution_rejections.loc[
        in_primary_cohort(execution_rejections)
    ].reset_index(drop=True)
    sensitivity_trades = sensitivity_trades.loc[
        in_primary_cohort(sensitivity_trades)
    ].reset_index(drop=True)
    sensitivity_rejections = sensitivity_rejections.loc[
        in_primary_cohort(sensitivity_rejections)
    ].reset_index(drop=True)

    cost_grid_trades = materialize_cost_grid(trades, costs)
    conditional_daily = make_daily_returns(cost_grid_trades)
    conditional_stats = day_cluster_statistics(
        conditional_daily,
        cost_grid_bps=costs,
        bootstrap_reps=bootstrap_reps,
        primary_template_ids=GAP_REVERSAL_TEMPLATE_IDS,
    )
    slot_daily, slot_summary = candidate_slot_portfolios(
        signals, trades, primary_sessions, cost_grid_bps=costs
    )
    primary_daily = _primary_daily(slot_daily)
    primary_stats = day_cluster_statistics(
        primary_daily,
        cost_grid_bps=costs,
        bootstrap_reps=bootstrap_reps,
        primary_template_ids=GAP_REVERSAL_TEMPLATE_IDS,
    )
    sensitivity_slot_daily, sensitivity_summary = candidate_slot_portfolios(
        signals, sensitivity_trades, primary_sessions, cost_grid_bps=costs
    )
    sensitivity_primary = _primary_daily(sensitivity_slot_daily)
    sensitivity_stats = day_cluster_statistics(
        sensitivity_primary,
        cost_grid_bps=costs,
        bootstrap_reps=bootstrap_reps,
        primary_template_ids=GAP_REVERSAL_TEMPLATE_IDS,
    )
    primary_selected_candidates = select_candidate_slots(
        signals,
        trades,
        capacity_slots=PRIMARY_CAPACITY_SLOTS,
        min_abs_gap_atr=0.0,
        cost_bps=PRIMARY_COST_BPS,
        eligible_sessions=primary_sessions,
        execution_rejections=execution_rejections,
    )
    primary_selected_fills = primary_selected_candidates.loc[
        primary_selected_candidates["filled"]
    ].copy()
    ticker_summary, sector_summary = selected_slot_concentration_summaries(
        primary_selected_candidates
    )
    selected_outcome_summary = _selected_outcome_summary(
        primary_selected_candidates
    )

    market_aligned = market_daily.reindex(expected_sessions)
    calendar_rows = [
        {
            "trade_date": day,
            "expected_nyse_session": True,
            "market_session_status": market_status.loc[day],
            "market_bars_in_session": market_aligned.loc[day, "bars_in_session"],
            "n_loaded_inputs_observed": observed_counts[day],
            "missing_from_all_loaded_inputs": observed_counts[day] == 0,
            "n_signal_time_observable_candidates": observable_universe_counts[day],
            "primary_inference_session": day in primary_sessions,
        }
        for day in expected_sessions
    ]
    material_views = slot_summary.loc[
        slot_summary["capacity_slots"].eq(PRIMARY_CAPACITY_SLOTS)
        & slot_summary["min_abs_gap_atr"].gt(0)
    ].copy()
    material_views["view_label"] = np.where(
        material_views["min_abs_gap_atr"].eq(0.50),
        "economically_cleaner_prespecified_view",
        "prespecified_descriptive_view",
    )
    return GapReversalResearchResult(
        data_dir=root,
        requested_tickers=requested,
        loaded_candidate_tickers=tuple(loaded_candidates),
        expected_sessions=expected_sessions,
        full_sessions=full_sessions,
        primary_sessions=primary_sessions,
        signals=signals,
        signal_rejections=signal_rejections,
        input_rejections=input_rejections,
        signal_generation_audit=pd.DataFrame(audit_rows),
        eligibility_summary=pd.DataFrame(eligibility_rows),
        trades=trades,
        execution_rejections=execution_rejections,
        opening_bar_sensitivity_trades=sensitivity_trades,
        opening_bar_sensitivity_rejections=sensitivity_rejections,
        cost_grid_trades=cost_grid_trades,
        cost_grid_summary=_cost_grid_summary(cost_grid_trades),
        conditional_daily_returns=conditional_daily,
        conditional_day_cluster_stats=conditional_stats,
        slot_daily_returns=slot_daily,
        slot_summary=slot_summary,
        primary_daily_returns=primary_daily,
        primary_day_cluster_stats=primary_stats,
        annual_stats=annual_diagnostics(primary_daily, primary_sessions),
        leave_one_year_out=leave_one_year_out_diagnostics(primary_daily),
        rolling_diagnostics=rolling_five_year_train_one_year_test(
            primary_daily, expected_sessions
        ),
        side_summary=side_diagnostics(cost_grid_trades),
        material_gap_views=material_views,
        opening_bar_sensitivity_summary=sensitivity_summary,
        opening_bar_sensitivity_stats=sensitivity_stats,
        combined_diagnostic=_combined_diagnostic(primary_daily),
        primary_selected_candidates=primary_selected_candidates,
        primary_selected_fills=primary_selected_fills,
        selected_outcome_summary=selected_outcome_summary,
        ticker_summary=ticker_summary,
        sector_summary=sector_summary,
        coverage_audit=pd.DataFrame(coverage_rows).sort_values(
            ["role", "ticker"], ignore_index=True
        ),
        market_calendar_audit=pd.DataFrame(calendar_rows),
        eligibility_config=eligibility_config,
        discontinuity_config=discontinuity_config,
        cost_grid_bps=costs,
        bootstrap_reps=bootstrap_reps,
        metadata_fingerprint=metadata_fingerprint,
        universe_fingerprint=universe_fingerprint,
        source_provenance=dict(source_provenance or {}),
    )


def _html_table(frame: pd.DataFrame, *, max_rows: int | None = 40) -> str:
    if frame.empty:
        return "<p class='muted'>No rows.</p>"
    display = frame.copy() if max_rows is None else frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in display.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in display.itertuples(index=False, name=None)
    )
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"


def render_gap_reversal_report(result: GapReversalResearchResult) -> str:
    primary = result.primary_day_cluster_stats.loc[
        result.primary_day_cluster_stats["primary_cost_case"]
    ]
    material = result.material_gap_views.loc[
        result.material_gap_views["cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    primary_20bps = result.primary_day_cluster_stats.loc[
        result.primary_day_cluster_stats["cost_bps"].eq(20.0)
    ]
    chronology_costs = (PRIMARY_COST_BPS, 20.0)
    annual = result.annual_stats.loc[
        result.annual_stats["cost_bps"].isin(chronology_costs)
    ]
    leave_one_year_out = result.leave_one_year_out.loc[
        result.leave_one_year_out["cost_bps"].isin(chronology_costs)
    ]
    rolling = result.rolling_diagnostics.loc[
        result.rolling_diagnostics["cost_bps"].isin(chronology_costs)
    ]
    ticker_concentration = result.ticker_summary.sort_values(
        ["template_id", "share_of_template_absolute_endpoint_contribution", "group"],
        ascending=[True, False, True],
    )
    sector_concentration = result.sector_summary.sort_values(
        ["template_id", "share_of_template_absolute_endpoint_contribution", "group"],
        ascending=[True, False, True],
    )
    selected_tape_blocker = bool(
        result.selected_outcome_summary[
            "advance_blocked_by_selected_tape_quality"
        ].any()
    )
    interpretation_status = (
        "INCONCLUSIVE / NON-ADVANCE — selected outcomes have missing or invalid later tape"
        if selected_tape_blocker
        else "PENDING STATISTICAL INTERPRETATION — no selected tape-quality blocker"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gap-Reversal v1 Research</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;background:#0b1020;color:#e7edf7}}main{{max-width:1240px;margin:auto;padding:32px}}
h1,h2{{color:#fff}}.warning{{padding:14px;border:1px solid #ba7b14;background:#2b210f;border-radius:8px}}.muted{{color:#9aa7ba}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}.card{{background:#151d31;padding:14px;border-radius:8px}}
.table-wrap{{overflow:auto;background:#11182a;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid #26314a;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:#19233a}}
</style></head><body><main><h1>Gap-Reversal v1</h1>
<p class="warning"><strong>Research only.</strong> No order, broker action, production write, schedule, deployment, or automatic promotion. The two arms are co-primary; the combined view cannot rescue either arm.</p>
<p class="warning"><strong>Interpretation status:</strong> {html.escape(interpretation_status)}</p>
<div class="cards"><div class="card"><strong>{len(result.requested_tickers)}</strong><br>requested tickers</div><div class="card"><strong>{len(result.loaded_candidate_tickers)}</strong><br>evaluated tickers</div><div class="card"><strong>{len(result.signals):,}</strong><br>primary-cohort gap candidates</div><div class="card"><strong>{len(result.trades):,}</strong><br>primary-cohort eligible fills</div><div class="card"><strong>{len(result.primary_selected_fills):,}</strong><br>selected top-3 fills</div></div>
<h2>Primary: fixed three slots, 10 bps, literal signed gaps</h2><p class="muted">Candidates rank at 09:30 by |gap/ATR|. Unfilled retained slots earn zero; lower-ranked candidates do not substitute. Holm covers the two arms.</p>{_html_table(primary)}
<h2>Required 20-bps robustness — both arms</h2>{_html_table(primary_20bps, max_rows=None)}
<h2>Selected-slot outcome completeness</h2><p class="muted">Any selected later-tape quality failure blocks an advance label; the slot stays cash and is never backfilled.</p>{_html_table(result.selected_outcome_summary, max_rows=None)}
<h2>Cost and slot robustness — complete grid for both arms</h2>{_html_table(result.slot_summary, max_rows=None)}
<h2>Prespecified material-gap views</h2><p class="muted">The 0.50-ATR row is the economically cleaner prespecified view, not a replacement primary or winner selection.</p>{_html_table(material)}
<h2>Conditional-on-fill secondary view</h2>{_html_table(result.conditional_day_cluster_stats)}
<h2>Optimistic opening-bar sensitivity</h2><p class="muted">Allows an exact-limit touch inside the 09:30 bar. It is non-primary because 15-minute data cannot establish the causal sequence.</p>{_html_table(result.opening_bar_sensitivity_stats.loc[result.opening_bar_sensitivity_stats["primary_cost_case"]])}
<h2>Combined diagnostic only</h2>{_html_table(result.combined_diagnostic)}
<h2>Chronology / annual results at 10 and 20 bps</h2>{_html_table(annual, max_rows=None)}
<h2>Leave-one-year-out at 10 and 20 bps</h2>{_html_table(leave_one_year_out, max_rows=None)}
<h2>Rolling five-year history → one-year test at 10 and 20 bps</h2>{_html_table(rolling, max_rows=None)}
<h2>Primary selected-slot concentration</h2><p class="muted">Ticker and sector concentration use only actual fills among each arm/day's pre-fill-ranked top three candidates at 10 bps. Each fill contributes return divided by the fixed three-slot denominator; lower-ranked fills are excluded.</p>
<h3>Ticker concentration — complete, sorted by absolute contribution</h3>{_html_table(ticker_concentration, max_rows=None)}
<h3>Sector concentration — complete, sorted by absolute contribution</h3>{_html_table(sector_concentration, max_rows=None)}
<h2>Coverage</h2>{_html_table(result.coverage_audit, max_rows=250)}
</main></body></html>"""


def write_gap_reversal_artifacts(
    result: GapReversalResearchResult, output_dir: str | Path
) -> Path:
    """Write a fresh self-contained bundle beneath the ignored artifact root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = (repo_root / "artifacts").resolve()
    target = Path(output_dir).resolve()
    try:
        target.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(f"output must stay beneath research artifact root: {artifact_root}") from exc
    if target == result.data_dir or target.is_relative_to(result.data_dir):
        raise ValueError("output directory cannot alias or sit inside the input data directory")
    if target.exists():
        raise FileExistsError(f"fresh output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    parquet_outputs = {
        "signals.parquet": result.signals,
        "signal_rejections.parquet": result.signal_rejections,
        "signal_input_rejections.parquet": result.input_rejections,
        "primary_cohort_eligible_fills_10bps.parquet": result.trades,
        "execution_rejections.parquet": result.execution_rejections,
        "opening_bar_sensitivity_trades.parquet": result.opening_bar_sensitivity_trades,
        "opening_bar_sensitivity_rejections.parquet": result.opening_bar_sensitivity_rejections,
        "cost_grid_trades.parquet": result.cost_grid_trades,
        "conditional_daily_returns.parquet": result.conditional_daily_returns,
        "slot_daily_returns.parquet": result.slot_daily_returns,
        "primary_three_slot_daily_returns.parquet": result.primary_daily_returns,
        "primary_selected_candidates_10bps.parquet": result.primary_selected_candidates,
        "primary_selected_fills_10bps.parquet": result.primary_selected_fills,
    }
    csv_outputs = {
        "signal_generation_audit.csv": result.signal_generation_audit,
        "eligibility_summary.csv": result.eligibility_summary,
        "cost_grid_summary.csv": result.cost_grid_summary,
        "primary_day_cluster_stats.csv": result.primary_day_cluster_stats,
        "conditional_day_cluster_stats.csv": result.conditional_day_cluster_stats,
        "annual_stats.csv": result.annual_stats,
        "leave_one_year_out.csv": result.leave_one_year_out,
        "rolling_5y_train_1y_test.csv": result.rolling_diagnostics,
        "side_summary.csv": result.side_summary,
        "candidate_slot_summary.csv": result.slot_summary,
        "material_gap_views.csv": result.material_gap_views,
        "opening_bar_sensitivity_summary.csv": result.opening_bar_sensitivity_summary,
        "opening_bar_sensitivity_stats.csv": result.opening_bar_sensitivity_stats,
        "combined_diagnostic.csv": result.combined_diagnostic,
        "selected_outcome_summary.csv": result.selected_outcome_summary,
        "ticker_summary.csv": result.ticker_summary,
        "sector_summary.csv": result.sector_summary,
        "coverage_audit.csv": result.coverage_audit,
        "market_calendar_audit.csv": result.market_calendar_audit,
    }
    for filename, frame in parquet_outputs.items():
        frame.to_parquet(target / filename, index=False)
    for filename, frame in csv_outputs.items():
        frame.to_csv(target / filename, index=False)
    report_path = target / "report.html"
    report_path.write_text(render_gap_reversal_report(result), encoding="utf-8")
    report_sha256 = _sha256_file(report_path)
    selected_tape_blocker = bool(
        result.selected_outcome_summary[
            "advance_blocked_by_selected_tape_quality"
        ].any()
    )
    manifest = {
        "schema_version": "gap_reversal_research.v1",
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "network_reads_or_writes": False,
        "automatic_promotion": False,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_dir": str(result.data_dir),
        "requested_tickers": list(result.requested_tickers),
        "loaded_candidate_tickers": list(result.loaded_candidate_tickers),
        "universe_sha256": result.universe_fingerprint,
        "metadata_sha256": result.metadata_fingerprint,
        "raw_source_provenance": result.source_provenance,
        "templates": list(GAP_REVERSAL_TEMPLATE_IDS),
        "combined_portfolio_is_diagnostic_only": True,
        "atr_sessions": ATR_SESSIONS,
        "atr_known_through": "T-1",
        "long_limit_atr": LONG_LIMIT_ATR,
        "short_limit_atr": SHORT_LIMIT_ATR,
        "primary_activation_clock": "09:45",
        "primary_opening_bar_excluded": True,
        "activation_open_through_fill_policy": "exact_limit_no_price_improvement",
        "last_entry_bar": "15:30",
        "exit_bar": "15:45_close",
        "literal_primary_min_abs_gap_atr": 0.0,
        "material_gap_views_atr": [0.25, 0.50, 1.00],
        "economically_cleaner_view_atr": 0.50,
        "primary_capacity_slots": PRIMARY_CAPACITY_SLOTS,
        "primary_session_denominator": (
            "exact_canonical_SPY_full_sessions_with_at_least_one_loaded_ticker_"
            "T_minus_1_eligible_and_valid_prior_1545_ATR14_and_0930_open"
        ),
        "n_primary_inference_sessions": len(result.primary_sessions),
        "no_candidate_arm_days_included_as_zero_return": True,
        "candidate_ranking": "09:30 descending abs(gap/ATR), ticker tie-break",
        "unused_slots_are_cash": True,
        "costs_charged_only_on_fills": True,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "cost_grid_bps": list(result.cost_grid_bps),
        "capacity_slots": list(CAPACITY_SLOTS),
        "holm_family_size": len(GAP_REVERSAL_TEMPLATE_IDS),
        "bootstrap_reps": result.bootstrap_reps,
        "bootstrap_seed_policy": "sha256(template_id|cost_bps)",
        "eligibility_config": asdict(result.eligibility_config),
        "raw_price_discontinuity_config": asdict(result.discontinuity_config),
        "n_signals": len(result.signals),
        "n_primary_cohort_eligible_fills": len(result.trades),
        "n_primary_selected_fills": len(result.primary_selected_fills),
        "concentration_population": "actual_fills_among_prefill_ranked_top3_per_arm_day_at_10bps",
        "concentration_return_weight": "net_return_divided_by_fixed_3_slot_denominator",
        "lower_ranked_fills_excluded_from_concentration": True,
        "selected_outcome_quality": result.selected_outcome_summary.to_dict("records"),
        "any_selected_tape_quality_failure_blocks_advance": True,
        "selected_tape_quality_blocker_active": selected_tape_blocker,
        "research_interpretation_status": (
            "INCONCLUSIVE_NON_ADVANCE_SELECTED_OUTCOME_MISSINGNESS"
            if selected_tape_blocker
            else "PENDING_STATISTICAL_INTERPRETATION"
        ),
        "n_execution_rejections": len(result.execution_rejections),
        "opening_bar_sensitivity_is_optimistic_non_primary": True,
        "report": {"filename": report_path.name, "sha256": report_sha256},
        "manifest_written_last": True,
    }
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return target
