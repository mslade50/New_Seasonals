"""Causal 40/40 closing-pivot context and OLV entry-policy resolution.

The production scanner and portfolio backtester both consume the indicator
columns produced here.  A pivot at bar ``p`` is not exposed until bar
``p + right_bars`` has closed, which prevents the centered-window look-ahead
that would otherwise make historical pivots unavailable in real time.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


OLV_PIVOT_HIGH_COL = "OLV_ClosePivotHigh40"
OLV_PIVOT_HIGH_DATE_COL = "OLV_ClosePivotHigh40Date"
OLV_PIVOT_HIGH_SOURCE_AGE_COL = "OLV_ClosePivotHigh40SourceAgeBars"
OLV_PIVOT_LOW_COL = "OLV_ClosePivotLow40"
OLV_PIVOT_LOW_DATE_COL = "OLV_ClosePivotLow40Date"
OLV_PIVOT_LOW_SOURCE_AGE_COL = "OLV_ClosePivotLow40SourceAgeBars"


def causal_close_pivot_context(
    close: pd.Series,
    left_bars: int = 40,
    right_bars: int = 40,
) -> pd.DataFrame:
    """Return the latest eligible closing-price pivot high and low per bar.

    The rolling window is trailing at the *confirmation* bar.  At confirmation
    bar ``q``, the candidate pivot is ``q - right_bars`` and the window spans
    exactly ``left_bars`` observations before it and ``right_bars`` after it.
    This is equivalent to a centered pivot calculation followed by a
    ``right_bars`` shift, but is causal by construction.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must both be positive")

    values = pd.to_numeric(close, errors="coerce").astype(float)
    window = left_bars + right_bars + 1
    candidate = values.shift(right_bars)
    rolling_high = values.rolling(window, min_periods=window).max()
    rolling_low = values.rolling(window, min_periods=window).min()

    cand_arr = candidate.to_numpy(dtype=float)
    high_arr = rolling_high.to_numpy(dtype=float)
    low_arr = rolling_low.to_numpy(dtype=float)
    high_mask = pd.Series(
        np.isfinite(cand_arr)
        & np.isfinite(high_arr)
        & np.isclose(cand_arr, high_arr, rtol=0.0, atol=1e-8),
        index=values.index,
    )
    low_mask = pd.Series(
        np.isfinite(cand_arr)
        & np.isfinite(low_arr)
        & np.isclose(cand_arr, low_arr, rtol=0.0, atol=1e-8),
        index=values.index,
    )

    positions = pd.Series(
        np.arange(len(values), dtype=float), index=values.index
    )
    pivot_source_dates = pd.Series(values.index, index=values.index).shift(right_bars)
    pivot_source_positions = positions.shift(right_bars)
    high_source_positions = pivot_source_positions.where(high_mask).ffill()
    low_source_positions = pivot_source_positions.where(low_mask).ffill()
    return pd.DataFrame(
        {
            OLV_PIVOT_HIGH_COL: candidate.where(high_mask).ffill(),
            OLV_PIVOT_HIGH_DATE_COL: pivot_source_dates.where(high_mask).ffill(),
            OLV_PIVOT_HIGH_SOURCE_AGE_COL: positions - high_source_positions,
            OLV_PIVOT_LOW_COL: candidate.where(low_mask).ffill(),
            OLV_PIVOT_LOW_DATE_COL: pivot_source_dates.where(low_mask).ffill(),
            OLV_PIVOT_LOW_SOURCE_AGE_COL: positions - low_source_positions,
        },
        index=values.index,
    )


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def resolve_olv_pivot_entry(
    *,
    signal_close: Any,
    atr: Any,
    pivot_high: Any,
    pivot_low: Any,
    pivot_high_date: Any = None,
    pivot_low_date: Any = None,
    pivot_high_source_age_bars: Any = None,
    pivot_low_source_age_bars: Any = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the row-specific OLV limit offset and optional skip action.

    The latest causally eligible high and low are first expired independently
    when ``max_source_age_bars`` is configured, then the surviving levels are
    compared in price space. Policy bands apply only when the high is nearer,
    matching the research definition. With ``enabled=False`` the proposed
    decision is still returned for audit, while the actual decision falls back
    to the default offset and never skips.
    """
    spec = dict(policy or {})
    enabled = bool(spec.get("enabled", False))
    default_offset = float(spec.get("default_offset_atr", 0.25))
    version = str(spec.get("version", "olv_close_pivot_40_v2"))
    max_age_raw = spec.get("max_source_age_bars")
    max_source_age_bars = None
    if max_age_raw not in (None, ""):
        max_source_age_bars = _finite_float(max_age_raw)
        if max_source_age_bars is None or max_source_age_bars < 0:
            raise ValueError("max_source_age_bars must be a non-negative number")

    close_value = _finite_float(signal_close)
    atr_value = _finite_float(atr)
    high_value = _finite_float(pivot_high)
    low_value = _finite_float(pivot_low)
    high_age = _finite_float(pivot_high_source_age_bars)
    low_age = _finite_float(pivot_low_source_age_bars)

    # When an age cap is live, a level without a finite source age is not safe
    # to use. This makes stale/partial indicator data fall back to the ordinary
    # 0.25-ATR entry instead of silently applying a structural adjustment.
    high_expired = bool(
        high_value is not None
        and max_source_age_bars is not None
        and (high_age is None or high_age > max_source_age_bars)
    )
    low_expired = bool(
        low_value is not None
        and max_source_age_bars is not None
        and (low_age is None or low_age > max_source_age_bars)
    )
    eligible_high = high_value if not high_expired else None
    eligible_low = low_value if not low_expired else None

    nearest_type = ""
    nearest_level = None
    nearest_date = None
    nearest_source_age_bars = None
    if close_value is not None:
        if eligible_high is not None and eligible_low is not None:
            # Research tie-break: choices are [High, Low], so an exact tie is High.
            if abs(close_value - eligible_high) <= abs(close_value - eligible_low):
                nearest_type, nearest_level, nearest_date = (
                    "High", eligible_high, pivot_high_date
                )
                nearest_source_age_bars = high_age
            else:
                nearest_type, nearest_level, nearest_date = (
                    "Low", eligible_low, pivot_low_date
                )
                nearest_source_age_bars = low_age
        elif eligible_high is not None:
            nearest_type, nearest_level, nearest_date = (
                "High", eligible_high, pivot_high_date
            )
            nearest_source_age_bars = high_age
        elif eligible_low is not None:
            nearest_type, nearest_level, nearest_date = (
                "Low", eligible_low, pivot_low_date
            )
            nearest_source_age_bars = low_age

    distance_atr = None
    if (
        close_value is not None
        and nearest_level is not None
        and atr_value is not None
        and atr_value > 0
    ):
        distance_atr = (close_value - nearest_level) / atr_value

    proposed_action = "stage"
    proposed_offset = default_offset
    matched_rule = "default"
    if nearest_type == "High" and distance_atr is not None:
        for rule in spec.get("rules", []):
            lo = float(rule.get("min_exclusive", -np.inf))
            hi = float(rule.get("max_inclusive", np.inf))
            if distance_atr > lo and distance_atr <= hi:
                matched_rule = str(rule.get("name", "pivot_high_band"))
                proposed_action = str(rule.get("action", "stage")).strip().lower()
                if proposed_action != "skip":
                    proposed_offset = float(rule.get("offset_atr", default_offset))
                break

    actual_action = proposed_action if enabled else "stage"
    actual_offset = proposed_offset if enabled and actual_action != "skip" else default_offset
    return {
        "policy_enabled": enabled,
        "rule_version": version,
        "matched_rule": matched_rule,
        "nearest_type": nearest_type,
        "nearest_level": nearest_level,
        "nearest_date": nearest_date,
        "nearest_source_age_bars": nearest_source_age_bars,
        "distance_atr": distance_atr,
        "pivot_high_source_age_bars": high_age,
        "pivot_low_source_age_bars": low_age,
        "pivot_high_expired": high_expired,
        "pivot_low_expired": low_expired,
        "max_source_age_bars": max_source_age_bars,
        "proposed_action": proposed_action,
        "proposed_offset_atr": proposed_offset,
        "action": actual_action,
        "offset_atr": actual_offset,
        "skip": actual_action == "skip",
    }


def resolve_olv_pivot_entry_from_row(
    row: Mapping[str, Any],
    atr: Any,
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convenience adapter for an indicator row used by both consumers."""
    return resolve_olv_pivot_entry(
        signal_close=row.get("Close"),
        atr=atr,
        pivot_high=row.get(OLV_PIVOT_HIGH_COL),
        pivot_low=row.get(OLV_PIVOT_LOW_COL),
        pivot_high_date=row.get(OLV_PIVOT_HIGH_DATE_COL),
        pivot_low_date=row.get(OLV_PIVOT_LOW_DATE_COL),
        pivot_high_source_age_bars=row.get(OLV_PIVOT_HIGH_SOURCE_AGE_COL),
        pivot_low_source_age_bars=row.get(OLV_PIVOT_LOW_SOURCE_AGE_COL),
        policy=policy,
    )
