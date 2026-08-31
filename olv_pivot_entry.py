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
OLV_PIVOT_LOW_COL = "OLV_ClosePivotLow40"
OLV_PIVOT_LOW_DATE_COL = "OLV_ClosePivotLow40Date"


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

    pivot_source_dates = pd.Series(values.index, index=values.index).shift(right_bars)
    return pd.DataFrame(
        {
            OLV_PIVOT_HIGH_COL: candidate.where(high_mask).ffill(),
            OLV_PIVOT_HIGH_DATE_COL: pivot_source_dates.where(high_mask).ffill(),
            OLV_PIVOT_LOW_COL: candidate.where(low_mask).ffill(),
            OLV_PIVOT_LOW_DATE_COL: pivot_source_dates.where(low_mask).ffill(),
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
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the row-specific OLV limit offset and optional skip action.

    The latest eligible high and latest eligible low are compared in price
    space.  Policy bands apply only when the high is the nearer of those two
    levels, matching the research definition.  With ``enabled=False`` the
    proposed decision is still returned for audit, while the actual decision
    falls back to the default offset and never skips.
    """
    spec = dict(policy or {})
    enabled = bool(spec.get("enabled", False))
    default_offset = float(spec.get("default_offset_atr", 0.25))
    version = str(spec.get("version", "olv_close_pivot_40_v1"))

    close_value = _finite_float(signal_close)
    atr_value = _finite_float(atr)
    high_value = _finite_float(pivot_high)
    low_value = _finite_float(pivot_low)

    nearest_type = ""
    nearest_level = None
    nearest_date = None
    if close_value is not None:
        if high_value is not None and low_value is not None:
            # Research tie-break: choices are [High, Low], so an exact tie is High.
            if abs(close_value - high_value) <= abs(close_value - low_value):
                nearest_type, nearest_level, nearest_date = "High", high_value, pivot_high_date
            else:
                nearest_type, nearest_level, nearest_date = "Low", low_value, pivot_low_date
        elif high_value is not None:
            nearest_type, nearest_level, nearest_date = "High", high_value, pivot_high_date
        elif low_value is not None:
            nearest_type, nearest_level, nearest_date = "Low", low_value, pivot_low_date

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
        "distance_atr": distance_atr,
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
        policy=policy,
    )
