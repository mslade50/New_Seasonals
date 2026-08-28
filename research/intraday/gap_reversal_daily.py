"""Research-only daily-OHLC screen for asymmetric gap-reversal limits.

This module deliberately has no imports from strategy configuration, scanners,
broker code, cloud caches, workflows, or production writers.  Daily OHLC can
show that a limit was inside the day's range, but cannot establish whether an
order submitted after observing the official open preceded that touch.  The
result is therefore an optimistic broad-universe screen, not execution proof.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

LONG_ARM_ID: Final[str] = "gap_down_long_open_minus_0p25atr"
SHORT_ARM_ID: Final[str] = "gap_up_short_open_plus_0p75atr"
PRIMARY_ARM_IDS: Final[tuple[str, str]] = (LONG_ARM_ID, SHORT_ARM_ID)
PRIMARY_COST_BPS: Final[float] = 10.0
DEFAULT_COST_GRID_BPS: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 20.0, 30.0)
MATERIAL_GAP_THRESHOLDS_ATR: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0)
PRIMARY_SLOTS: Final[int] = 3
ATR_LOOKBACK: Final[int] = 14
DOLLAR_VOLUME_LOOKBACK: Final[int] = 20
MIN_HISTORY_SESSIONS: Final[int] = 14
MIN_PRIOR_CLOSE: Final[float] = 5.0
MIN_PRIOR_MEDIAN_DOLLAR_VOLUME: Final[float] = 25_000_000.0
COMMON_SPLIT_FACTORS: Final[tuple[float, ...]] = (
    0.10,
    0.20,
    0.25,
    1.0 / 3.0,
    0.50,
    2.0 / 3.0,
    0.75,
    0.80,
    1.25,
    4.0 / 3.0,
    1.50,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
)

UNIVERSE_EXCLUSION_SPEC: Final[dict[str, object]] = {
    "exact": ["CBZ", "THS"],
    "suffixes": ["=F", "-USD"],
    "caret_exceptions": ["^GSPC", "^NDX"],
    "rule": "exclude exact CBZ/THS, suffix =F/-USD, and caret tickers except ^GSPC/^NDX",
}


@dataclass(frozen=True)
class FrozenUniverse:
    """A deterministic, audited candidate universe."""

    tickers: tuple[str, ...]
    audit: pd.DataFrame
    source_row_count: int
    unique_pre_filter_count: int
    post_filter_count: int
    available_count: int
    missing_from_prices_count: int
    pre_filter_sha256: str
    post_filter_sha256: str
    available_sha256: str


@dataclass(frozen=True)
class DailyGapReversalResult:
    """All tables needed for an auditable research artifact."""

    as_of: pd.Timestamp
    study_sessions: pd.DatetimeIndex
    universe: FrozenUniverse
    normalized_row_count: int
    cutoff_row_count: int
    candidates: pd.DataFrame
    selected_orders: pd.DataFrame
    eligibility_summary: pd.DataFrame
    coverage_by_date: pd.DataFrame
    material_gap_daily: pd.DataFrame
    material_gap_summary: pd.DataFrame
    primary_stats: pd.DataFrame
    all_filled_summary: pd.DataFrame
    annual_diagnostics: pd.DataFrame
    leave_one_year_out: pd.DataFrame
    rolling_diagnostics: pd.DataFrame
    ticker_concentration: pd.DataFrame
    combined_diagnostic: pd.DataFrame
    cost_grid_bps: tuple[float, ...]
    bootstrap_reps: int


def _ticker_hash(tickers: list[str] | tuple[str, ...] | pd.Series) -> str:
    values = sorted({str(value).upper().strip() for value in tickers if str(value).strip()})
    return sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def universe_exclusion_reason(ticker: str) -> str | None:
    """Apply the frozen CSV_UNIVERSE eligibility exclusions."""

    value = str(ticker).upper().strip()
    if value in {"CBZ", "THS"}:
        return "excluded_exact"
    if value.endswith("=F"):
        return "excluded_futures_suffix"
    if value.endswith("-USD"):
        return "excluded_crypto_suffix"
    if value.startswith("^") and value not in {"^GSPC", "^NDX"}:
        return "excluded_caret_ticker"
    return None


def freeze_universe(
    source_tickers: list[str] | tuple[str, ...] | pd.Series,
    available_tickers: list[str] | tuple[str, ...] | pd.Series,
    *,
    source_row_count: int | None = None,
) -> FrozenUniverse:
    """Dedupe, filter, intersect, and hash a universe without prod imports."""

    raw = [str(value).upper().strip() for value in source_tickers if str(value).strip()]
    unique = sorted(set(raw))
    available = {str(value).upper().strip() for value in available_tickers}
    rows: list[dict[str, object]] = []
    selected: list[str] = []
    post_filter: list[str] = []
    for ticker in unique:
        reason = universe_exclusion_reason(ticker)
        if reason is not None:
            rows.append(
                {"ticker": ticker, "status": "excluded", "reason": reason, "in_prices": ticker in available}
            )
            continue
        post_filter.append(ticker)
        if ticker not in available:
            rows.append(
                {"ticker": ticker, "status": "missing", "reason": "missing_from_price_input", "in_prices": False}
            )
            continue
        selected.append(ticker)
        rows.append({"ticker": ticker, "status": "included", "reason": "", "in_prices": True})
    if not selected:
        raise ValueError("frozen universe has no eligible ticker present in price input")
    return FrozenUniverse(
        tickers=tuple(selected),
        audit=pd.DataFrame(rows).sort_values("ticker", ignore_index=True),
        source_row_count=len(raw) if source_row_count is None else int(source_row_count),
        unique_pre_filter_count=len(unique),
        post_filter_count=len(post_filter),
        available_count=len(selected),
        missing_from_prices_count=len(post_filter) - len(selected),
        pre_filter_sha256=_ticker_hash(unique),
        post_filter_sha256=_ticker_hash(post_filter),
        available_sha256=_ticker_hash(selected),
    )


def _canonical_columns(frame: pd.DataFrame) -> dict[str, object]:
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    required = ("ticker", "date", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in aliases]
    if missing:
        raise ValueError(f"daily price input is missing columns: {missing}")
    return {column: aliases[column] for column in required}


def normalize_daily_prices(
    frame: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Validate the long daily-OHLC contract and apply a completed-day cutoff."""

    if frame.empty:
        raise ValueError("daily price input is empty")
    columns = _canonical_columns(frame)
    work = frame[list(columns.values())].rename(columns={value: key for key, value in columns.items()}).copy()
    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    if work["ticker"].eq("").any():
        raise ValueError("daily price input contains blank tickers")
    parsed = pd.to_datetime(work["date"], errors="raise")
    if parsed.dt.tz is not None:
        raise ValueError("daily date values must be timezone-naive session dates")
    work["date"] = parsed.dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
        if not np.isfinite(work[column]).all():
            raise ValueError(f"daily price input contains non-finite {column}")
    if work[["open", "high", "low", "close"]].le(0).any().any():
        raise ValueError("daily OHLC prices must be positive")
    if work["volume"].lt(0).any():
        raise ValueError("daily volume cannot be negative")
    if (
        work["high"].lt(work[["open", "close", "low"]].max(axis=1)).any()
        or work["low"].gt(work[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("daily input contains impossible OHLC relationships")
    if work.duplicated(["ticker", "date"]).any():
        sample = work.loc[work.duplicated(["ticker", "date"], keep=False), ["ticker", "date"]].head()
        raise ValueError(f"daily input contains duplicate ticker/date rows: {sample.to_dict('records')}")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        raise ValueError("as_of must be a timezone-naive completed session date")
    cutoff = cutoff.normalize()
    original_count = len(work)
    work = work.loc[work["date"].le(cutoff)].sort_values(["ticker", "date"], ignore_index=True)
    if work.empty:
        raise ValueError(f"no daily rows remain at or before as_of={cutoff.date()}")
    return work, original_count


def _raw_discontinuity_flag(ratio: pd.Series) -> pd.Series:
    values = ratio.astype(float)
    extreme = values.lt(0.20) | values.gt(5.0)
    clean = values.fillna(1.0).to_numpy()
    factor_distance = np.column_stack(
        [np.abs(clean - factor) / factor for factor in COMMON_SPLIT_FACTORS]
    )
    near_factor = pd.Series(np.min(factor_distance, axis=1) <= 0.03, index=values.index)
    return values.sub(1.0).abs().ge(0.20) & near_factor | extreme


def prepare_daily_candidates(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute lagged PIT features and the two asymmetric candidate arms."""

    if prices.empty:
        raise ValueError("prices cannot be empty")
    work = prices.sort_values(["ticker", "date"], ignore_index=True).copy()
    grouped = work.groupby("ticker", sort=False, observed=True)
    work["prior_close"] = grouped["close"].shift(1)
    work["history_sessions"] = grouped.cumcount()
    true_range = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - work["prior_close"]).abs(),
            (work["low"] - work["prior_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    work["true_range"] = true_range
    work["atr14_lagged"] = true_range.groupby(work["ticker"], sort=False).transform(
        lambda values: values.rolling(ATR_LOOKBACK, min_periods=ATR_LOOKBACK).mean().shift(1)
    )
    work["daily_dollar_volume"] = work["close"] * work["volume"]
    work["prior_median_dollar_volume20"] = work["daily_dollar_volume"].groupby(
        work["ticker"], sort=False
    ).transform(
        lambda values: values.rolling(
            DOLLAR_VOLUME_LOOKBACK, min_periods=DOLLAR_VOLUME_LOOKBACK
        ).median().shift(1)
    )
    work["open_prior_close_ratio"] = work["open"] / work["prior_close"]
    work["raw_discontinuity_flag"] = _raw_discontinuity_flag(work["open_prior_close_ratio"])
    work["passes_history"] = work["history_sessions"].ge(MIN_HISTORY_SESSIONS)
    work["passes_price"] = work["prior_close"].ge(MIN_PRIOR_CLOSE)
    work["passes_dollar_volume"] = work["prior_median_dollar_volume20"].ge(
        MIN_PRIOR_MEDIAN_DOLLAR_VOLUME
    )
    work["passes_atr"] = work["atr14_lagged"].gt(0) & work["atr14_lagged"].notna()
    work["eligible"] = (
        work["passes_history"]
        & work["passes_price"]
        & work["passes_dollar_volume"]
        & work["passes_atr"]
        & ~work["raw_discontinuity_flag"]
    )
    gap = work["open"] - work["prior_close"]
    work["gap_dollars"] = gap
    work["gap_atr"] = gap.abs() / work["atr14_lagged"]

    eligible = work.loc[work["eligible"] & gap.ne(0)].copy()
    eligible["arm_id"] = np.where(gap.loc[eligible.index].lt(0), LONG_ARM_ID, SHORT_ARM_ID)
    eligible["side"] = np.where(eligible["arm_id"].eq(LONG_ARM_ID), 1, -1)
    eligible["limit_atr_from_open"] = np.where(eligible["side"].eq(1), 0.25, 0.75)
    eligible["limit_price"] = eligible["open"] + eligible["side"] * -1.0 * eligible[
        "limit_atr_from_open"
    ] * eligible["atr14_lagged"]
    eligible["valid_limit"] = eligible["limit_price"].gt(0)
    eligible = eligible.loc[eligible["valid_limit"]].copy()
    eligible["filled"] = np.where(
        eligible["side"].eq(1),
        eligible["low"].le(eligible["limit_price"]),
        eligible["high"].ge(eligible["limit_price"]),
    )
    eligible["fill_price"] = eligible["limit_price"].where(eligible["filled"])
    eligible["gross_return"] = np.where(
        eligible["filled"],
        eligible["side"] * (eligible["close"] / eligible["limit_price"] - 1.0),
        0.0,
    )
    candidate_columns = [
        "arm_id",
        "ticker",
        "date",
        "side",
        "open",
        "high",
        "low",
        "close",
        "prior_close",
        "true_range",
        "atr14_lagged",
        "prior_median_dollar_volume20",
        "gap_dollars",
        "gap_atr",
        "open_prior_close_ratio",
        "limit_atr_from_open",
        "limit_price",
        "filled",
        "fill_price",
        "gross_return",
    ]
    candidates = eligible[candidate_columns].sort_values(
        ["date", "arm_id", "gap_atr", "ticker"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )

    eligibility_rows: list[dict[str, object]] = []
    for ticker, group in work.groupby("ticker", sort=True, observed=True):
        eligible_group = group.loc[group["eligible"]]
        ticker_candidates = candidates.loc[candidates["ticker"].eq(ticker)]
        eligibility_rows.append(
            {
                "ticker": ticker,
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "n_rows": len(group),
                "n_eligible_rows": len(eligible_group),
                "n_candidate_orders": len(ticker_candidates),
                "n_discontinuity_filtered": int(group["raw_discontinuity_flag"].sum()),
                "n_price_gate_fail": int((~group["passes_price"]).sum()),
                "n_dollar_volume_gate_fail": int((~group["passes_dollar_volume"]).sum()),
                "n_atr_or_history_gate_fail": int((~(group["passes_atr"] & group["passes_history"])).sum()),
            }
        )
    return candidates, pd.DataFrame(eligibility_rows)


def _validated_costs(values: tuple[float, ...]) -> tuple[float, ...]:
    costs = tuple(sorted({float(value) for value in values}))
    if not costs or any(not np.isfinite(value) or value < 0 for value in costs):
        raise ValueError("cost grid must contain finite non-negative values")
    if PRIMARY_COST_BPS not in costs:
        raise ValueError("cost grid must contain the 10 bps primary case")
    return costs


def _stable_seed(key: str) -> int:
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")


def _bootstrap_ci(values: np.ndarray, *, key: str, reps: int) -> tuple[float, float]:
    if reps < 100:
        raise ValueError("bootstrap_reps must be at least 100")
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(_stable_seed(key))
    means = np.empty(reps, dtype=float)
    for start in range(0, reps, 128):
        count = min(128, reps - start)
        index = rng.integers(0, len(clean), size=(count, len(clean)))
        means[start : start + count] = clean[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _return_stats(values: pd.Series, *, key: str, bootstrap_reps: int) -> dict[str, float | int]:
    clean = values.astype(float).dropna().to_numpy()
    n = len(clean)
    mean = float(np.mean(clean)) if n else np.nan
    median = float(np.median(clean)) if n else np.nan
    std = float(np.std(clean, ddof=1)) if n > 1 else np.nan
    if n > 1 and np.isfinite(std) and std > 0:
        t_stat = mean / (std / sqrt(n))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        sharpe = mean / std * sqrt(252.0)
    elif n > 1 and std == 0 and mean != 0:
        t_stat = float(np.sign(mean) * np.inf)
        p_value = 0.0
        sharpe = float(np.sign(mean) * np.inf)
    else:
        t_stat = p_value = sharpe = np.nan
    ci_low, ci_high = _bootstrap_ci(clean, key=key, reps=bootstrap_reps)
    return {
        "n_sessions": n,
        "mean_session_return": mean,
        "median_session_return": median,
        "annualized_sharpe": sharpe,
        "t_stat": t_stat,
        "p_value_two_sided": p_value,
        "bootstrap_ci_2_5": ci_low,
        "bootstrap_ci_97_5": ci_high,
    }


def _holm_two_arms(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["holm_p_value_primary"] = np.nan
    mask = (
        output["material_gap_threshold_atr"].eq(0.0)
        & output["cost_bps"].eq(PRIMARY_COST_BPS)
        & output["arm_id"].isin(PRIMARY_ARM_IDS)
        & output["p_value_two_sided"].notna()
    )
    primary = output.loc[mask].sort_values(["p_value_two_sided", "arm_id"])
    running = 0.0
    for rank, (index, row) in enumerate(primary.iterrows()):
        adjusted = min(1.0, (len(PRIMARY_ARM_IDS) - rank) * float(row["p_value_two_sided"]))
        running = max(running, adjusted)
        output.loc[index, "holm_p_value_primary"] = running
    return output


def build_material_gap_views(
    candidates: pd.DataFrame,
    study_sessions: pd.DatetimeIndex,
    *,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    thresholds: tuple[float, ...] = MATERIAL_GAP_THRESHOLDS_ATR,
    slots: int = PRIMARY_SLOTS,
    bootstrap_reps: int = 2_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build causal-rank K-slot endpoints and all-filled conditional views."""

    costs = _validated_costs(cost_grid_bps)
    threshold_values = tuple(sorted({float(value) for value in thresholds}))
    if 0.0 not in threshold_values or any(value < 0 for value in threshold_values):
        raise ValueError("material-gap thresholds must be non-negative and include zero")
    if slots < 1:
        raise ValueError("slots must be positive")
    sessions = pd.DatetimeIndex(study_sessions).normalize().sort_values().unique()
    if sessions.empty:
        raise ValueError("study_sessions cannot be empty")
    ordered = candidates.sort_values(
        ["date", "arm_id", "gap_atr", "ticker"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    primary_selected = pd.DataFrame()
    for threshold in threshold_values:
        eligible = ordered.loc[ordered["gap_atr"].ge(threshold)].copy()
        eligible["candidate_rank"] = eligible.groupby(
            ["arm_id", "date"], sort=False, observed=True
        ).cumcount() + 1
        selected = eligible.loc[eligible["candidate_rank"].le(slots)].copy()
        if threshold == 0.0:
            primary_selected = selected.copy()
        for arm_id in PRIMARY_ARM_IDS:
            arm_candidates = eligible.loc[eligible["arm_id"].eq(arm_id)]
            arm_selected = selected.loc[selected["arm_id"].eq(arm_id)]
            available_by_day = arm_candidates.groupby("date", observed=True).size()
            selected_by_day = arm_selected.groupby("date", observed=True).size()
            fills_by_day = arm_selected.groupby("date", observed=True)["filled"].sum()
            gross_by_day = arm_selected.groupby("date", observed=True)["gross_return"].sum()
            for cost_bps in costs:
                daily = pd.DataFrame({"date": sessions})
                daily["arm_id"] = arm_id
                daily["cost_bps"] = cost_bps
                daily["material_gap_threshold_atr"] = threshold
                daily["capacity_slots"] = slots
                daily["candidate_orders_available"] = daily["date"].map(available_by_day).fillna(0).astype(int)
                daily["slots_used"] = daily["date"].map(selected_by_day).fillna(0).astype(int)
                daily["fills"] = daily["date"].map(fills_by_day).fillna(0).astype(int)
                gross_sum = daily["date"].map(gross_by_day).fillna(0.0).astype(float)
                daily["slot_portfolio_return"] = (
                    gross_sum - daily["fills"] * cost_bps / 10_000.0
                ) / slots
                daily_frames.append(daily)
                metrics = _return_stats(
                    daily["slot_portfolio_return"],
                    key=f"{arm_id}|{threshold:g}|{cost_bps:g}|top{slots}",
                    bootstrap_reps=bootstrap_reps,
                )
                summary_rows.append(
                    {
                        "arm_id": arm_id,
                        "cost_bps": cost_bps,
                        "material_gap_threshold_atr": threshold,
                        "capacity_slots": slots,
                        "n_candidate_orders": len(arm_candidates),
                        "n_selected_orders": len(arm_selected),
                        "n_selected_fills": int(arm_selected["filled"].sum()),
                        "selected_fill_rate": (
                            float(arm_selected["filled"].mean()) if len(arm_selected) else np.nan
                        ),
                        "n_candidate_days": int(arm_candidates["date"].nunique()),
                        "n_selected_days": int(arm_selected["date"].nunique()),
                        **metrics,
                    }
                )
                filled = arm_candidates.loc[arm_candidates["filled"]]
                net = filled["gross_return"] - cost_bps / 10_000.0
                filled_daily = (
                    pd.DataFrame({"date": filled["date"], "net_return": net})
                    .groupby("date", observed=True)["net_return"]
                    .mean()
                )
                conditional_rows.append(
                    {
                        "arm_id": arm_id,
                        "cost_bps": cost_bps,
                        "material_gap_threshold_atr": threshold,
                        "n_candidate_orders": len(arm_candidates),
                        "n_filled_orders": len(filled),
                        "fill_rate": float(len(filled) / len(arm_candidates)) if len(arm_candidates) else np.nan,
                        "n_fill_days": len(filled_daily),
                        "mean_filled_trade_net_return": float(net.mean()) if len(net) else np.nan,
                        "mean_filled_day_equal_notional_return": float(filled_daily.mean()) if len(filled_daily) else np.nan,
                        "conditional_view_only": True,
                    }
                )
    daily_output = pd.concat(daily_frames, ignore_index=True)
    summary = _holm_two_arms(pd.DataFrame(summary_rows))
    primary_stats = summary.loc[
        summary["material_gap_threshold_atr"].eq(0.0) & summary["cost_bps"].eq(PRIMARY_COST_BPS)
    ].reset_index(drop=True)
    primary_selected = primary_selected.copy()
    primary_selected["primary_cost_bps"] = PRIMARY_COST_BPS
    primary_selected["net_slot_return_10bps"] = np.where(
        primary_selected["filled"],
        primary_selected["gross_return"] - PRIMARY_COST_BPS / 10_000.0,
        0.0,
    ) / slots
    return (
        daily_output,
        summary,
        primary_stats,
        pd.DataFrame(conditional_rows),
        primary_selected,
    )


def _annual(primary_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = primary_daily.copy()
    work["year"] = work["date"].dt.year
    for (arm_id, year), group in work.groupby(["arm_id", "year"], sort=True, observed=True):
        returns = group["slot_portfolio_return"].astype(float)
        rows.append(
            {
                "arm_id": arm_id,
                "year": int(year),
                "n_sessions": len(group),
                "n_candidate_days": int(group["candidate_orders_available"].gt(0).sum()),
                "n_candidate_orders": int(group["candidate_orders_available"].sum()),
                "n_selected_orders": int(group["slots_used"].sum()),
                "n_fills": int(group["fills"].sum()),
                "mean_session_return": float(returns.mean()),
                "win_rate_sessions": float(returns.gt(0).mean()),
                "compound_return": float((1.0 + returns).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_year_out(primary_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = primary_daily.copy()
    work["year"] = work["date"].dt.year
    for arm_id, group in work.groupby("arm_id", sort=True, observed=True):
        full_mean = float(group["slot_portfolio_return"].mean())
        for year in sorted(group["year"].unique()):
            remaining = group.loc[group["year"].ne(year), "slot_portfolio_return"]
            rows.append(
                {
                    "arm_id": arm_id,
                    "omitted_year": int(year),
                    "n_remaining_sessions": len(remaining),
                    "full_sample_mean_return": full_mean,
                    "remaining_mean_return": float(remaining.mean()) if len(remaining) else np.nan,
                    "mean_sign_preserved": bool(np.sign(remaining.mean()) == np.sign(full_mean)) if len(remaining) and not np.isclose(full_mean, 0.0) else False,
                }
            )
    return pd.DataFrame(rows)


def _rolling(primary_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = primary_daily.copy()
    work["year"] = work["date"].dt.year
    min_year = int(work["year"].min())
    max_year = int(work["year"].max())
    for arm_id, group in work.groupby("arm_id", sort=True, observed=True):
        for test_year in range(min_year + 5, max_year + 1):
            train = group.loc[group["year"].between(test_year - 5, test_year - 1), "slot_portfolio_return"]
            test = group.loc[group["year"].eq(test_year), "slot_portfolio_return"]
            if train.empty and test.empty:
                continue
            rows.append(
                {
                    "arm_id": arm_id,
                    "test_year": test_year,
                    "train_start_year": test_year - 5,
                    "train_end_year": test_year - 1,
                    "n_train_sessions": len(train),
                    "n_test_sessions": len(test),
                    "train_mean_return": float(train.mean()) if len(train) else np.nan,
                    "test_mean_return": float(test.mean()) if len(test) else np.nan,
                    "rule_selected_or_refit": False,
                }
            )
    return pd.DataFrame(rows)


def _ticker_concentration(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=["arm_id", "ticker", "n_selected", "n_fills", "endpoint_contribution", "absolute_contribution_share"]
        )
    rows: list[dict[str, object]] = []
    for (arm_id, ticker), group in selected.groupby(["arm_id", "ticker"], sort=True, observed=True):
        contribution = float(group["net_slot_return_10bps"].sum())
        rows.append(
            {
                "arm_id": arm_id,
                "ticker": ticker,
                "n_selected": len(group),
                "n_fills": int(group["filled"].sum()),
                "endpoint_contribution": contribution,
            }
        )
    output = pd.DataFrame(rows)
    absolute_totals = output.assign(abs_value=output["endpoint_contribution"].abs()).groupby(
        "arm_id", observed=True
    )["abs_value"].transform("sum")
    output["absolute_contribution_share"] = np.where(
        absolute_totals.gt(0), output["endpoint_contribution"].abs() / absolute_totals, np.nan
    )
    return output


def _combined(primary_daily: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    pivot = primary_daily.pivot(index="date", columns="arm_id", values="slot_portfolio_return").fillna(0.0)
    for arm_id in PRIMARY_ARM_IDS:
        if arm_id not in pivot:
            pivot[arm_id] = 0.0
    combined = (pivot[LONG_ARM_ID] + pivot[SHORT_ARM_ID]) / 2.0
    metrics = _return_stats(combined, key="combined_50_50|10bps", bootstrap_reps=bootstrap_reps)
    return pd.DataFrame(
        [
            {
                "diagnostic_id": "fixed_50_50_long_short_arms",
                "cost_bps_per_filled_order": PRIMARY_COST_BPS,
                "long_arm_weight": 0.5,
                "short_arm_weight": 0.5,
                "primary_endpoint": False,
                **metrics,
                "compound_return": float((1.0 + combined).prod() - 1.0),
            }
        ]
    )


def run_daily_gap_reversal_research(
    normalized_prices: pd.DataFrame,
    universe: FrozenUniverse,
    *,
    as_of: str | pd.Timestamp,
    original_row_count: int | None = None,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    bootstrap_reps: int = 2_000,
) -> DailyGapReversalResult:
    """Evaluate both preregistered arms from an explicit normalized frame."""

    cutoff = pd.Timestamp(as_of).normalize()
    selected = normalized_prices.loc[normalized_prices["ticker"].isin(universe.tickers)].copy()
    if selected.empty:
        raise ValueError("no rows remain after applying the frozen universe")
    candidates, eligibility = prepare_daily_candidates(selected)
    if candidates.empty:
        raise ValueError("no eligible non-zero-gap candidate orders were generated")
    first_candidate = pd.Timestamp(candidates["date"].min()).normalize()
    sessions = pd.DatetimeIndex(
        sorted(selected.loc[selected["date"].ge(first_candidate), "date"].unique())
    )
    coverage = (
        selected.groupby("date", sort=True, observed=True)["ticker"]
        .nunique()
        .rename("n_tickers_observed")
        .reindex(sessions, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    daily, summary, primary_stats, conditional, selected_orders = build_material_gap_views(
        candidates,
        sessions,
        cost_grid_bps=cost_grid_bps,
        bootstrap_reps=bootstrap_reps,
    )
    primary_daily = daily.loc[
        daily["material_gap_threshold_atr"].eq(0.0)
        & daily["cost_bps"].eq(PRIMARY_COST_BPS)
    ].copy()
    return DailyGapReversalResult(
        as_of=cutoff,
        study_sessions=sessions,
        universe=universe,
        normalized_row_count=int(original_row_count if original_row_count is not None else len(normalized_prices)),
        cutoff_row_count=len(normalized_prices),
        candidates=candidates,
        selected_orders=selected_orders,
        eligibility_summary=eligibility,
        coverage_by_date=coverage,
        material_gap_daily=daily,
        material_gap_summary=summary,
        primary_stats=primary_stats,
        all_filled_summary=conditional,
        annual_diagnostics=_annual(primary_daily),
        leave_one_year_out=_leave_one_year_out(primary_daily),
        rolling_diagnostics=_rolling(primary_daily),
        ticker_concentration=_ticker_concentration(selected_orders),
        combined_diagnostic=_combined(primary_daily, bootstrap_reps),
        cost_grid_bps=_validated_costs(cost_grid_bps),
        bootstrap_reps=bootstrap_reps,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bps(value: object) -> str:
    numeric = float(value)
    return "—" if not np.isfinite(numeric) else f"{numeric * 10_000:+.2f}"


def build_daily_gap_html(result: DailyGapReversalResult) -> str:
    """Return a concise self-contained report with the causal warning upfront."""

    primary_rows = []
    for row in result.primary_stats.itertuples(index=False):
        primary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.arm_id))}</td>"
            f"<td>{int(row.n_candidate_orders):,}</td>"
            f"<td>{int(row.n_selected_fills):,} / {int(row.n_selected_orders):,}</td>"
            f"<td>{float(row.selected_fill_rate) * 100:.1f}%</td>"
            f"<td>{_format_bps(row.mean_session_return)}</td>"
            f"<td>[{_format_bps(row.bootstrap_ci_2_5)}, {_format_bps(row.bootstrap_ci_97_5)}]</td>"
            f"<td>{float(row.holm_p_value_primary):.4g}</td>"
            "</tr>"
        )
    sensitivity_rows = []
    view = result.material_gap_summary.loc[result.material_gap_summary["cost_bps"].eq(PRIMARY_COST_BPS)]
    for row in view.itertuples(index=False):
        sensitivity_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.arm_id))}</td><td>{float(row.material_gap_threshold_atr):g}</td>"
            f"<td>{int(row.n_candidate_orders):,}</td><td>{float(row.selected_fill_rate) * 100:.1f}%</td>"
            f"<td>{_format_bps(row.mean_session_return)}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily OHLC Gap-Reversal Screen</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#16202a}}h1,h2{{line-height:1.2}}.warning{{background:#fff0db;border:2px solid #c65b00;padding:16px;border-radius:8px}}.meta{{color:#53606d}}table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{padding:8px;border-bottom:1px solid #d8dee4;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef2f5;padding:2px 4px}}</style></head>
<body><h1>Daily OHLC gap-reversal screen</h1>
<p class="meta">Research only · completed-session cutoff {result.as_of.date()} · {len(result.universe.tickers):,} available tickers · {len(result.study_sessions):,} study sessions</p>
<div class="warning"><strong>Optimistic range-touch screen, not causal execution evidence.</strong> Daily OHLC does not reveal whether the low/high occurred before an order could be placed after observing the official open. Any apparent edge must survive a 15-minute (preferably finer) causal rerun; this report cannot override that test.</div>
<h2>Co-primary top-three endpoints at 10 bps</h2>
<p>Each arm ranks candidate orders at the open by absolute gap/lagged ATR, uses three equal slots, leaves unused and unfilled slots in cash, and charges cost only to fills.</p>
<table><thead><tr><th>Arm</th><th>All candidates</th><th>Selected fills / orders</th><th>Fill rate</th><th>Mean/session (bps)</th><th>95% bootstrap CI (bps)</th><th>Holm p</th></tr></thead><tbody>{''.join(primary_rows)}</tbody></table>
<h2>Prespecified material-gap views at 10 bps</h2>
<table><thead><tr><th>Arm</th><th>Minimum |gap| / ATR</th><th>Candidates</th><th>Selected fill rate</th><th>Mean/session (bps)</th></tr></thead><tbody>{''.join(sensitivity_rows)}</tbody></table>
<h2>Definition and boundaries</h2><ul>
<li>Long: open below prior close; buy limit = open − 0.25 × lagged ATR14.</li>
<li>Short: open above prior close; sell limit = open + 0.75 × lagged ATR14.</li>
<li>ATR14 is the simple mean of 14 completed daily true ranges, shifted through T−1. Exit is the same-day close.</li>
<li>PIT gates: prior close ≥ $5; prior 20-session median daily dollar volume ≥ $25m; at least 14 prior sessions; conservative raw-price discontinuity filter.</li>
<li>No spread, queue, partial fill, opening latency, borrow, halt, news, earnings, integer-share, or market-impact model.</li>
</ul></body></html>"""


def write_daily_gap_research_artifacts(
    result: DailyGapReversalResult,
    output_dir: str | Path,
    *,
    input_provenance: dict[str, object],
) -> Path:
    """Write only a fresh ignored-artifact bundle, with the manifest last."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact_root = (repo_root / "artifacts").resolve()
    target = Path(output_dir).resolve()
    try:
        target.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(f"output must stay beneath artifact root: {artifact_root}") from exc
    input_path = Path(str(input_provenance["price_input_path"])).resolve()
    if target == input_path or input_path.is_relative_to(target):
        raise ValueError("output directory cannot contain or alias the price input")
    if target.exists():
        raise FileExistsError(f"fresh output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)

    parquet_outputs = {
        "selected_orders_primary.parquet": result.selected_orders,
        "top3_daily_returns.parquet": result.material_gap_daily,
    }
    csv_outputs = {
        "universe_audit.csv": result.universe.audit,
        "eligibility_summary.csv": result.eligibility_summary,
        "coverage_by_date.csv": result.coverage_by_date,
        "material_gap_summary.csv": result.material_gap_summary,
        "primary_stats.csv": result.primary_stats,
        "all_filled_conditional_summary.csv": result.all_filled_summary,
        "annual_diagnostics.csv": result.annual_diagnostics,
        "leave_one_year_out.csv": result.leave_one_year_out,
        "rolling_5y_train_1y_test.csv": result.rolling_diagnostics,
        "ticker_concentration.csv": result.ticker_concentration,
        "combined_diagnostic.csv": result.combined_diagnostic,
    }
    for filename, frame in parquet_outputs.items():
        frame.to_parquet(target / filename, index=False)
    for filename, frame in csv_outputs.items():
        frame.to_csv(target / filename, index=False)
    report_path = target / "report.html"
    report_path.write_text(build_daily_gap_html(result), encoding="utf-8")

    output_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(target.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "schema_version": "daily_gap_reversal_research.v1",
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "manifest_written_last": True,
        "as_of_completed_session": str(result.as_of.date()),
        "daily_ohlc_range_touch_is_optimistic": True,
        "cannot_override_15m_causal_test": True,
        "input_provenance": dict(input_provenance),
        "universe": {
            "filter_spec": UNIVERSE_EXCLUSION_SPEC,
            "source_row_count": result.universe.source_row_count,
            "unique_pre_filter_count": result.universe.unique_pre_filter_count,
            "post_filter_count": result.universe.post_filter_count,
            "available_count": result.universe.available_count,
            "missing_from_prices_count": result.universe.missing_from_prices_count,
            "pre_filter_sha256": result.universe.pre_filter_sha256,
            "post_filter_sha256": result.universe.post_filter_sha256,
            "available_sha256": result.universe.available_sha256,
        },
        "row_counts": {
            "price_rows_before_as_of_cutoff": result.normalized_row_count,
            "price_rows_at_or_before_as_of": result.cutoff_row_count,
            "candidate_orders": len(result.candidates),
            "selected_primary_orders": len(result.selected_orders),
            "study_sessions": len(result.study_sessions),
        },
        "locked_design": {
            "long_arm": "Open < prior Close; buy limit Open - 0.25 * lagged ATR14; exit Close",
            "short_arm": "Open > prior Close; short limit Open + 0.75 * lagged ATR14; exit Close",
            "atr": "simple mean of 14 daily true ranges, shifted one session",
            "primary_endpoint": "per-arm top 3 by |gap|/ATR at open; 1/3 per slot; unfilled and unused slots zero; costs only fills",
            "primary_cost_bps": PRIMARY_COST_BPS,
            "cost_grid_bps": list(result.cost_grid_bps),
            "material_gap_thresholds_atr": list(MATERIAL_GAP_THRESHOLDS_ATR),
            "multiple_testing": "Holm across the two co-primary arms only",
        },
        "bootstrap_reps": result.bootstrap_reps,
        "outputs_sha256": output_hashes,
    }
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return target
