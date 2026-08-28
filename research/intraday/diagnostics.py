"""Deterministic diagnostics for fixed-clock intraday research.

These helpers evaluate pre-registered signals.  They do not select a template,
direction, threshold, cost case, ticker, sector, or capacity overlay.
"""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from trading_calendar import TRADING_DAY

from .templates import GAP_FIRST_HOUR_TEMPLATE_ID, INTRADAY_SHOCK_TEMPLATE_ID

DEFAULT_COST_GRID_BPS: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 20.0, 30.0)
PRIMARY_COST_BPS: Final[float] = 10.0
CAPACITY_SLOTS: Final[tuple[int, ...]] = (1, 3, 5, 10)
PRIMARY_TEMPLATE_IDS: Final[tuple[str, ...]] = (
    GAP_FIRST_HOUR_TEMPLATE_ID,
    INTRADAY_SHOCK_TEMPLATE_ID,
)


def validate_cost_grid(cost_grid_bps: tuple[float, ...]) -> tuple[float, ...]:
    """Return a sorted, unique finite grid that contains the primary case."""

    values = tuple(sorted({float(value) for value in cost_grid_bps}))
    if not values:
        raise ValueError("cost grid cannot be empty")
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("cost grid must contain finite, non-negative bps values")
    if PRIMARY_COST_BPS not in values:
        raise ValueError(f"cost grid must contain primary {PRIMARY_COST_BPS:g} bps")
    return values


def materialize_cost_grid(
    trades: pd.DataFrame,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
) -> pd.DataFrame:
    """Expand executed gross returns across pre-declared round-trip costs."""

    costs = validate_cost_grid(cost_grid_bps)
    if trades.empty:
        columns = list(trades.columns)
        for column in ("cost_bps", "net_return"):
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)
    if "gross_return" not in trades.columns:
        raise ValueError("trades must contain gross_return")
    frames: list[pd.DataFrame] = []
    for cost_bps in costs:
        frame = trades.drop(columns=["net_return", "round_trip_cost_bps"], errors="ignore").copy()
        frame["cost_bps"] = cost_bps
        frame["net_return"] = frame["gross_return"] - cost_bps / 10_000.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["cost_bps", "trade_date", "template_id", "ticker"],
        ignore_index=True,
    )


def make_daily_returns(cost_grid_trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse concurrent signals into one equal-notional observation per day."""

    columns = [
        "template_id",
        "cost_bps",
        "trade_date",
        "n_trades",
        "daily_equal_notional_return",
    ]
    if cost_grid_trades.empty:
        return pd.DataFrame(columns=columns)
    daily = (
        cost_grid_trades.groupby(
            ["template_id", "cost_bps", "trade_date"], sort=True, observed=True
        )["net_return"]
        .agg(n_trades="size", daily_equal_notional_return="mean")
        .reset_index()
    )
    return daily[columns]


def _stable_seed(key: str) -> int:
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")


def _cluster_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    key: str,
    reps: int,
) -> tuple[float, float]:
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
    batch_size = min(128, reps)
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, len(clean), size=(batch, len(clean)))
        means[cursor : cursor + batch] = clean[indices].mean(axis=1)
        cursor += batch
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _holm_adjust_primary(
    rows: pd.DataFrame,
    primary_template_ids: tuple[str, ...] = PRIMARY_TEMPLATE_IDS,
) -> pd.DataFrame:
    output = rows.copy()
    output["holm_p_value_primary"] = np.nan
    primary = output.loc[
        output["cost_bps"].eq(PRIMARY_COST_BPS)
        & output["template_id"].isin(primary_template_ids)
        & output["p_value_two_sided"].notna()
    ].sort_values(["p_value_two_sided", "template_id"])
    if primary.empty:
        return output
    # The family always contains both pre-registered templates.  A template
    # with no testable observations does not make the surviving test cheaper.
    family_size = len(primary_template_ids)
    running = 0.0
    for rank, (index, row) in enumerate(primary.iterrows()):
        adjusted = min(1.0, (family_size - rank) * float(row["p_value_two_sided"]))
        running = max(running, adjusted)
        output.loc[index, "holm_p_value_primary"] = running
    return output


def day_cluster_statistics(
    daily_returns: pd.DataFrame,
    *,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    bootstrap_reps: int = 2_000,
    primary_template_ids: tuple[str, ...] = PRIMARY_TEMPLATE_IDS,
) -> pd.DataFrame:
    """Compute day-cluster t tests and deterministic cluster-bootstrap CIs."""

    costs = validate_cost_grid(cost_grid_bps)
    rows: list[dict[str, float | int | str]] = []
    if not primary_template_ids or len(set(primary_template_ids)) != len(
        primary_template_ids
    ):
        raise ValueError("primary_template_ids must be non-empty and unique")
    for template_id in primary_template_ids:
        for cost_bps in costs:
            group = daily_returns.loc[
                daily_returns["template_id"].eq(template_id)
                & daily_returns["cost_bps"].eq(cost_bps),
                "daily_equal_notional_return",
            ].dropna()
            values = group.to_numpy(dtype=float)
            n_days = len(values)
            mean_return = float(np.mean(values)) if n_days else np.nan
            median_return = float(np.median(values)) if n_days else np.nan
            std_return = float(np.std(values, ddof=1)) if n_days > 1 else np.nan
            if n_days > 1 and np.isfinite(std_return) and std_return > 0:
                t_stat = mean_return / (std_return / sqrt(n_days))
                p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n_days - 1))
                active_day_sharpe = mean_return / std_return * sqrt(252.0)
            elif n_days > 1 and std_return == 0 and mean_return != 0:
                t_stat = float(np.sign(mean_return) * np.inf)
                p_value = 0.0
                active_day_sharpe = float(np.sign(mean_return) * np.inf)
            else:
                t_stat = np.nan
                p_value = np.nan
                active_day_sharpe = np.nan
            ci_low, ci_high = _cluster_bootstrap_mean_ci(
                values,
                key=f"{template_id}|{cost_bps:g}",
                reps=bootstrap_reps,
            )
            rows.append(
                {
                    "template_id": template_id,
                    "cost_bps": cost_bps,
                    "primary_cost_case": cost_bps == PRIMARY_COST_BPS,
                    "n_days": n_days,
                    "mean_daily_return": mean_return,
                    "median_daily_return": median_return,
                    "active_day_sharpe_annualized": active_day_sharpe,
                    "t_stat_day_cluster": t_stat,
                    "p_value_two_sided": p_value,
                    "bootstrap_mean_ci_2_5": ci_low,
                    "bootstrap_mean_ci_97_5": ci_high,
                    "bootstrap_reps": bootstrap_reps,
                }
            )
    return _holm_adjust_primary(pd.DataFrame(rows), primary_template_ids)


def annual_diagnostics(
    daily_returns: pd.DataFrame,
    full_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Report annual stability; no year is used to modify the rule."""

    columns = [
        "template_id",
        "cost_bps",
        "year",
        "n_full_sessions",
        "n_active_days",
        "n_trades",
        "mean_active_day_return",
        "win_rate_active_day",
        "compound_equal_notional_active_day_return",
    ]
    if daily_returns.empty:
        return pd.DataFrame(columns=columns)
    full_sessions = pd.DatetimeIndex(full_sessions).normalize()
    rows: list[dict[str, float | int | str]] = []
    for (template_id, cost_bps, year), group in daily_returns.assign(
        year=pd.to_datetime(daily_returns["trade_date"]).dt.year
    ).groupby(["template_id", "cost_bps", "year"], sort=True, observed=True):
        returns = group["daily_equal_notional_return"].astype(float)
        rows.append(
            {
                "template_id": template_id,
                "cost_bps": float(cost_bps),
                "year": int(year),
                "n_full_sessions": int((full_sessions.year == year).sum()),
                "n_active_days": len(group),
                "n_trades": int(group["n_trades"].sum()),
                "mean_active_day_return": float(returns.mean()),
                "win_rate_active_day": float(returns.gt(0).mean()),
                "compound_equal_notional_active_day_return": float(
                    (1.0 + returns).prod() - 1.0
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def leave_one_year_out_diagnostics(
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Show whether a fixed-rule mean survives removal of each calendar year."""

    columns = [
        "template_id",
        "cost_bps",
        "omitted_year",
        "n_remaining_days",
        "full_sample_mean_return",
        "remaining_mean_return",
        "remaining_win_rate",
        "mean_sign_preserved",
    ]
    if daily_returns.empty:
        return pd.DataFrame(columns=columns)
    work = daily_returns.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    rows: list[dict[str, float | int | str | bool]] = []
    for (template_id, cost_bps), group in work.groupby(
        ["template_id", "cost_bps"], sort=True, observed=True
    ):
        full_mean = float(group["daily_equal_notional_return"].mean())
        for omitted_year in sorted(group["year"].unique()):
            remaining = group.loc[
                group["year"].ne(omitted_year), "daily_equal_notional_return"
            ].astype(float)
            remaining_mean = float(remaining.mean()) if len(remaining) else np.nan
            rows.append(
                {
                    "template_id": template_id,
                    "cost_bps": float(cost_bps),
                    "omitted_year": int(omitted_year),
                    "n_remaining_days": len(remaining),
                    "full_sample_mean_return": full_mean,
                    "remaining_mean_return": remaining_mean,
                    "remaining_win_rate": (
                        float(remaining.gt(0).mean()) if len(remaining) else np.nan
                    ),
                    "mean_sign_preserved": (
                        bool(np.sign(remaining_mean) == np.sign(full_mean))
                        if np.isfinite(remaining_mean) and not np.isclose(full_mean, 0.0)
                        else False
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def side_diagnostics(cost_grid_trades: pd.DataFrame) -> pd.DataFrame:
    """Report prespecified long/short economics without selecting a side."""

    columns = [
        "template_id",
        "cost_bps",
        "side",
        "direction",
        "n_trades",
        "n_days",
        "mean_trade_net_return",
        "mean_daily_equal_notional_return",
        "median_daily_equal_notional_return",
        "win_rate_daily",
    ]
    if cost_grid_trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, float | int | str]] = []
    grouped = cost_grid_trades.groupby(
        ["template_id", "cost_bps", "side"], sort=True, observed=True
    )
    for (template_id, cost_bps, side), group in grouped:
        daily = group.groupby("trade_date", observed=True)["net_return"].mean()
        rows.append(
            {
                "template_id": template_id,
                "cost_bps": float(cost_bps),
                "side": int(side),
                "direction": "long" if int(side) > 0 else "short",
                "n_trades": len(group),
                "n_days": int(group["trade_date"].nunique()),
                "mean_trade_net_return": float(group["net_return"].mean()),
                "mean_daily_equal_notional_return": float(daily.mean()),
                "median_daily_equal_notional_return": float(daily.median()),
                "win_rate_daily": float(daily.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def rolling_five_year_train_one_year_test(
    daily_returns: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Materialize fixed-rule five-calendar-year/train, one-year/test windows."""

    columns = [
        "template_id",
        "cost_bps",
        "test_year",
        "train_start_year",
        "train_end_year",
        "n_train_days",
        "n_test_days",
        "train_mean_return",
        "test_mean_return",
        "train_win_rate",
        "test_win_rate",
        "test_year_complete",
        "eligible_for_stability_gate",
        "rule_selected_or_refit",
    ]
    if daily_returns.empty:
        return pd.DataFrame(columns=columns)
    expected = pd.DatetimeIndex(expected_sessions).normalize().sort_values().unique()
    if expected.empty:
        raise ValueError("expected_sessions cannot be empty")
    work = daily_returns.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    calendar_years = range(int(expected.year.min()), int(expected.year.max()) + 1)
    complete_years = {
        year
        for year in calendar_years
        if pd.date_range(
            f"{year}-01-01", f"{year}-12-31", freq=TRADING_DAY
        ).normalize().isin(expected).all()
    }
    rows: list[dict[str, float | int | str | bool]] = []
    for (template_id, cost_bps), group in work.groupby(
        ["template_id", "cost_bps"], sort=True, observed=True
    ):
        min_year = int(expected.year.min())
        max_year = int(expected.year.max())
        for test_year in range(min_year + 5, max_year + 1):
            prior_years = set(range(test_year - 5, test_year))
            if not prior_years.issubset(complete_years):
                continue
            train = group.loc[group["year"].between(test_year - 5, test_year - 1)]
            test = group.loc[group["year"].eq(test_year)]
            if train.empty and test.empty:
                continue
            train_returns = train["daily_equal_notional_return"].astype(float)
            test_returns = test["daily_equal_notional_return"].astype(float)
            rows.append(
                {
                    "template_id": template_id,
                    "cost_bps": float(cost_bps),
                    "test_year": test_year,
                    "train_start_year": test_year - 5,
                    "train_end_year": test_year - 1,
                    "n_train_days": len(train),
                    "n_test_days": len(test),
                    "train_mean_return": (
                        float(train_returns.mean()) if len(train_returns) else np.nan
                    ),
                    "test_mean_return": (
                        float(test_returns.mean()) if len(test_returns) else np.nan
                    ),
                    "train_win_rate": (
                        float(train_returns.gt(0).mean())
                        if len(train_returns)
                        else np.nan
                    ),
                    "test_win_rate": (
                        float(test_returns.gt(0).mean()) if len(test_returns) else np.nan
                    ),
                    "test_year_complete": test_year in complete_years,
                    "eligible_for_stability_gate": (
                        test_year in complete_years and len(test_returns) > 0
                    ),
                    "rule_selected_or_refit": False,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.astype(float)).cumprod()
    if wealth.empty:
        return np.nan
    return float((wealth / wealth.cummax() - 1.0).min())


def capacity_overlays(
    cost_grid_trades: pd.DataFrame,
    full_sessions: pd.DatetimeIndex,
    *,
    slots: tuple[int, ...] = CAPACITY_SLOTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank by pre-existing strength and report K-slot equal-notional returns.

    Unused slots earn zero.  The overlays model scarce trade slots, not broker
    rules, shares, commissions, spreads, or dollar capacity.
    """

    daily_columns = [
        "template_id",
        "cost_bps",
        "trade_date",
        "capacity_slots",
        "slots_used",
        "slot_portfolio_return",
    ]
    summary_columns = [
        "template_id",
        "cost_bps",
        "capacity_slots",
        "n_full_sessions",
        "n_active_days",
        "mean_session_return",
        "annualized_return_arithmetic",
        "annualized_volatility",
        "annualized_sharpe",
        "compound_return",
        "max_drawdown",
    ]
    if any(slot < 1 for slot in slots):
        raise ValueError("capacity slots must be positive")
    if cost_grid_trades.empty:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=summary_columns)

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    grouped = cost_grid_trades.groupby(
        ["template_id", "cost_bps", "trade_date"], sort=True, observed=True
    )
    for (template_id, cost_bps, trade_date), group in grouped:
        ranked = group.sort_values(
            ["signal_strength", "ticker"], ascending=[False, True]
        )
        for slot_count in slots:
            selected = ranked.head(slot_count)
            rows.append(
                {
                    "template_id": template_id,
                    "cost_bps": float(cost_bps),
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    "capacity_slots": int(slot_count),
                    "slots_used": len(selected),
                    "slot_portfolio_return": float(
                        selected["net_return"].sum() / slot_count
                    ),
                }
            )
    daily = pd.DataFrame(rows, columns=daily_columns)
    full_index = pd.DatetimeIndex(full_sessions).normalize().sort_values().unique()
    summaries: list[dict[str, float | int | str]] = []
    for (template_id, cost_bps, slot_count), group in daily.groupby(
        ["template_id", "cost_bps", "capacity_slots"], sort=True, observed=True
    ):
        series = group.set_index("trade_date")["slot_portfolio_return"].reindex(
            full_index, fill_value=0.0
        )
        mean_return = float(series.mean())
        volatility = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        sharpe = (
            mean_return / volatility * sqrt(252.0)
            if np.isfinite(volatility) and volatility > 0
            else np.nan
        )
        summaries.append(
            {
                "template_id": template_id,
                "cost_bps": float(cost_bps),
                "capacity_slots": int(slot_count),
                "n_full_sessions": len(series),
                "n_active_days": int(group["trade_date"].nunique()),
                "mean_session_return": mean_return,
                "annualized_return_arithmetic": mean_return * 252.0,
                "annualized_volatility": volatility * sqrt(252.0),
                "annualized_sharpe": sharpe,
                "compound_return": float((1.0 + series).prod() - 1.0),
                "max_drawdown": _max_drawdown(series),
            }
        )
    return daily, pd.DataFrame(summaries, columns=summary_columns)


def concentration_summaries(
    primary_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ticker and sector concentration/stability summaries at 10 bps."""

    columns = [
        "template_id",
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
    if primary_trades.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()

    weighted = primary_trades.copy()
    day_counts = weighted.groupby(
        ["template_id", "trade_date"], observed=True
    )["ticker"].transform("size")
    weighted["endpoint_contribution"] = weighted["net_return"] / day_counts

    def summarize(group_column: str) -> pd.DataFrame:
        totals = weighted.groupby("template_id", observed=True).size()
        return_totals = weighted.groupby("template_id", observed=True)[
            "endpoint_contribution"
        ].sum()
        absolute_return_totals = (
            weighted.assign(
                absolute_endpoint_contribution=weighted["endpoint_contribution"].abs()
            )
            .groupby("template_id", observed=True)["absolute_endpoint_contribution"]
            .sum()
        )
        rows: list[dict[str, float | int | str]] = []
        for (template_id, label), group in weighted.groupby(
            ["template_id", group_column], sort=True, observed=True, dropna=False
        ):
            returns = group["net_return"].astype(float)
            contribution_sum = float(group["endpoint_contribution"].sum())
            absolute_sum = float(group["endpoint_contribution"].abs().sum())
            template_net_sum = float(return_totals[template_id])
            template_absolute_sum = float(absolute_return_totals[template_id])
            rows.append(
                {
                    "template_id": template_id,
                    "group": str(label),
                    "n_trades": len(group),
                    "n_days": int(pd.to_datetime(group["trade_date"]).nunique()),
                    "mean_net_return": float(returns.mean()),
                    "median_net_return": float(returns.median()),
                    "win_rate_net": float(returns.gt(0).mean()),
                    "share_of_template_trades": float(len(group) / totals[template_id]),
                    "endpoint_contribution_sum": contribution_sum,
                    "absolute_endpoint_contribution_sum": absolute_sum,
                    "share_of_template_endpoint_return": (
                        contribution_sum / template_net_sum
                        if not np.isclose(template_net_sum, 0.0)
                        else np.nan
                    ),
                    "share_of_template_absolute_endpoint_contribution": (
                        absolute_sum / template_absolute_sum
                        if not np.isclose(template_absolute_sum, 0.0)
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    return summarize("ticker"), summarize("sector")
