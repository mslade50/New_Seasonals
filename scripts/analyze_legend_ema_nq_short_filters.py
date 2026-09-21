"""Test point-in-time daily trend filters on Legend EMA futures shorts.

The futures archive is an unadjusted continuous contract.  Daily moving
averages on that raw series can be distorted by quarterly roll gaps, so the
primary analysis constructs a forward, additive roll-neutral close series.
On a contract-change session it removes the open-to-prior-close splice and
retains that session's open-to-close move.  A ratio-spliced series is emitted
as a robustness check.

Every filter is evaluated using the setup-session close and an indicator
whose window ends on that same setup session.  The trade occurs on the next
RTH session, so no entry-day information enters the filter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_legend_ema_futures import (
    BASE_VARIANT,
    DEFAULT_DATA_DIR,
    INSTRUMENTS,
    build_15_minute_bars,
    build_daily_sessions,
    load_symbol_minutes,
    summarize_trades,
)


DEFAULT_TRADES_PATH = (
    ROOT / "artifacts" / "databento" / "legend_ema_backtest" / "trades.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "databento"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "legend_ema_nq_short_filters"
SERIES_METHODS = ("additive", "ratio")
MA_WINDOWS = {"ema21": 21, "sma50": 50, "sma200": 200}
NEIGHBOR_SPANS = {
    "ema": (15, 18, 21, 24, 27, 30),
    "sma": (35, 40, 45, 50, 55, 60, 150, 175, 200),
}


def build_roll_neutral_daily(
    daily: pd.DataFrame,
    trusted_start: str = "2016-01-01",
) -> pd.DataFrame:
    """Return valid RTH closes with additive and ratio roll-neutral series.

    A normal session contributes close-to-prior-close movement.  A session
    whose instrument ID differs from the prior valid session contributes only
    its own open-to-close movement, neutralizing the unknowable mixture of
    contract basis and true overnight movement at the splice.
    """

    required = {
        "open",
        "close",
        "minute_count",
        "first_ts",
        "instrument_id",
        "instrument_count",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily data is missing required columns: {sorted(missing)}")

    data = daily.loc[pd.to_datetime(daily.index) >= pd.Timestamp(trusted_start)].copy()
    first_clock = data["first_ts"].map(lambda value: value.strftime("%H:%M"))
    finite_prices = np.isfinite(data["open"]) & np.isfinite(data["close"])
    valid = (
        finite_prices
        & data["open"].gt(0)
        & data["close"].gt(0)
        & data["instrument_count"].eq(1)
        & data["minute_count"].ge(100)
        & first_clock.eq("09:30")
    )
    data = data.loc[valid].sort_index()
    if data.empty:
        raise ValueError("No valid daily RTH sessions remained for the trend filter")

    contract_changed = data["instrument_id"].ne(data["instrument_id"].shift())
    prior_close = data["close"].shift()

    additive_move = data["close"] - prior_close
    additive_move = additive_move.where(~contract_changed, data["close"] - data["open"])
    additive_move.iloc[0] = 0.0
    data["additive_close"] = float(data["close"].iloc[0]) + additive_move.cumsum()

    ratio_move = data["close"] / prior_close
    ratio_move = ratio_move.where(~contract_changed, data["close"] / data["open"])
    ratio_move.iloc[0] = 1.0
    data["ratio_close"] = 100.0 * ratio_move.cumprod()
    data["contract_changed"] = contract_changed

    for method in SERIES_METHODS:
        close = data[f"{method}_close"]
        data[f"{method}_ema21"] = close.ewm(
            span=21,
            adjust=False,
            min_periods=21,
        ).mean()
        data[f"{method}_sma50"] = close.rolling(50, min_periods=50).mean()
        data[f"{method}_sma200"] = close.rolling(200, min_periods=200).mean()

    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    data.index.name = "setup_date"
    return data


def filter_masks(data: pd.DataFrame, method: str) -> dict[str, pd.Series]:
    """Build mutually exclusive single-MA branches and simple combinations."""

    close = data[f"{method}_close"]
    above = {name: close.gt(data[f"{method}_{name}"]) for name in MA_WINDOWS}
    below = {name: close.lt(data[f"{method}_{name}"]) for name in MA_WINDOWS}
    return {
        "above_ema21": above["ema21"],
        "below_ema21": below["ema21"],
        "above_sma50": above["sma50"],
        "below_sma50": below["sma50"],
        "above_sma200": above["sma200"],
        "below_sma200": below["sma200"],
        "above_ema21_sma50": above["ema21"] & above["sma50"],
        "below_ema21_sma50": below["ema21"] & below["sma50"],
        "above_ema21_sma200": above["ema21"] & above["sma200"],
        "below_ema21_sma200": below["ema21"] & below["sma200"],
        "above_all_three": above["ema21"] & above["sma50"] & above["sma200"],
        "below_all_three": below["ema21"] & below["sma50"] & below["sma200"],
    }


def load_symbol_trades(trades_path: Path, symbol: str) -> pd.DataFrame:
    if symbol not in INSTRUMENTS:
        raise ValueError(f"Unsupported futures root: {symbol}")
    trades = pd.read_csv(trades_path)
    required = {
        "symbol",
        "variant",
        "direction",
        "setup_date",
        "entry_date",
        "entry_ts",
        "pnl_dollars",
        "return_bps",
    }
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Trade file is missing required columns: {sorted(missing)}")

    trades = trades.loc[
        trades["symbol"].eq(symbol)
        & trades["variant"].eq(BASE_VARIANT)
    ].copy()
    trades["setup_date"] = pd.to_datetime(trades["setup_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trusted_start = pd.Timestamp(INSTRUMENTS[symbol].trusted_start)
    trades = trades.loc[trades["entry_date"].ge(trusted_start)]
    return trades.sort_values("entry_ts").reset_index(drop=True)


def load_nq_trades(trades_path: Path) -> pd.DataFrame:
    """Backward-compatible convenience wrapper for the original NQ report."""

    return load_symbol_trades(trades_path, "NQ")


def load_nq_shorts(trades_path: Path) -> pd.DataFrame:
    return load_nq_trades(trades_path).loc[lambda frame: frame["direction"].eq("short")].copy()


def _safe_number(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary_row(
    trades: pd.DataFrame,
    eligible: pd.DataFrame,
    method: str,
    period: str,
    filter_name: str,
) -> dict[str, object]:
    summary = {key: _safe_number(value) for key, value in summarize_trades(trades).items()}
    baseline_pnl = float(eligible["pnl_dollars"].sum())
    summary.update(
        {
            "series_method": method,
            "period": period,
            "filter": filter_name,
            "eligible_trades": int(len(eligible)),
            "retained_pct": float(len(trades) / len(eligible) * 100) if len(eligible) else None,
            "baseline_pnl_dollars": baseline_pnl,
            "pnl_share_pct": (
                float(trades["pnl_dollars"].sum() / baseline_pnl * 100)
                if baseline_pnl
                else None
            ),
        }
    )
    return summary


def analyze_filters(trades: pd.DataFrame, trend: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all branches on a common SMA200-ready sample and by era."""

    merged = trades.merge(
        trend.reset_index(),
        on="setup_date",
        how="left",
        validate="many_to_one",
    )
    # Common support makes every full-sample filter directly comparable.
    common = merged.loc[
        merged[[f"{method}_sma200" for method in SERIES_METHODS]].notna().all(axis=1)
    ].copy()

    period_masks: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "full_common": lambda frame: pd.Series(True, index=frame.index),
        "train_2016_2020": lambda frame: frame["entry_date"].dt.year.le(2020),
        "holdout_2021_present": lambda frame: frame["entry_date"].dt.year.ge(2021),
        "recent_2022_present": lambda frame: frame["entry_date"].dt.year.ge(2022),
    }
    rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for method in SERIES_METHODS:
        masks = filter_masks(common, method)
        masks = {"all": pd.Series(True, index=common.index), **masks}
        for period, period_mask_fn in period_masks.items():
            period_mask = period_mask_fn(common)
            eligible = common.loc[period_mask]
            for filter_name, mask in masks.items():
                selected = common.loc[period_mask & mask.fillna(False)]
                rows.append(
                    _summary_row(selected, eligible, method, period, filter_name)
                )

        years = sorted(common["entry_date"].dt.year.unique())
        for year in years:
            year_mask = common["entry_date"].dt.year.eq(year)
            eligible = common.loc[year_mask]
            for filter_name, mask in masks.items():
                selected = common.loc[year_mask & mask.fillna(False)]
                yearly_rows.append(
                    _summary_row(selected, eligible, method, str(year), filter_name)
                )

    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


def build_branch_selection(results: pd.DataFrame) -> pd.DataFrame:
    """Select above/below using pre-2021 expectancy, then show later results."""

    rows: list[dict[str, object]] = []
    for method in SERIES_METHODS:
        method_rows = results.loc[results["series_method"].eq(method)]
        for indicator in MA_WINDOWS:
            candidates = [f"above_{indicator}", f"below_{indicator}"]
            train = method_rows.loc[
                method_rows["period"].eq("train_2016_2020")
                & method_rows["filter"].isin(candidates)
            ].copy()
            train = train.sort_values(
                ["avg_return_bps", "trades"], ascending=[False, False]
            )
            if train.empty:
                continue
            chosen = str(train.iloc[0]["filter"])
            for period in ("train_2016_2020", "holdout_2021_present", "recent_2022_present"):
                result = method_rows.loc[
                    method_rows["period"].eq(period)
                    & method_rows["filter"].eq(chosen)
                ]
                if result.empty:
                    continue
                row = result.iloc[0].to_dict()
                row["indicator"] = indicator
                row["selected_branch"] = chosen
                rows.append(row)
    return pd.DataFrame(rows)


def build_classification_agreement(trades: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    merged = trades.merge(trend.reset_index(), on="setup_date", how="left")
    merged = merged.loc[
        merged[[f"{method}_sma200" for method in SERIES_METHODS]].notna().all(axis=1)
    ].copy()
    additive = filter_masks(merged, "additive")
    ratio = filter_masks(merged, "ratio")
    rows = []
    for name in additive:
        comparable = additive[name].notna() & ratio[name].notna()
        rows.append(
            {
                "filter": name,
                "trades_compared": int(comparable.sum()),
                "agreement_pct": float(
                    additive[name].loc[comparable].eq(ratio[name].loc[comparable]).mean()
                    * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def _common_short_sample(
    trades: pd.DataFrame,
    trend: pd.DataFrame,
    method: str = "additive",
) -> pd.DataFrame:
    columns = [
        f"{method}_close",
        f"{method}_ema21",
        f"{method}_sma50",
        f"{method}_sma200",
    ]
    merged = trades.merge(
        trend[columns].reset_index(),
        on="setup_date",
        how="left",
        validate="many_to_one",
    )
    return merged.loc[merged[f"{method}_sma200"].notna()].copy()


def analyze_neighbor_spans(
    trades: pd.DataFrame,
    trend: pd.DataFrame,
    method: str = "additive",
) -> pd.DataFrame:
    """Check whether an exact MA result survives nearby window choices."""

    common = _common_short_sample(trades, trend, method)
    close_column = f"{method}_close"
    daily_close = trend[close_column]
    rows: list[dict[str, object]] = []
    for family, spans in NEIGHBOR_SPANS.items():
        for span in spans:
            if family == "ema":
                daily_ma = daily_close.ewm(
                    span=span,
                    adjust=False,
                    min_periods=span,
                ).mean()
            else:
                daily_ma = daily_close.rolling(span, min_periods=span).mean()
            ma_at_setup = common["setup_date"].map(daily_ma)
            selected = common.loc[common[close_column].lt(ma_at_setup)].copy()
            row = _summary_row(
                selected,
                common,
                method,
                "full_common",
                f"below_{family}{span}",
            )

            winners = selected.loc[selected["pnl_dollars"].gt(0), "pnl_dollars"].sort_values(
                ascending=False
            )
            net_pnl = float(selected["pnl_dollars"].sum())
            gross_wins = float(winners.sum())
            row.update(
                {
                    "family": family,
                    "span": span,
                    "active_years": int(selected["entry_date"].dt.year.nunique()),
                    "profitable_years": int(
                        selected.groupby(selected["entry_date"].dt.year)["pnl_dollars"]
                        .sum()
                        .gt(0)
                        .sum()
                    ),
                    "top1_share_of_net_pct": (
                        float(winners.head(1).sum() / net_pnl * 100) if net_pnl else None
                    ),
                    "top3_share_of_net_pct": (
                        float(winners.head(3).sum() / net_pnl * 100) if net_pnl else None
                    ),
                    "top1_share_of_gross_wins_pct": (
                        float(winners.head(1).sum() / gross_wins * 100)
                        if gross_wins
                        else None
                    ),
                    "top3_share_of_gross_wins_pct": (
                        float(winners.head(3).sum() / gross_wins * 100)
                        if gross_wins
                        else None
                    ),
                }
            )

            loyo = []
            for year in sorted(selected["entry_date"].dt.year.unique()):
                remaining = selected.loc[selected["entry_date"].dt.year.ne(year)]
                loyo.append((int(year), summarize_trades(remaining)))
            if loyo:
                worst_year, worst_values = min(
                    loyo,
                    key=lambda item: float(item[1]["total_pnl_dollars"]),
                )
                finite_pfs = [
                    float(values["profit_factor"])
                    for _, values in loyo
                    if values.get("profit_factor") is not None
                    and math.isfinite(float(values["profit_factor"]))
                ]
                row.update(
                    {
                        "worst_loyo_omitted_year": worst_year,
                        "worst_loyo_pnl_dollars": float(
                            worst_values["total_pnl_dollars"]
                        ),
                        "worst_loyo_profit_factor": min(finite_pfs) if finite_pfs else None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_combined_symbol(
    all_symbol_trades: pd.DataFrame,
    short_trades: pd.DataFrame,
    trend: pd.DataFrame,
    method: str = "additive",
) -> pd.DataFrame:
    """Keep every trusted long and apply selected gates only to shorts."""

    longs = all_symbol_trades.loc[all_symbol_trades["direction"].eq("long")].copy()
    common_shorts = _common_short_sample(short_trades, trend, method)
    masks = filter_masks(common_shorts, method)
    configurations = {
        "all_longs_all_common_shorts": pd.Series(True, index=common_shorts.index),
        "all_longs_no_shorts": pd.Series(False, index=common_shorts.index),
        "all_longs_below_ema21_shorts": masks["below_ema21"],
        "all_longs_above_ema21_shorts": masks["above_ema21"],
        "all_longs_below_sma50_shorts": masks["below_sma50"],
        "all_longs_above_sma50_shorts": masks["above_sma50"],
        "all_longs_below_sma200_shorts": masks["below_sma200"],
        "all_longs_above_sma200_shorts": masks["above_sma200"],
    }
    rows = []
    baseline_pnl = float(
        pd.concat([longs, common_shorts], ignore_index=True)["pnl_dollars"].sum()
    )
    for name, mask in configurations.items():
        selected_shorts = common_shorts.loc[mask.fillna(False)]
        combined = pd.concat([longs, selected_shorts], ignore_index=True)
        summary = {
            key: _safe_number(value) for key, value in summarize_trades(combined).items()
        }
        summary.update(
            {
                "configuration": name,
                "long_trades": int(len(longs)),
                "short_trades": int(len(selected_shorts)),
                "baseline_pnl_dollars": baseline_pnl,
                "pnl_retention_pct": (
                    float(summary["total_pnl_dollars"] / baseline_pnl * 100)
                    if baseline_pnl
                    else None
                ),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def analyze_combined_nq(
    all_nq_trades: pd.DataFrame,
    short_trades: pd.DataFrame,
    trend: pd.DataFrame,
    method: str = "additive",
) -> pd.DataFrame:
    """Backward-compatible wrapper for callers of the original NQ analysis."""

    return analyze_combined_symbol(all_nq_trades, short_trades, trend, method)


def _format_metric(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    symbol: str,
    trades: pd.DataFrame,
    trend: pd.DataFrame,
    results: pd.DataFrame,
    yearly: pd.DataFrame,
    selection: pd.DataFrame,
    agreement: pd.DataFrame,
    full_session_results: pd.DataFrame,
    full_session_yearly: pd.DataFrame,
    full_session_trend: pd.DataFrame,
    neighbor_spans: pd.DataFrame,
    combined_symbol: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "filter_results.csv", index=False)
    yearly.to_csv(output_dir / "yearly.csv", index=False)
    selection.to_csv(output_dir / "train_selected_holdout.csv", index=False)
    agreement.to_csv(output_dir / "roll_method_agreement.csv", index=False)
    trend.reset_index().to_parquet(output_dir / "daily_trend_series.parquet", index=False)
    full_session_results.to_csv(
        output_dir / "full_session_only_filter_results.csv", index=False
    )
    full_session_yearly.to_csv(
        output_dir / "full_session_only_yearly.csv", index=False
    )
    full_session_trend.reset_index().to_parquet(
        output_dir / "full_session_only_daily_trend_series.parquet", index=False
    )
    neighbor_spans.to_csv(output_dir / "neighbor_spans.csv", index=False)
    combined_symbol.to_csv(
        output_dir / f"combined_{symbol.lower()}.csv", index=False
    )

    primary = results.loc[
        results["series_method"].eq("additive")
        & results["period"].eq("full_common")
        & results["filter"].isin(
            [
                "all",
                "above_ema21",
                "below_ema21",
                "above_sma50",
                "below_sma50",
                "above_sma200",
                "below_sma200",
                "above_all_three",
                "below_all_three",
            ]
        )
    ].copy()
    order = {
        name: index
        for index, name in enumerate(
            [
                "all",
                "above_ema21",
                "below_ema21",
                "above_sma50",
                "below_sma50",
                "above_sma200",
                "below_sma200",
                "above_all_three",
                "below_all_three",
            ]
        )
    }
    primary["_order"] = primary["filter"].map(order)
    primary = primary.sort_values("_order")
    display = pd.DataFrame(
        {
            "Filter": primary["filter"],
            "Trades": primary["trades"].astype(int),
            "Win %": primary["win_rate_pct"].map(_format_metric),
            "PF": primary["profit_factor"].map(lambda value: _format_metric(value, 2)),
            "Avg bps": primary["avg_return_bps"].map(_format_metric),
            "PnL $": primary["total_pnl_dollars"].map(lambda value: _format_metric(value, 0)),
            "Max DD $": primary["max_drawdown_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
        }
    )

    sensitivity_filters = ["below_ema21", "below_sma50", "below_sma200"]
    primary_sensitivity = results.loc[
        results["series_method"].eq("additive")
        & results["period"].eq("full_common")
        & results["filter"].isin(sensitivity_filters)
    ].copy()
    primary_sensitivity["Session policy"] = "All valid futures sessions"
    strict_sensitivity = full_session_results.loc[
        full_session_results["series_method"].eq("additive")
        & full_session_results["period"].eq("full_common")
        & full_session_results["filter"].isin(sensitivity_filters)
    ].copy()
    strict_sensitivity["Session policy"] = "Full 09:30-16:00 only"
    session_sensitivity = pd.concat(
        [primary_sensitivity, strict_sensitivity], ignore_index=True
    )
    session_sensitivity.to_csv(
        output_dir / "session_policy_sensitivity.csv", index=False
    )
    session_display = pd.DataFrame(
        {
            "Session policy": session_sensitivity["Session policy"],
            "Filter": session_sensitivity["filter"],
            "Trades": session_sensitivity["trades"].astype(int),
            "PF": session_sensitivity["profit_factor"].map(
                lambda value: _format_metric(value, 2)
            ),
            "PnL $": session_sensitivity["total_pnl_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
            "Max DD $": session_sensitivity["max_drawdown_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
        }
    )
    neighbor_display_source = neighbor_spans.loc[
        neighbor_spans["span"].isin([15, 18, 21, 24, 27, 30, 35, 40, 45, 50, 55, 60, 200])
    ].copy()
    neighbor_display = pd.DataFrame(
        {
            "Filter": neighbor_display_source["filter"],
            "Trades": neighbor_display_source["trades"].astype(int),
            "PF": neighbor_display_source["profit_factor"].map(
                lambda value: _format_metric(value, 2)
            ),
            "PnL $": neighbor_display_source["total_pnl_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
            "Avg bps": neighbor_display_source["avg_return_bps"].map(_format_metric),
            "Max DD $": neighbor_display_source["max_drawdown_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
        }
    )
    combined_display = pd.DataFrame(
        {
            "Configuration": combined_symbol["configuration"],
            "Trades": combined_symbol["trades"].astype(int),
            "PF": combined_symbol["profit_factor"].map(
                lambda value: _format_metric(value, 2)
            ),
            "PnL $": combined_symbol["total_pnl_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
            "Max DD $": combined_symbol["max_drawdown_dollars"].map(
                lambda value: _format_metric(value, 0)
            ),
            "PnL retained %": combined_symbol["pnl_retention_pct"].map(_format_metric),
        }
    )

    common_count = int(
        results.loc[
            results["series_method"].eq("additive")
            & results["period"].eq("full_common")
            & results["filter"].eq("all"),
            "trades",
        ].iloc[0]
    )
    full_count = int(len(trades))
    mean_agreement = float(agreement["agreement_pct"].mean())
    payload = {
        "methodology": {
            "trade_variant": BASE_VARIANT,
            "direction": "short",
            "signal_timestamp": "setup-session RTH close",
            "execution_timestamp": "next-session executable entry",
            "primary_roll_method": "additive splice neutralization",
            "robustness_roll_method": "ratio splice neutralization",
            "common_sample_requires": "both roll methods have SMA200 history",
        },
        "sample": {
            f"trusted_{symbol.lower()}_shorts": full_count,
            "common_sma200_ready_shorts": common_count,
            "warmup_exclusions": full_count - common_count,
            "first_daily_session": trend.index.min().date().isoformat(),
            "last_daily_session": trend.index.max().date().isoformat(),
            "roll_method_mean_filter_agreement_pct": mean_agreement,
        },
        "full_common_additive": primary.drop(columns="_order").to_dict("records"),
        "full_session_only_sensitivity": session_sensitivity.to_dict("records"),
        "neighbor_spans": neighbor_spans.to_dict("records"),
        f"combined_{symbol.lower()}": combined_symbol.to_dict("records"),
        "train_selected_holdout": selection.to_dict("records"),
        "roll_method_agreement": agreement.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_safe_number)

    full_short_pnl = float(trades["pnl_dollars"].sum())
    common_short_pnl = float(
        results.loc[
            results["series_method"].eq("additive")
            & results["period"].eq("full_common")
            & results["filter"].eq("all"),
            "total_pnl_dollars",
        ].iloc[0]
    )
    long_count = int(combined_symbol["long_trades"].iloc[0])
    warmup_pnl = full_short_pnl - common_short_pnl
    readme = f"""# {symbol} short daily-trend filter test

This is a post-hoc filter analysis of the `{BASE_VARIANT}` {symbol} shorts.  Each
condition uses the setup day's RTH close and a daily indicator ending on that
close; entry is on the following session.  The primary daily series removes
quarterly futures roll splices additively.  A ratio-spliced reconstruction is
included as a robustness check.

The common table contains {common_count} of {full_count} trusted {symbol} shorts;
{full_count - common_count} are excluded only because SMA200 needs a clean
200-session warmup.  Average additive/ratio filter classification agreement
is {mean_agreement:.1f}%.

{_markdown_table(display)}

## Daily-session convention sensitivity

The primary series includes legitimate shortened CME holiday sessions.  The
second policy excludes them and is deliberately reported as a sensitivity,
not as an alternate rule chosen after seeing results.

{_markdown_table(session_display)}

## Neighboring moving-average spans

All rows keep shorts only when the setup-day close is below the named average.
This is a stability check, not a search for a new best parameter.

{_markdown_table(neighbor_display)}

## Combined {symbol}, longs unchanged

The apples-to-apples baseline keeps all {long_count} trusted longs and the
{common_count} shorts with an available SMA200 history.  The
{full_count - common_count} warmup shorts omitted from every row netted
{warmup_pnl:+.0f} dollars.

{_markdown_table(combined_display)}

`filter_results.csv` contains full, pre-2021 training, 2021+ holdout, and
2022+ results for both splice methods.  `train_selected_holdout.csv` chooses
above versus below separately for each MA using only pre-2021 average return,
then carries that choice unchanged into the later samples.  Treat that split
as an overfitting diagnostic, not a true untouched out-of-sample test, because
the idea itself was proposed after seeing the strategy results.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=sorted(INSTRUMENTS), default="NQ")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"legend_ema_{symbol.lower()}_short_filters"
    )
    all_symbol_trades = load_symbol_trades(args.trades_path, symbol)
    trades = all_symbol_trades.loc[
        all_symbol_trades["direction"].eq("short")
    ].copy()
    minutes = load_symbol_minutes(args.data_dir, INSTRUMENTS[symbol].symbol)
    bars15 = build_15_minute_bars(minutes)
    daily = build_daily_sessions(minutes, bars15)
    trend = build_roll_neutral_daily(daily, INSTRUMENTS[symbol].trusted_start)
    results, yearly = analyze_filters(trades, trend)
    selection = build_branch_selection(results)
    agreement = build_classification_agreement(trades, trend)
    full_session_trend = build_roll_neutral_daily(
        daily.loc[daily["complete_rth"].fillna(False)],
        INSTRUMENTS[symbol].trusted_start,
    )
    full_session_results, full_session_yearly = analyze_filters(
        trades, full_session_trend
    )
    neighbor_spans = analyze_neighbor_spans(trades, trend)
    combined_symbol = analyze_combined_symbol(all_symbol_trades, trades, trend)
    write_report(
        output_dir,
        symbol,
        trades,
        trend,
        results,
        yearly,
        selection,
        agreement,
        full_session_results,
        full_session_yearly,
        full_session_trend,
        neighbor_spans,
        combined_symbol,
    )
    baseline = results.loc[
        results["series_method"].eq("additive")
        & results["period"].eq("full_common")
        & results["filter"].eq("all")
    ].iloc[0]
    print(
        f"Wrote {output_dir} from {int(baseline['trades'])} common-sample "
        f"{symbol} short trades"
    )


if __name__ == "__main__":
    main()
