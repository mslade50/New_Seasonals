"""Independent replication of the Raschke / MrMilk next-day EMA setup.

The script intentionally keeps the research path separate from the production
strategy engine.  It consumes regular-session 15-minute parquet bars with the
schema ``ts, open, high, low, close, volume`` and writes auditable CSV/JSON/
Markdown outputs.

Base rule
---------
* EMA(20) on RTH closes, carried continuously across sessions.
* Prior full session: abs(close-open)/(high-low) >= 0.75 and no bar range
  intersects that bar's finalized EMA.
* Next full session: enter at the 09:30 open toward the prior completed EMA.
* During each bar the exit limit is the EMA known before that bar opened.
* If the limit is not reached, exit at the final bar's close.

This is research-only and never stages or places an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "volume")
EXPECTED_TIMES = tuple(pd.date_range("2000-01-01 09:30", periods=26, freq="15min").time)


@dataclass(frozen=True)
class BacktestConfig:
    ema_span: int = 20
    warmup_bars: int = 200
    trend_threshold: float = 0.75
    touch_mode: str = "same_bar"  # same_bar | prior_bar | close_only
    target_mode: str = "rolling"  # rolling | fixed
    stop_atr_mult: float | None = None
    friction_bps: float = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intraday(path: Path, ema_span: int = 20) -> tuple[pd.DataFrame, dict]:
    """Load, validate, RTH-filter, and add causal EMA columns."""
    raw = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    duplicate_count = int(pd.to_datetime(raw["ts"]).duplicated().sum())
    null_count = int(raw[list(REQUIRED_COLUMNS)].isna().sum().sum())
    bars = raw.loc[:, REQUIRED_COLUMNS].copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    if bars["ts"].dt.tz is not None:
        bars["ts"] = bars["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars = bars.sort_values("ts").drop_duplicates("ts", keep="last")
    in_rth = (bars["ts"].dt.time >= EXPECTED_TIMES[0]) & (
        bars["ts"].dt.time <= EXPECTED_TIMES[-1]
    )
    bars = bars.loc[in_rth].reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if bars[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError(f"{path} has non-numeric/null OHLCV values after parsing")
    if (
        (bars["low"] > bars["high"])
        | (bars["high"] < bars[["open", "close"]].max(axis=1))
        | (bars["low"] > bars[["open", "close"]].min(axis=1))
        | (bars["open"] <= 0)
        | (bars["close"] <= 0)
    ).any():
        raise ValueError(f"{path} has invalid OHLC values")

    bars["session"] = bars["ts"].dt.normalize()
    bars["ema"] = bars["close"].ewm(span=ema_span, adjust=False).mean()
    bars["ema_prev"] = bars["ema"].shift(1)

    session_counts = bars.groupby("session", sort=True).size()
    full_sessions = 0
    for _, group in bars.groupby("session", sort=True):
        if is_full_session(group):
            full_sessions += 1
    qc = {
        "source": str(path.resolve()),
        "rows_raw": len(raw),
        "rows_rth_deduped": len(bars),
        "first_ts": bars["ts"].min().isoformat(),
        "last_ts": bars["ts"].max().isoformat(),
        "sessions": len(session_counts),
        "full_26_bar_sessions": int(full_sessions),
        "partial_sessions": int(len(session_counts) - full_sessions),
        "duplicate_timestamps_raw": duplicate_count,
        "null_cells_raw": null_count,
        "min_bars_per_session": int(session_counts.min()),
        "max_bars_per_session": int(session_counts.max()),
        "sha256": sha256_file(path),
    }
    return bars, qc


def is_full_session(group: pd.DataFrame) -> bool:
    times = tuple(pd.to_datetime(group["ts"]).dt.time)
    return len(group) == 26 and times == EXPECTED_TIMES


def session_touch_flags(group: pd.DataFrame) -> dict[str, bool]:
    same_touch = bool(
        ((group["low"] <= group["ema"]) & (group["ema"] <= group["high"])).any()
    )
    prior_valid = group["ema_prev"].notna()
    prior_touch = bool(
        prior_valid.any()
        and (
            (group.loc[prior_valid, "low"] <= group.loc[prior_valid, "ema_prev"])
            & (group.loc[prior_valid, "ema_prev"] <= group.loc[prior_valid, "high"])
        ).any()
    )
    close_delta = group["close"] - group["ema"]
    close_only_no_touch = bool((close_delta > 0).all() or (close_delta < 0).all())
    return {
        "no_touch_same_bar": not same_touch,
        "no_touch_prior_bar": not prior_touch,
        "no_touch_close_only": close_only_no_touch,
    }


def build_daily(bars: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for session, group in bars.groupby("session", sort=True):
        group = group.sort_values("ts")
        day_range = float(group["high"].max() - group["low"].min())
        day_open = float(group.iloc[0]["open"])
        day_close = float(group.iloc[-1]["close"])
        flags = session_touch_flags(group)
        records.append(
            {
                "session": pd.Timestamp(session),
                "open": day_open,
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": day_close,
                "final_ema": float(group.iloc[-1]["ema"]),
                "n_bars": len(group),
                "full_session": bool(is_full_session(group)),
                "trend_ratio": abs(day_close - day_open) / day_range
                if day_range > 0
                else np.nan,
                **flags,
            }
        )
    daily = pd.DataFrame.from_records(records).set_index("session").sort_index()
    daily["bars_seen"] = daily["n_bars"].cumsum()
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return daily


def qualifies_setup(row: pd.Series, config: BacktestConfig) -> bool:
    if not bool(row["full_session"]):
        return False
    if int(row.get("bars_seen", config.warmup_bars)) < config.warmup_bars:
        return False
    if (
        not np.isfinite(row["trend_ratio"])
        or row["trend_ratio"] < config.trend_threshold
    ):
        return False
    touch_column = {
        "same_bar": "no_touch_same_bar",
        "prior_bar": "no_touch_prior_bar",
        "close_only": "no_touch_close_only",
    }.get(config.touch_mode)
    if touch_column is None:
        raise ValueError(f"Unknown touch mode: {config.touch_mode}")
    return bool(row[touch_column])


def limit_fill(row: pd.Series, target: float, side: int) -> float | None:
    """Fill a profit-taking limit, including opening-price improvement."""
    if side == 1:
        if float(row["open"]) >= target:
            return float(row["open"])
        if float(row["high"]) >= target:
            return float(target)
    else:
        if float(row["open"]) <= target:
            return float(row["open"])
        if float(row["low"]) <= target:
            return float(target)
    return None


def stop_fill(row: pd.Series, stop: float, side: int) -> float | None:
    """Fill a protective stop, including adverse opening gaps."""
    if side == 1:
        if float(row["open"]) <= stop:
            return float(row["open"])
        if float(row["low"]) <= stop:
            return float(stop)
    else:
        if float(row["open"]) >= stop:
            return float(row["open"])
        if float(row["high"]) >= stop:
            return float(stop)
    return None


def simulate_trade(
    trade_bars: pd.DataFrame,
    *,
    entry: float,
    side: int,
    initial_target: float,
    target_mode: str = "rolling",
    stop_distance: float | None = None,
) -> dict:
    """Walk one session without using the current bar's closing EMA."""
    if target_mode not in {"rolling", "fixed"}:
        raise ValueError(f"Unknown target mode: {target_mode}")
    protective_stop = None if stop_distance is None else entry - side * stop_distance
    ordered = trade_bars.sort_values("ts").reset_index(drop=True)

    for bar_index, row in ordered.iterrows():
        target = initial_target if target_mode == "fixed" else float(row["ema_prev"])
        open_price = float(row["open"])
        target_at_open = (side == 1 and open_price >= target) or (
            side == -1 and open_price <= target
        )
        stop_at_open = protective_stop is not None and (
            (side == 1 and open_price <= protective_stop)
            or (side == -1 and open_price >= protective_stop)
        )

        # Opening gaps have known chronology. If both orders are marketable at
        # the open, retain the conservative stop-first convention; otherwise a
        # marketable target cannot be displaced by a stop touched only later.
        if target_at_open and not stop_at_open:
            return {
                "exit": open_price,
                "exit_reason": "ema_target",
                "exit_bar": int(bar_index),
                "active_target": float(target),
            }
        if stop_at_open:
            return {
                "exit": open_price,
                "exit_reason": "stop",
                "exit_bar": int(bar_index),
                "active_target": float(target),
            }
        target_price = limit_fill(row, target, side)
        stop_price = (
            None if protective_stop is None else stop_fill(row, protective_stop, side)
        )
        if stop_price is not None:
            # A 15-minute OHLC bar cannot order two intrabar touches.  The stop
            # wins ties so the stop sensitivity cannot manufacture optimism.
            return {
                "exit": float(stop_price),
                "exit_reason": "stop",
                "exit_bar": int(bar_index),
                "active_target": float(target),
            }
        if target_price is not None:
            return {
                "exit": float(target_price),
                "exit_reason": "ema_target",
                "exit_bar": int(bar_index),
                "active_target": float(target),
            }

    final_row = ordered.iloc[-1]
    return {
        "exit": float(final_row["close"]),
        "exit_reason": "session_close",
        "exit_bar": int(len(ordered) - 1),
        "active_target": float(
            initial_target if target_mode == "fixed" else final_row["ema_prev"]
        ),
    }


def run_all_eligible_trades(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    session_groups: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Trade every eligible day and stamp whether its prior day was a setup."""
    sessions = list(daily.index)
    grouped = session_groups or {
        session: group.sort_values("ts")
        for session, group in bars.groupby("session", sort=True)
    }
    records: list[dict] = []
    for i in range(1, len(sessions)):
        signal_date = sessions[i - 1]
        trade_date = sessions[i]
        signal_day = daily.loc[signal_date]
        trade_day = daily.loc[trade_date]
        if not bool(signal_day["full_session"]) or not bool(trade_day["full_session"]):
            continue
        if int(signal_day["bars_seen"]) < config.warmup_bars:
            continue
        trade_bars = grouped[trade_date]
        entry = float(trade_bars.iloc[0]["open"])
        initial_target = float(signal_day["final_ema"])
        if not np.isfinite(initial_target) or math.isclose(
            entry, initial_target, rel_tol=0, abs_tol=1e-12
        ):
            continue
        side = 1 if entry < initial_target else -1
        atr = float(signal_day["atr14"]) if np.isfinite(signal_day["atr14"]) else np.nan
        stop_distance = None
        if config.stop_atr_mult is not None:
            if not np.isfinite(atr) or atr <= 0:
                continue
            stop_distance = config.stop_atr_mult * atr
        result = simulate_trade(
            trade_bars,
            entry=entry,
            side=side,
            initial_target=initial_target,
            target_mode=config.target_mode,
            stop_distance=stop_distance,
        )
        gross_return = side * (result["exit"] - entry) / entry
        gap_points = abs(entry - initial_target)
        records.append(
            {
                "ticker": ticker,
                "signal_date": pd.Timestamp(signal_date),
                "trade_date": pd.Timestamp(trade_date),
                "direction": "long" if side == 1 else "short",
                "side": side,
                "entry": entry,
                "exit": result["exit"],
                "exit_reason": result["exit_reason"],
                "exit_bar": result["exit_bar"],
                "holding_minutes_upper_bound": int((result["exit_bar"] + 1) * 15),
                "initial_ema_target": initial_target,
                "exit_active_target": result["active_target"],
                "gross_return": gross_return,
                "net_return": gross_return - config.friction_bps / 10_000,
                "gross_points_per_share": side * (result["exit"] - entry),
                "initial_gap_points": gap_points,
                "initial_gap_pct": gap_points / entry,
                "initial_gap_atr": gap_points / atr
                if np.isfinite(atr) and atr > 0
                else np.nan,
                "prior_atr14": atr,
                "prior_trend_ratio": float(signal_day["trend_ratio"]),
                "prior_no_touch_same_bar": bool(signal_day["no_touch_same_bar"]),
                "prior_no_touch_prior_bar": bool(signal_day["no_touch_prior_bar"]),
                "prior_no_touch_close_only": bool(signal_day["no_touch_close_only"]),
                "is_setup": qualifies_setup(signal_day, config),
            }
        )
    return pd.DataFrame.from_records(records)


def profit_factor(returns: pd.Series) -> float:
    positive = float(returns[returns > 0].sum())
    negative = float(-returns[returns < 0].sum())
    if negative == 0:
        return np.inf if positive > 0 else np.nan
    return positive / negative


def summarize_trades(trades: pd.DataFrame, *, label: str = "") -> dict:
    if trades.empty:
        return {"label": label, "n": 0}
    ordered = trades.sort_values("trade_date")
    returns = ordered["net_return"].astype(float)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    first = pd.Timestamp(ordered["trade_date"].min())
    last = pd.Timestamp(ordered["trade_date"].max())
    years = max((last - first).days / 365.25, 1 / 365.25)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    t_stat = (
        mean / (std / math.sqrt(len(returns)))
        if len(returns) > 1 and std > 0
        else np.nan
    )
    return {
        "label": label,
        "n": len(ordered),
        "first_trade": first.date().isoformat(),
        "last_trade": last.date().isoformat(),
        "trades_per_year": len(ordered) / years,
        "win_rate": float((returns > 0).mean()),
        "target_hit_rate": float((ordered["exit_reason"] == "ema_target").mean()),
        "avg_net_bps": mean * 10_000,
        "median_net_bps": float(returns.median() * 10_000),
        "profit_factor": profit_factor(returns),
        "arithmetic_net_return": float(returns.sum()),
        "compounded_net_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "worst_trade_bps": float(returns.min() * 10_000),
        "best_trade_bps": float(returns.max() * 10_000),
        "mean_t_stat_iid": t_stat,
        "avg_holding_minutes_upper_bound": float(
            ordered["holding_minutes_upper_bound"].mean()
        ),
    }


def match_controls(all_trades: pd.DataFrame) -> pd.DataFrame:
    """Match setup days to ordinary days on year, side, and gap/ATR."""
    setups = all_trades.loc[
        all_trades["is_setup"] & all_trades["initial_gap_atr"].notna()
    ].copy()
    controls = all_trades.loc[
        ~all_trades["is_setup"] & all_trades["initial_gap_atr"].notna()
    ].copy()
    setups["year"] = pd.to_datetime(setups["trade_date"]).dt.year
    controls["year"] = pd.to_datetime(controls["trade_date"]).dt.year
    used: set[int] = set()
    pairs: list[dict] = []

    # Match extreme gaps first because they have fewer close alternatives.
    for setup_index, setup in setups.sort_values(
        "initial_gap_atr", ascending=False
    ).iterrows():
        candidates = controls.loc[
            (controls["side"] == setup["side"])
            & (controls["year"] == setup["year"])
            & (~controls.index.isin(used))
        ]
        match_scope = "same_year"
        if candidates.empty:
            candidates = controls.loc[
                (controls["side"] == setup["side"])
                & ((controls["year"] - setup["year"]).abs() <= 1)
                & (~controls.index.isin(used))
            ]
            match_scope = "adjacent_year"
        if candidates.empty:
            continue
        distance = (candidates["initial_gap_atr"] - setup["initial_gap_atr"]).abs()
        control_index = int(distance.idxmin())
        control = controls.loc[control_index]
        used.add(control_index)
        pairs.append(
            {
                "ticker": setup["ticker"],
                "setup_trade_date": setup["trade_date"],
                "control_trade_date": control["trade_date"],
                "direction": setup["direction"],
                "match_scope": match_scope,
                "setup_gap_atr": float(setup["initial_gap_atr"]),
                "control_gap_atr": float(control["initial_gap_atr"]),
                "absolute_gap_atr_error": float(
                    abs(setup["initial_gap_atr"] - control["initial_gap_atr"])
                ),
                "setup_net_return": float(setup["net_return"]),
                "control_net_return": float(control["net_return"]),
                "paired_difference": float(setup["net_return"] - control["net_return"]),
                "setup_exit_reason": setup["exit_reason"],
                "control_exit_reason": control["exit_reason"],
            }
        )
    return pd.DataFrame.from_records(pairs)


def summarize_controls(pairs: pd.DataFrame, ticker: str) -> dict:
    if pairs.empty:
        return {"ticker": ticker, "n_pairs": 0}
    diff = pairs["paired_difference"]
    std = float(diff.std(ddof=1))
    return {
        "ticker": ticker,
        "n_pairs": len(pairs),
        "setup_avg_net_bps": float(pairs["setup_net_return"].mean() * 10_000),
        "control_avg_net_bps": float(pairs["control_net_return"].mean() * 10_000),
        "paired_edge_bps": float(diff.mean() * 10_000),
        "paired_t_stat_iid": float(diff.mean() / (std / math.sqrt(len(diff))))
        if len(diff) > 1 and std > 0
        else np.nan,
        "setup_win_rate": float((pairs["setup_net_return"] > 0).mean()),
        "control_win_rate": float((pairs["control_net_return"] > 0).mean()),
        "median_abs_gap_atr_error": float(pairs["absolute_gap_atr_error"].median()),
        "adjacent_year_matches": int((pairs["match_scope"] == "adjacent_year").sum()),
    }


def yearly_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
    rows = []
    for (ticker, year), group in frame.groupby(["ticker", "year"], sort=True):
        stats = summarize_trades(group, label=str(year))
        rows.append(
            {
                "ticker": ticker,
                "year": int(year),
                "n": stats["n"],
                "win_rate": stats["win_rate"],
                "avg_net_bps": stats["avg_net_bps"],
                "profit_factor": stats["profit_factor"],
                "arithmetic_net_return": stats["arithmetic_net_return"],
            }
        )
    return pd.DataFrame(rows)


def breakdown_table(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = ((2000, 2010), (2011, 2019), (2020, 2099))
    for ticker, ticker_trades in trades.groupby("ticker", sort=True):
        for direction, group in ticker_trades.groupby("direction", sort=True):
            rows.append(
                {
                    "ticker": ticker,
                    "slice": f"direction:{direction}",
                    **summarize_trades(group),
                }
            )
        years = pd.to_datetime(ticker_trades["trade_date"]).dt.year
        for lo, hi in periods:
            group = ticker_trades.loc[(years >= lo) & (years <= hi)]
            if not group.empty:
                rows.append(
                    {
                        "ticker": ticker,
                        "slice": f"period:{lo}-{min(hi, 2026)}",
                        **summarize_trades(group),
                    }
                )
    return pd.DataFrame(rows)


def stability_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Show whether one calendar year carries the aggregate result."""
    rows = []
    for ticker, group in trades.groupby("ticker", sort=True):
        frame = group.copy()
        frame["year"] = pd.to_datetime(frame["trade_date"]).dt.year
        yearly_returns = (
            frame.groupby("year")["net_return"].sum().sort_values(ascending=False)
        )
        best_year = int(yearly_returns.index[0])
        best_trade_index = frame["net_return"].idxmax()
        best_trade = frame.loc[best_trade_index]
        without_best = frame.loc[frame["year"] != best_year]
        without_stats = summarize_trades(without_best)
        without_best_trade = frame.drop(index=best_trade_index)
        without_trade_stats = summarize_trades(without_best_trade)
        total_arithmetic = float(frame["net_return"].sum())
        rows.append(
            {
                "ticker": ticker,
                "calendar_years": len(yearly_returns),
                "positive_years": int((yearly_returns > 0).sum()),
                "best_year": best_year,
                "best_year_arithmetic_return": float(yearly_returns.iloc[0]),
                "best_year_share_of_total": float(
                    yearly_returns.iloc[0] / total_arithmetic
                )
                if total_arithmetic
                else np.nan,
                "drop_best_year_n": without_stats["n"],
                "drop_best_year_avg_net_bps": without_stats["avg_net_bps"],
                "drop_best_year_profit_factor": without_stats["profit_factor"],
                "best_trade_date": pd.Timestamp(best_trade["trade_date"])
                .date()
                .isoformat(),
                "best_trade_net_return": float(best_trade["net_return"]),
                "best_trade_share_of_total": float(
                    best_trade["net_return"] / total_arithmetic
                )
                if total_arithmetic
                else np.nan,
                "drop_best_trade_avg_net_bps": without_trade_stats["avg_net_bps"],
                "drop_best_trade_profit_factor": without_trade_stats["profit_factor"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(
    frame: pd.DataFrame, columns: Iterable[str], formats: dict[str, str] | None = None
) -> str:
    formats = formats or {}
    columns = list(columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                rendered = "—"
            elif column in formats:
                rendered = formats[column].format(value)
            else:
                rendered = str(value)
            values.append(rendered)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_equity_plot(trades: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axis = plt.subplots(figsize=(10, 5.2))
    for ticker, group in trades.groupby("ticker", sort=True):
        ordered = group.sort_values("trade_date")
        equity = (1 + ordered["net_return"]).cumprod() - 1
        axis.plot(
            pd.to_datetime(ordered["trade_date"]),
            equity * 100,
            label=ticker,
            linewidth=1.8,
        )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_title("Legend EMA proxy replication — full-notional trade equity")
    axis.set_ylabel("Compounded net return (%)")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def create_report(
    output_dir: Path,
    qc: pd.DataFrame,
    summary: pd.DataFrame,
    common_summary: pd.DataFrame,
    variants: pd.DataFrame,
    control_summary: pd.DataFrame,
    stability: pd.DataFrame,
    friction_bps: float,
    common_start: pd.Timestamp,
) -> None:
    metric_formats = {
        "win_rate": "{:.1%}",
        "target_hit_rate": "{:.1%}",
        "avg_net_bps": "{:.2f}",
        "profit_factor": "{:.2f}",
        "compounded_net_return": "{:.1%}",
        "max_drawdown": "{:.1%}",
        "trades_per_year": "{:.1f}",
    }
    summary_view = summary.rename(columns={"label": "ticker"})
    variants_view = variants.loc[
        variants["variant"].isin(
            [
                "base",
                "trend_70",
                "trend_80",
                "close_only_touch",
                "prior_bar_touch",
                "fixed_target",
                "atr_stop_0.5",
            ]
        )
    ].copy()
    lines = [
        "# Legend EMA next-day mean-reversion replication",
        "",
        "Independent SPY/QQQ proxy test of the rules posted by MrMilkTrading, derived from Linda Raschke's pit-session observation.",
        "",
        "## Headline results",
        "",
        markdown_table(
            summary_view,
            [
                "ticker",
                "n",
                "trades_per_year",
                "win_rate",
                "target_hit_rate",
                "avg_net_bps",
                "profit_factor",
                "compounded_net_return",
                "max_drawdown",
            ],
            metric_formats,
        ),
        "",
        f"Net results deduct {friction_bps:.1f} bps round-trip on every ETF trade. Full-notional compounded return assumes cash earns 0% between signals.",
        "",
        f"## Common sample ({common_start.date().isoformat()} onward)",
        "",
        markdown_table(
            common_summary.rename(columns={"label": "ticker"}),
            [
                "ticker",
                "n",
                "win_rate",
                "avg_net_bps",
                "profit_factor",
                "compounded_net_return",
                "max_drawdown",
            ],
            metric_formats,
        ),
        "",
        "## Matched ordinary-day control",
        "",
        "Each setup is paired without replacement to a non-setup trade in the same instrument, calendar year, and direction with the nearest initial gap/ATR. Adjacent years are used only when a same-year match is unavailable.",
        "",
        markdown_table(
            control_summary,
            [
                "ticker",
                "n_pairs",
                "setup_avg_net_bps",
                "control_avg_net_bps",
                "paired_edge_bps",
                "setup_win_rate",
                "control_win_rate",
                "median_abs_gap_atr_error",
            ],
            {
                "setup_avg_net_bps": "{:.2f}",
                "control_avg_net_bps": "{:.2f}",
                "paired_edge_bps": "{:.2f}",
                "setup_win_rate": "{:.1%}",
                "control_win_rate": "{:.1%}",
                "median_abs_gap_atr_error": "{:.4f}",
            },
        ),
        "",
        "## Calendar and outlier stability",
        "",
        markdown_table(
            stability,
            [
                "ticker",
                "calendar_years",
                "positive_years",
                "best_year",
                "best_year_share_of_total",
                "drop_best_year_avg_net_bps",
                "best_trade_date",
                "best_trade_share_of_total",
                "drop_best_trade_avg_net_bps",
                "drop_best_trade_profit_factor",
            ],
            {
                "best_year_share_of_total": "{:.1%}",
                "drop_best_year_avg_net_bps": "{:.2f}",
                "best_trade_share_of_total": "{:.1%}",
                "drop_best_trade_avg_net_bps": "{:.2f}",
                "drop_best_trade_profit_factor": "{:.2f}",
            },
        ),
        "",
        "## Robustness variants",
        "",
        markdown_table(
            variants_view,
            [
                "ticker",
                "variant",
                "n",
                "win_rate",
                "avg_net_bps",
                "profit_factor",
                "compounded_net_return",
                "max_drawdown",
            ],
            metric_formats,
        ),
        "",
        "## Methodology",
        "",
        "- RTH-only 15-minute bars, 09:30 through 15:45 ET, with EMA(20) carried continuously across sessions and no overnight bars.",
        "- Base setup requires a full 26-bar prior session and trade session, body/range >= 75%, and no inclusive low/high intersection with the finalized same-bar EMA.",
        "- At 09:30, enter toward the prior session's final EMA. During bar *t*, the working target is EMA[t-1]; EMA[t] cannot retroactively fill bar *t*.",
        "- A limit gapped through receives opening-price improvement. The optional 0.5x prior ATR(14) stop is pessimistically assumed to trigger first if stop and target share one 15-minute bar.",
        "- Returns are simple notional returns, so old and new observations have equal percentage weight.",
        "",
        "## Data quality",
        "",
        markdown_table(
            qc,
            [
                "ticker",
                "rows_rth_deduped",
                "first_ts",
                "last_ts",
                "sessions",
                "full_26_bar_sessions",
                "partial_sessions",
                "duplicate_timestamps_raw",
                "null_cells_raw",
            ],
        ),
        "",
        "## Exact ES/NQ data status",
        "",
        "No 10-25 year ES/NQ intraday files or callable futures-data credentials were present locally, so no paid download or account action was taken. Databento documents CME Globex coverage from June 2010 and continuous futures symbols, but those mapped prices are unadjusted across rolls. FirstRate Data advertises paid 1-minute ES and NQ series from January 2008 in unadjusted and adjusted continuous forms. This report therefore uses the requested ETF fallback rather than presenting proxies as futures results.",
        "",
        "## Interpretation limits",
        "",
        "- SPY and QQQ are cash-session ETF proxies, not ES/NQ futures. Their opening auctions, dividends, financing, overnight gaps, and basis differ.",
        "- The local bars are unadjusted hybrid FMP/yfinance data. The base excludes partial sessions; it does not infer missing intrabar paths beyond the stated conservative stop rule.",
        "- Limit fills at an observed price level are not queue simulations. Friction sensitivities are in `variants.csv`.",
        "- ETF opening prints can be unusually hard to capture during auction/LULD dislocations. The stability table therefore shows results after removing the single best trade and best calendar year.",
        "- QQQ's largest setup gap (the 2015-08-24 dislocation) has a loose matched control. Excluding that pair leaves an 11.63 bps paired edge (iid t-stat 2.04).",
        "- Target-hit rate is not win rate: the rolling EMA can move beyond entry, so some EMA-target exits realize a loss.",
        "- A signal is never carried past an incomplete next trading session; the base requires the immediately adjacent observed sessions to both contain all 26 RTH bars.",
        "- The iid t-statistics in CSV outputs are descriptive only; clustered signals and regime dependence reduce effective sample size.",
        "- This is an independent replication of a pre-specified public rule, not a production recommendation or order path.",
        "",
        "## Source links",
        "",
        "- [MrMilk setup rules](https://x.com/MrMilkTrading/status/2094768637563338935) and [trade rules](https://x.com/MrMilkTrading/status/2094768655468724490)",
        "- [Linda Raschke original observation](https://x.com/LindaRaschke/status/2083196822621687898)",
        "- [Databento CME coverage](https://databento.com/docs/knowledge-base/datasets) and [continuous-contract methodology](https://databento.com/docs/standards-and-conventions/symbology)",
        "- [FirstRate Data ES](https://firstratedata.com/i/futures/ES) and [NQ](https://firstratedata.com/i/futures/NQ) coverage pages",
        "- [Local intraday architecture](../../docs/intraday_data_plan.md)",
        "",
        "Supporting files: `base_trades.csv`, `variants.csv`, `yearly.csv`, `breakdowns.csv`, `stability.csv`, `control_pairs.csv`, `control_summary.csv`, `data_qc.csv`, and `manifest.json`.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory holding {TICKER}_15min.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--friction-bps", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = BacktestConfig(friction_bps=args.friction_bps)
    qc_rows = []
    base_setup_frames = []
    base_all_by_ticker: dict[str, pd.DataFrame] = {}
    variant_rows = []
    available_starts = []
    inputs = []

    for ticker in args.tickers:
        source_path = (args.data_dir / f"{ticker}_15min.parquet").resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        bars, qc = load_intraday(source_path, ema_span=base_config.ema_span)
        daily = build_daily(bars)
        qc["ticker"] = ticker
        qc_rows.append(qc)
        inputs.append(
            {"ticker": ticker, "path": str(source_path), "sha256": qc["sha256"]}
        )

        available_starts.append(pd.Timestamp(qc["first_ts"]).normalize())
        session_groups = {
            session: group.sort_values("ts")
            for session, group in bars.groupby("session", sort=True)
        }
        base_all = run_all_eligible_trades(
            ticker, bars, daily, base_config, session_groups=session_groups
        )
        static_masks = {
            "base": base_all["is_setup"],
            "trend_70": (base_all["prior_trend_ratio"] >= 0.70)
            & base_all["prior_no_touch_same_bar"],
            "trend_80": (base_all["prior_trend_ratio"] >= 0.80)
            & base_all["prior_no_touch_same_bar"],
            "close_only_touch": (base_all["prior_trend_ratio"] >= 0.75)
            & base_all["prior_no_touch_close_only"],
            "prior_bar_touch": (base_all["prior_trend_ratio"] >= 0.75)
            & base_all["prior_no_touch_prior_bar"],
        }
        for variant_name, mask in static_masks.items():
            setups = base_all.loc[mask].copy()
            variant_rows.append(
                {
                    "ticker": ticker,
                    "variant": variant_name,
                    **summarize_trades(setups, label=ticker),
                }
            )

        for variant_name, config in {
            "fixed_target": replace(base_config, target_mode="fixed"),
            "atr_stop_0.5": replace(base_config, stop_atr_mult=0.5),
        }.items():
            path_all = run_all_eligible_trades(
                ticker, bars, daily, config, session_groups=session_groups
            )
            path_setups = path_all.loc[path_all["is_setup"]].copy()
            variant_rows.append(
                {
                    "ticker": ticker,
                    "variant": variant_name,
                    **summarize_trades(path_setups, label=ticker),
                }
            )

        base_setups = base_all.loc[base_all["is_setup"]].copy()
        base_setup_frames.append(base_setups)
        base_all_by_ticker[ticker] = base_all

        for friction in (0.0, 1.0, 2.0, 5.0):
            adjusted = base_setups.copy()
            adjusted["net_return"] = adjusted["gross_return"] - friction / 10_000
            variant_rows.append(
                {
                    "ticker": ticker,
                    "variant": f"friction_{friction:g}bps",
                    **summarize_trades(adjusted, label=ticker),
                }
            )

    base_trades = pd.concat(base_setup_frames, ignore_index=True).sort_values(
        ["ticker", "trade_date"]
    )
    base_trades.to_csv(output_dir / "base_trades.csv", index=False)
    summary = pd.DataFrame(
        [
            summarize_trades(group, label=ticker)
            for ticker, group in base_trades.groupby("ticker", sort=True)
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    common_start = max(available_starts)
    common_rows = []
    for ticker, group in base_trades.groupby("ticker", sort=True):
        common = group.loc[pd.to_datetime(group["trade_date"]) >= common_start]
        common_rows.append(summarize_trades(common, label=ticker))
    common_summary = pd.DataFrame(common_rows)
    common_summary.to_csv(output_dir / "common_sample_summary.csv", index=False)

    variants = pd.DataFrame(variant_rows)
    variants.to_csv(output_dir / "variants.csv", index=False)
    yearly = yearly_table(base_trades)
    yearly.to_csv(output_dir / "yearly.csv", index=False)
    breakdowns = breakdown_table(base_trades)
    breakdowns.to_csv(output_dir / "breakdowns.csv", index=False)
    stability = stability_table(base_trades)
    stability.to_csv(output_dir / "stability.csv", index=False)

    all_pairs = []
    control_rows = []
    for ticker, all_trades in base_all_by_ticker.items():
        pairs = match_controls(all_trades)
        if not pairs.empty:
            all_pairs.append(pairs)
        control_rows.append(summarize_controls(pairs, ticker))
    control_pairs = (
        pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    )
    control_pairs.to_csv(output_dir / "control_pairs.csv", index=False)
    control_summary = pd.DataFrame(control_rows)
    control_summary.to_csv(output_dir / "control_summary.csv", index=False)

    qc_frame = pd.DataFrame(qc_rows)
    qc_frame.to_csv(output_dir / "data_qc.csv", index=False)
    plot_written = write_equity_plot(base_trades, output_dir / "equity_curves.png")
    create_report(
        output_dir,
        qc_frame,
        summary,
        common_summary,
        variants,
        control_summary,
        stability,
        args.friction_bps,
        common_start,
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "config": base_config.__dict__,
        "common_sample_start": common_start.date().isoformat(),
        "inputs": inputs,
        "outputs": sorted(
            {path.name for path in output_dir.iterdir()} | {"manifest.json"}
        ),
        "plot_written": plot_written,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
