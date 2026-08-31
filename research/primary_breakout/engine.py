"""Causal event-driven backtest for the primary-universe breakout study.

This module is deliberately isolated from the production strategy book.  It
reads price data and universe constants, but it never writes scanner, staging,
portfolio, broker, cache, or site state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

REQUIRED_PRICE_COLUMNS = ("ticker", "date", "Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2000-01-03"
    end_date: str | None = None
    initial_equity: float = 750_000.0
    breakout_window: int = 100
    roc_window: int = 100
    ck_atr_period: int = 10
    ck_atr_multiple: float = 3.0
    ck_stop_period: int = 9
    risk_fraction: float = 0.005
    max_name_fraction: float = 0.10
    max_gross_fraction: float = 1.00
    max_open_risk_fraction: float = 0.05
    cost_bps: float = 5.0
    benchmark_ticker: str = "SPY"

    def validate(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        for name in ("breakout_window", "roc_window", "ck_atr_period", "ck_stop_period"):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be at least 2")
        for name in (
            "risk_fraction",
            "max_name_fraction",
            "max_gross_fraction",
            "max_open_risk_fraction",
        ):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.ck_atr_multiple <= 0 or self.cost_bps < 0:
            raise ValueError("CK multiplier must be positive and costs non-negative")


@dataclass
class BacktestResult:
    config: BacktestConfig
    universe: list[str]
    equity: pd.DataFrame
    trades: pd.DataFrame
    candidates: pd.DataFrame
    yearly_returns: pd.DataFrame
    metrics: dict[str, float | int | str | bool]
    data_quality: dict[str, object]


@dataclass
class _Position:
    ticker: str
    shares: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    raw_entry_open: float
    initial_stop: float
    active_stop: float
    risk_per_share: float
    target_risk: float
    initial_risk: float
    signal_roc: float
    entry_notional: float
    entry_index: int


def build_stock_universe(primary: Iterable[str], etf_exclusions: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return sorted stock-like primary tickers and the applied exclusions."""

    ordered = [str(t).upper().strip() for t in primary if str(t).strip()]
    excluded_set = {str(t).upper().strip() for t in etf_exclusions if str(t).strip()}
    excluded = sorted({t for t in ordered if t in excluded_set})
    stocks = sorted({t for t in ordered if t not in excluded_set})
    overlap = set(stocks) & set(excluded)
    if overlap:
        raise ValueError(f"universe classification overlap: {sorted(overlap)}")
    return stocks, excluded


def load_price_panel(
    prices_path: str | Path,
    tickers: Iterable[str],
    benchmark_ticker: str = "SPY",
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, dict[str, object]]:
    """Load selected adjusted OHLCV and align it to the benchmark calendar.

    Missing observations are retained as NaN after reindexing.  Indicator
    windows require complete observations, and an open position encountering a
    missing session fails the simulation rather than silently carrying stale
    marks.
    """

    path = Path(prices_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    wanted = sorted({str(t).upper().strip() for t in tickers} | {benchmark_ticker.upper()})
    dataset = pads.dataset(str(path), format="parquet")
    missing_cols = sorted(set(REQUIRED_PRICE_COLUMNS) - set(dataset.schema.names))
    if missing_cols:
        raise ValueError(f"price parquet missing columns: {missing_cols}")
    table = dataset.to_table(
        columns=list(REQUIRED_PRICE_COLUMNS),
        filter=pads.field("ticker").isin(wanted),
    )
    raw = table.to_pandas()
    if raw.empty:
        raise ValueError("price parquet returned no selected rows")
    raw["ticker"] = raw["ticker"].astype(str).str.upper().str.strip()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    raw = raw.dropna(subset=["ticker", "date"])
    dupes = raw.duplicated(["ticker", "date"], keep=False)
    if dupes.any():
        sample = raw.loc[dupes, ["ticker", "date"]].head(10).to_dict("records")
        raise ValueError(f"duplicate ticker/date bars: {sample}")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    nonpositive = (raw[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    malformed = (
        (raw["High"] + 1e-6 < raw[["Open", "Low", "Close"]].max(axis=1))
        | (raw["Low"] - 1e-6 > raw[["Open", "High", "Close"]].min(axis=1))
    )
    if nonpositive.any() or malformed.any():
        bad = raw.loc[nonpositive | malformed, ["ticker", "date", "Open", "High", "Low", "Close"]]
        raise ValueError(f"invalid OHLC rows in selected universe: {bad.head(10).to_dict('records')}")

    found = set(raw["ticker"].unique())
    missing_tickers = sorted(set(wanted) - found)
    if benchmark_ticker.upper() in missing_tickers:
        raise ValueError(f"benchmark {benchmark_ticker} missing from price data")

    benchmark_rows = raw[raw["ticker"] == benchmark_ticker.upper()].sort_values("date")
    calendar = pd.DatetimeIndex(benchmark_rows["date"].unique()).sort_values()
    if len(calendar) < 252:
        raise ValueError("benchmark calendar is too short")

    panel: dict[str, pd.DataFrame] = {}
    observed_counts: dict[str, int] = {}
    for ticker, group in raw.groupby("ticker", sort=True):
        frame = group.sort_values("date").set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
        frame = frame.reindex(calendar)
        panel[ticker] = frame
        observed_counts[ticker] = int(frame["Close"].notna().sum())

    quality = {
        "prices_path": str(path),
        "selected_tickers": len(wanted),
        "loaded_tickers": len(found),
        "missing_tickers": missing_tickers,
        "calendar_start": calendar.min().date().isoformat(),
        "calendar_end": calendar.max().date().isoformat(),
        "calendar_sessions": len(calendar),
        "observed_bar_counts": observed_counts,
        "adjustment_basis": "yfinance auto_adjust=True cache (adjusted OHLCV)",
    }
    return panel, calendar, quality


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Exact Wilder ATR, resetting after any missing OHLC observation."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    values = tr.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    i = 0
    while i < len(values):
        while i < len(values) and not np.isfinite(values[i]):
            i += 1
        start = i
        while i < len(values) and np.isfinite(values[i]):
            i += 1
        end = i
        if end - start < period:
            continue
        seed_idx = start + period - 1
        out[seed_idx] = float(np.mean(values[start : seed_idx + 1]))
        for j in range(seed_idx + 1, end):
            out[j] = (out[j - 1] * (period - 1) + values[j]) / period
    return pd.Series(out, index=high.index, name="ATR")


def compute_features(bars: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Compute causal breakout, raw ROC, and long Chande-Kroll stop series."""

    config.validate()
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(bars.columns):
        raise ValueError(f"bars missing {sorted(required - set(bars.columns))}")
    out = bars.copy()
    prior_high = out["High"].shift(1).rolling(
        config.breakout_window, min_periods=config.breakout_window
    ).max()
    breakout_state_valid = out["Close"].notna() & prior_high.notna()
    breakout = (out["Close"] > prior_high) & breakout_state_valid
    prior_state_valid = breakout_state_valid.shift(1, fill_value=False).astype(bool)
    fresh = breakout & prior_state_valid & ~breakout.shift(1, fill_value=False).astype(bool)
    roc = out["Close"] / out["Close"].shift(config.roc_window) - 1.0

    atr = _wilder_atr(out["High"], out["Low"], out["Close"], config.ck_atr_period)
    ck_high = out["High"].rolling(
        config.ck_atr_period, min_periods=config.ck_atr_period
    ).max()
    ck_prelim = ck_high - config.ck_atr_multiple * atr
    ck_stop = ck_prelim.rolling(
        config.ck_stop_period, min_periods=config.ck_stop_period
    ).max()

    out["PriorBreakoutHigh"] = prior_high
    out["Breakout"] = breakout.astype(bool)
    out["FreshBreakout"] = fresh.astype(bool)
    out["ROC"] = roc
    out["ATR"] = atr
    out["CKPreliminary"] = ck_prelim
    out["CKStop"] = ck_stop
    return out


def _close_trade(
    position: _Position,
    exit_date: pd.Timestamp,
    exit_price: float,
    raw_exit_price: float,
    reason: str,
    exit_index: int,
) -> dict[str, object]:
    pnl = (exit_price - position.entry_price) * position.shares
    achieved_risk = max(position.initial_risk, 1e-12)
    return {
        "Ticker": position.ticker,
        "SignalDate": position.signal_date,
        "EntryDate": position.entry_date,
        "ExitDate": exit_date,
        "Shares": position.shares,
        "EntryPrice": position.entry_price,
        "RawEntryOpen": position.raw_entry_open,
        "InitialStop": position.initial_stop,
        "ExitPrice": exit_price,
        "RawExitPrice": raw_exit_price,
        "ExitReason": reason,
        "SignalROC100": position.signal_roc,
        "TargetRisk": position.target_risk,
        "InitialRisk": position.initial_risk,
        "AchievedRiskPct": position.initial_risk / position.target_risk if position.target_risk else np.nan,
        "PnL": pnl,
        "R": pnl / achieved_risk,
        "HoldingSessions": int(exit_index - position.entry_index + 1),
        "EntryNotional": position.entry_notional,
        "ExitNotional": exit_price * position.shares,
    }


def _binding_reason(limits: dict[str, int]) -> str:
    if not limits:
        return "unknown_constraint"
    minimum = min(limits.values())
    binders = sorted(k for k, value in limits.items() if value == minimum)
    return "+".join(binders)


def run_backtest(
    panel: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    universe: Iterable[str],
    config: BacktestConfig,
    data_quality: dict[str, object] | None = None,
) -> BacktestResult:
    """Run the daily portfolio simulation with next-open ranked admission."""

    config.validate()
    tickers = sorted({str(t).upper() for t in universe})
    missing = sorted(set(tickers) - set(panel))
    usable = [t for t in tickers if t in panel]
    if not usable:
        raise ValueError("no universe tickers have price data")

    features = {ticker: compute_features(panel[ticker], config) for ticker in usable}
    feature_arrays: dict[str, dict[str, np.ndarray]] = {}
    for ticker in usable:
        frame = features[ticker].reindex(calendar)
        feature_arrays[ticker] = {
            column: frame[column].to_numpy(dtype=float, copy=False)
            for column in ("Open", "High", "Low", "Close", "ROC", "CKStop")
        }
        feature_arrays[ticker]["FreshBreakout"] = frame["FreshBreakout"].to_numpy(
            dtype=bool, copy=False
        )
    start = pd.Timestamp(config.start_date).normalize()
    end = pd.Timestamp(config.end_date).normalize() if config.end_date else calendar.max()
    start_index = int(calendar.searchsorted(start, side="left"))
    end_index = int(calendar.searchsorted(end, side="right")) - 1
    if start_index >= len(calendar) or end_index < start_index:
        raise ValueError("simulation dates do not overlap the benchmark calendar")
    sim_indices = range(start_index, end_index + 1)
    sim_calendar = calendar[start_index : end_index + 1]
    if len(sim_calendar) < 2:
        raise ValueError("simulation calendar has fewer than two sessions")

    # Fresh breakouts are sparse.  Index them once by next-session entry date
    # instead of probing every ticker with millions of pandas scalar lookups
    # inside each daily portfolio loop.
    signal_events: dict[int, list[tuple[str, float, float]]] = {}
    for ticker in usable:
        arrays = feature_arrays[ticker]
        signal_indices = np.flatnonzero(arrays["FreshBreakout"])
        for signal_index in signal_indices:
            entry_index = int(signal_index) + 1
            if start_index <= entry_index <= end_index:
                signal_events.setdefault(entry_index, []).append(
                    (
                        ticker,
                        float(arrays["ROC"][signal_index]),
                        float(arrays["CKStop"][signal_index]),
                    )
                )

    cost = config.cost_bps / 10_000.0
    cash = float(config.initial_equity)
    positions: dict[str, _Position] = {}
    trades: list[dict[str, object]] = []
    candidate_log: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    prior_equity = float(config.initial_equity)

    for calendar_index in sim_indices:
        date = calendar[calendar_index]
        previous_date = calendar[calendar_index - 1] if calendar_index > 0 else None
        day_start_equity = prior_equity
        held_at_prior_close = set(positions)

        # Missing bars while held are a hard data-quality failure; stale marks
        # can otherwise invent both stop behavior and capital availability.
        for ticker in list(positions):
            arrays = feature_arrays[ticker]
            if np.isnan(
                [
                    arrays["Open"][calendar_index],
                    arrays["High"][calendar_index],
                    arrays["Low"][calendar_index],
                    arrays["Close"][calendar_index],
                ]
            ).any():
                raise ValueError(f"held position {ticker} missing OHLC on {date.date()}")

        # Existing positions that gap through yesterday's active stop exit at
        # the open before new entries are admitted.
        for ticker in list(positions):
            position = positions[ticker]
            raw_open = float(feature_arrays[ticker]["Open"][calendar_index])
            if raw_open <= position.active_stop:
                exit_price = raw_open * (1.0 - cost)
                cash += exit_price * position.shares
                trades.append(
                    _close_trade(position, date, exit_price, raw_open, "StopGap", calendar_index)
                )
                del positions[ticker]

        gross_open = 0.0
        open_stop_risk = 0.0
        for ticker, position in positions.items():
            raw_open = float(feature_arrays[ticker]["Open"][calendar_index])
            gross_open += raw_open * position.shares
            modeled_stop_exit = position.active_stop * (1.0 - cost)
            open_stop_risk += max(raw_open - modeled_stop_exit, 0.0) * position.shares

        # Signals known at the previous close compete at today's open.
        candidates: list[tuple[float, str, float]] = []
        if previous_date is not None:
            for ticker, roc, initial_stop in signal_events.get(calendar_index, []):
                if ticker in held_at_prior_close:
                    candidate_log.append(
                        {
                            "SignalDate": previous_date,
                            "EntryDate": date,
                            "Ticker": ticker,
                            "ROC": roc,
                            "Status": "Rejected",
                            "Reason": "already_held",
                            "Shares": 0,
                        }
                    )
                    continue
                if not np.isfinite(roc) or not np.isfinite(initial_stop):
                    candidate_log.append(
                        {
                            "SignalDate": previous_date,
                            "EntryDate": date,
                            "Ticker": ticker,
                            "ROC": roc,
                            "Status": "Rejected",
                            "Reason": "missing_signal_feature",
                            "Shares": 0,
                        }
                    )
                    continue
                arrays = feature_arrays[ticker]
                if np.isnan(
                    [
                        arrays["Open"][calendar_index],
                        arrays["High"][calendar_index],
                        arrays["Low"][calendar_index],
                        arrays["Close"][calendar_index],
                    ]
                ).any():
                    candidate_log.append(
                        {
                            "SignalDate": previous_date,
                            "EntryDate": date,
                            "Ticker": ticker,
                            "ROC": roc,
                            "Status": "Rejected",
                            "Reason": "missing_entry_bar",
                            "Shares": 0,
                        }
                    )
                    continue
                candidates.append((roc, ticker, initial_stop))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        for roc, ticker, initial_stop in candidates:
            raw_open = float(feature_arrays[ticker]["Open"][calendar_index])
            entry_price = raw_open * (1.0 + cost)
            if raw_open <= initial_stop or entry_price <= initial_stop:
                candidate_log.append(
                    {
                        "SignalDate": previous_date,
                        "EntryDate": date,
                        "Ticker": ticker,
                        "ROC": roc,
                        "Status": "Rejected",
                        "Reason": "open_at_or_below_stop",
                        "Shares": 0,
                    }
                )
                continue

            modeled_stop_exit = initial_stop * (1.0 - cost)
            risk_per_share = entry_price - modeled_stop_exit
            target_risk = config.risk_fraction * day_start_equity
            risk_headroom = max(
                config.max_open_risk_fraction * day_start_equity - open_stop_risk, 0.0
            )
            gross_headroom = max(
                config.max_gross_fraction * day_start_equity - gross_open, 0.0
            )
            limits = {
                "risk_target": int(np.floor(target_risk / risk_per_share)),
                "name_cap": int(
                    np.floor(config.max_name_fraction * day_start_equity / entry_price)
                ),
                "cash": int(np.floor(max(cash, 0.0) / entry_price)),
                "gross_cap": int(np.floor(gross_headroom / entry_price)),
                "open_risk_cap": int(np.floor(risk_headroom / risk_per_share)),
            }
            shares = max(min(limits.values()), 0)
            if shares < 1:
                candidate_log.append(
                    {
                        "SignalDate": previous_date,
                        "EntryDate": date,
                        "Ticker": ticker,
                        "ROC": roc,
                        "Status": "Rejected",
                        "Reason": _binding_reason(limits),
                        "Shares": 0,
                    }
                )
                continue

            entry_notional = entry_price * shares
            initial_risk = risk_per_share * shares
            cash -= entry_notional
            gross_open += raw_open * shares
            open_stop_risk += initial_risk
            positions[ticker] = _Position(
                ticker=ticker,
                shares=shares,
                signal_date=previous_date,
                entry_date=date,
                entry_price=entry_price,
                raw_entry_open=raw_open,
                initial_stop=initial_stop,
                active_stop=initial_stop,
                risk_per_share=risk_per_share,
                target_risk=target_risk,
                initial_risk=initial_risk,
                signal_roc=roc,
                entry_notional=entry_notional,
                entry_index=calendar_index,
            )
            candidate_log.append(
                {
                    "SignalDate": previous_date,
                    "EntryDate": date,
                    "Ticker": ticker,
                    "ROC": roc,
                    "Status": "Entered",
                    "Reason": _binding_reason(limits),
                    "Shares": shares,
                    "TargetRisk": target_risk,
                    "InitialRisk": initial_risk,
                    "InitialRiskPctOfTarget": initial_risk / target_risk,
                }
            )

        # Intraday stop touches use the stop that was active before today's
        # close.  Today's newly calculated CK value is not allowed to reach
        # backward into today's range.
        for ticker in list(positions):
            position = positions[ticker]
            if float(feature_arrays[ticker]["Low"][calendar_index]) <= position.active_stop:
                raw_exit = position.active_stop
                exit_price = raw_exit * (1.0 - cost)
                cash += exit_price * position.shares
                trades.append(
                    _close_trade(position, date, exit_price, raw_exit, "StopTouch", calendar_index)
                )
                del positions[ticker]

        # Mark the book, then ratchet today's computed CK stop for tomorrow.
        gross_close = 0.0
        equity_close = cash
        for ticker, position in positions.items():
            close = float(feature_arrays[ticker]["Close"][calendar_index])
            market_value = close * position.shares
            gross_close += market_value
            equity_close += market_value

        for ticker, position in positions.items():
            ck_today = feature_arrays[ticker]["CKStop"][calendar_index]
            if np.isfinite(ck_today):
                position.active_stop = max(position.active_stop, float(ck_today))

        effective_risk = 0.0
        for ticker, position in positions.items():
            close = float(feature_arrays[ticker]["Close"][calendar_index])
            modeled_stop_exit = position.active_stop * (1.0 - cost)
            effective_risk += max(close - modeled_stop_exit, 0.0) * position.shares
        daily.append(
            {
                "Date": date,
                "Equity": equity_close,
                "Cash": cash,
                "Gross": gross_close,
                "GrossPct": gross_close / equity_close if equity_close else np.nan,
                "OpenStopRisk": effective_risk,
                "OpenStopRiskPct": effective_risk / equity_close if equity_close else np.nan,
                "Positions": len(positions),
            }
        )
        prior_equity = equity_close

    # End-of-sample liquidation makes variants comparable and includes one
    # final exit-cost assumption.  It is not a strategy time stop.
    last_date = sim_calendar[-1]
    last_index = end_index
    for ticker in list(positions):
        position = positions[ticker]
        raw_exit = float(feature_arrays[ticker]["Close"][last_index])
        exit_price = raw_exit * (1.0 - cost)
        cash += exit_price * position.shares
        trades.append(
            _close_trade(position, last_date, exit_price, raw_exit, "EndOfSample", last_index)
        )
        del positions[ticker]
    if daily:
        daily[-1].update(
            {
                "Equity": cash,
                "Cash": cash,
                "Gross": 0.0,
                "GrossPct": 0.0,
                "OpenStopRisk": 0.0,
                "OpenStopRiskPct": 0.0,
                "Positions": 0,
            }
        )

    equity = pd.DataFrame(daily).set_index("Date")
    equity["Return"] = equity["Equity"].pct_change().fillna(
        equity["Equity"].iloc[0] / config.initial_equity - 1.0
    )
    equity["Peak"] = equity["Equity"].cummax()
    equity["Drawdown"] = equity["Equity"] / equity["Peak"] - 1.0

    benchmark = panel[config.benchmark_ticker.upper()].reindex(equity.index)
    if benchmark["Close"].isna().any():
        raise ValueError("benchmark has missing closes inside simulation window")
    benchmark_curve = config.initial_equity * benchmark["Close"] / benchmark["Close"].iloc[0]
    equity["BenchmarkEquity"] = benchmark_curve
    equity["BenchmarkReturn"] = benchmark_curve.pct_change().fillna(0.0)
    equity["BenchmarkPeak"] = benchmark_curve.cummax()
    equity["BenchmarkDrawdown"] = benchmark_curve / equity["BenchmarkPeak"] - 1.0

    # Static-universe, daily-rebalanced equal-weight comparator.  It carries
    # the same survivorship caveat as the study universe and has no costs; its
    # purpose is to separate the breakout/stop logic from merely owning this
    # hand-selected group of liquid stocks.
    equal_weight_returns = pd.concat(
        [
            panel[ticker]["Close"].pct_change(fill_method=None).rename(ticker)
            for ticker in usable
        ],
        axis=1,
    ).reindex(equity.index).mean(axis=1, skipna=True).fillna(0.0)
    equal_weight_curve = config.initial_equity * (1.0 + equal_weight_returns).cumprod()
    equity["EqualWeightReturn"] = equal_weight_returns
    equity["EqualWeightEquity"] = equal_weight_curve
    equity["EqualWeightPeak"] = equal_weight_curve.cummax()
    equity["EqualWeightDrawdown"] = equal_weight_curve / equity["EqualWeightPeak"] - 1.0

    trades_df = pd.DataFrame(trades)
    candidates_df = pd.DataFrame(candidate_log)
    yearly = _yearly_returns(equity)
    metrics = _performance_metrics(equity, trades_df, candidates_df, config)
    quality = dict(data_quality or {})
    quality.update(
        {
            "requested_universe_count": len(tickers),
            "usable_universe_count": len(usable),
            "missing_requested_tickers": missing,
            "simulation_start": equity.index.min().date().isoformat(),
            "simulation_end": equity.index.max().date().isoformat(),
            "config": asdict(config),
        }
    )
    return BacktestResult(
        config=config,
        universe=usable,
        equity=equity,
        trades=trades_df,
        candidates=candidates_df,
        yearly_returns=yearly,
        metrics=metrics,
        data_quality=quality,
    )


def _yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    grouped = equity[["Return", "BenchmarkReturn", "EqualWeightReturn"]].groupby(
        equity.index.year
    )
    rows = []
    for year, frame in grouped:
        rows.append(
            {
                "Year": int(year),
                "Strategy": float((1.0 + frame["Return"]).prod() - 1.0),
                "SPY": float((1.0 + frame["BenchmarkReturn"]).prod() - 1.0),
                "EqualWeightPrimary": float(
                    (1.0 + frame["EqualWeightReturn"]).prod() - 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _curve_stats(
    equity: pd.Series,
    returns: pd.Series,
    initial_value: float | None = None,
) -> dict[str, float]:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 252.0)
    base = float(initial_value if initial_value is not None else equity.iloc[0])
    total = float(equity.iloc[-1] / base - 1.0)
    cagr = float((equity.iloc[-1] / base) ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(252.0))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)) if returns.std(ddof=1) > 0 else np.nan
    downside = returns.where(returns < 0, 0.0).std(ddof=1)
    sortino = float(returns.mean() / downside * np.sqrt(252.0)) if downside > 0 else np.nan
    peak = equity.cummax().clip(lower=base)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def _performance_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    candidates: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float | int | str | bool]:
    strategy = _curve_stats(
        equity["Equity"], equity["Return"], initial_value=config.initial_equity
    )
    benchmark = _curve_stats(
        equity["BenchmarkEquity"],
        equity["BenchmarkReturn"],
        initial_value=config.initial_equity,
    )
    equal_weight = _curve_stats(
        equity["EqualWeightEquity"],
        equity["EqualWeightReturn"],
        initial_value=config.initial_equity,
    )
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 252.0)
    metrics: dict[str, float | int | str | bool] = {
        **strategy,
        **{f"spy_{key}": value for key, value in benchmark.items()},
        **{f"ew_{key}": value for key, value in equal_weight.items()},
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "sessions": len(equity),
        "trades": len(trades),
        "candidate_events": len(candidates),
        "entered_candidates": int((candidates.get("Status") == "Entered").sum()) if not candidates.empty else 0,
        "average_positions": float(equity["Positions"].mean()),
        "max_positions": int(equity["Positions"].max()),
        "time_in_market": float((equity["Positions"] > 0).mean()),
        "average_gross": float(equity["GrossPct"].mean()),
        "max_gross": float(equity["GrossPct"].max()),
        "average_open_stop_risk": float(equity["OpenStopRiskPct"].mean()),
        "max_open_stop_risk": float(equity["OpenStopRiskPct"].max()),
    }
    if trades.empty:
        metrics.update(
            {
                "win_rate": np.nan,
                "profit_factor": np.nan,
                "average_r": np.nan,
                "median_r": np.nan,
                "average_holding_sessions": np.nan,
                "annual_turnover": 0.0,
                "top5_pnl_concentration": np.nan,
                "average_achieved_risk_pct": np.nan,
            }
        )
        return metrics

    wins = trades.loc[trades["PnL"] > 0, "PnL"].sum()
    losses = -trades.loc[trades["PnL"] < 0, "PnL"].sum()
    ticker_pnl = trades.groupby("Ticker")["PnL"].sum().sort_values(ascending=False)
    positive_total = ticker_pnl[ticker_pnl > 0].sum()
    top5 = ticker_pnl.head(5).clip(lower=0).sum()
    one_way_turnover = min(trades["EntryNotional"].sum(), trades["ExitNotional"].sum())
    metrics.update(
        {
            "win_rate": float((trades["PnL"] > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else np.inf,
            "average_r": float(trades["R"].mean()),
            "median_r": float(trades["R"].median()),
            "average_holding_sessions": float(trades["HoldingSessions"].mean()),
            "annual_turnover": float(one_way_turnover / equity["Equity"].mean() / years),
            "top5_pnl_concentration": float(top5 / positive_total) if positive_total > 0 else np.nan,
            "average_achieved_risk_pct": float(trades["AchievedRiskPct"].mean()),
            "worst_r": float(trades["R"].min()),
            "best_r": float(trades["R"].max()),
            "gap_stop_share": float((trades["ExitReason"] == "StopGap").mean()),
        }
    )
    return metrics


def slice_metrics(result: BacktestResult, start_date: str) -> dict[str, float]:
    """Compute strategy/benchmark curve metrics from a chronological cutoff."""

    frame = result.equity.loc[pd.Timestamp(start_date) :].copy()
    if len(frame) < 2:
        return {}
    strategy_base = frame["Equity"].iloc[0]
    benchmark_base = frame["BenchmarkEquity"].iloc[0]
    strategy_equity = frame["Equity"] / strategy_base
    benchmark_equity = frame["BenchmarkEquity"] / benchmark_base
    strategy_returns = strategy_equity.pct_change().fillna(0.0)
    benchmark_returns = benchmark_equity.pct_change().fillna(0.0)
    stats = _curve_stats(strategy_equity, strategy_returns)
    stats.update({f"spy_{k}": v for k, v in _curve_stats(benchmark_equity, benchmark_returns).items()})
    return stats


def period_metrics(
    result: BacktestResult,
    start_date: str,
    end_date: str | None = None,
) -> dict[str, float]:
    """Normalize and score a bounded chronological segment."""

    frame = result.equity.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date) if end_date else None]
    if len(frame) < 2:
        return {}
    output: dict[str, float] = {}
    for label, equity_col in (
        ("", "Equity"),
        ("spy_", "BenchmarkEquity"),
        ("ew_", "EqualWeightEquity"),
    ):
        normalized = frame[equity_col] / frame[equity_col].iloc[0]
        returns = normalized.pct_change().fillna(0.0)
        output.update({f"{label}{key}": value for key, value in _curve_stats(normalized, returns).items()})
    return output


def monthly_block_bootstrap_sharpe(
    equity: pd.DataFrame,
    samples: int = 2_000,
    block_months: int = 6,
    seed: int = 17,
) -> dict[str, float]:
    """Moving-block bootstrap interval for annualized monthly Sharpe."""

    monthly = (1.0 + equity["Return"]).resample("ME").prod() - 1.0
    values = monthly.dropna().to_numpy(dtype=float)
    if len(values) < max(24, block_months * 2):
        return {"sharpe_p05": np.nan, "sharpe_p50": np.nan, "sharpe_p95": np.nan, "prob_sharpe_gt_zero": np.nan}
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(values) - block_months + 1)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        drawn: list[float] = []
        while len(drawn) < len(values):
            start = int(rng.choice(starts))
            drawn.extend(values[start : start + block_months])
        sample = np.asarray(drawn[: len(values)], dtype=float)
        std = sample.std(ddof=1)
        boot[i] = sample.mean() / std * np.sqrt(12.0) if std > 0 else np.nan
    clean = boot[np.isfinite(boot)]
    if clean.size == 0:
        return {
            "sharpe_p05": np.nan,
            "sharpe_p50": np.nan,
            "sharpe_p95": np.nan,
            "prob_sharpe_gt_zero": np.nan,
        }
    return {
        "sharpe_p05": float(np.quantile(clean, 0.05)),
        "sharpe_p50": float(np.quantile(clean, 0.50)),
        "sharpe_p95": float(np.quantile(clean, 0.95)),
        "prob_sharpe_gt_zero": float((clean > 0).mean()),
    }
