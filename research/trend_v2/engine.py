"""Point-in-time research engines for the Trend V2 workstream.

The module is deliberately detached from production orchestration.  It accepts
already-adjusted local price panels, forms signals using observations available
at each signal close, applies targets one period later, and returns in-memory
research objects.  File output is owned by ``runner.py`` and requires an
explicit artifact directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BenchmarkSpec,
    CrossSectionalSpec,
    FROZEN_BENCHMARK,
    MultiSpeedSpec,
)


@dataclass(frozen=True)
class PriceData:
    """Adjusted daily price panels loaded from a local parquet."""

    close: pd.DataFrame
    open: pd.DataFrame | None
    source: str | None = None


@dataclass(frozen=True)
class TargetResult:
    targets: pd.DataFrame
    signal_state: pd.DataFrame
    votes: pd.DataFrame | None = None
    scores: pd.DataFrame | None = None
    ranks: pd.DataFrame | None = None


@dataclass(frozen=True)
class BacktestResult:
    monthly: pd.DataFrame
    targets: pd.DataFrame
    held_weights: pd.DataFrame


def _clean_panel(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    out = panel.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    if out.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate dates")
    out.columns = [str(column).upper().strip() for column in out.columns]
    if out.columns.duplicated().any():
        raise ValueError(f"{label} contains duplicate ticker columns")
    out = out.apply(pd.to_numeric, errors="coerce").sort_index()
    out = out.where(out > 0.0).dropna(how="all")
    if out.empty:
        raise ValueError(f"{label} has no positive observations")
    return out


def load_adjusted_price_parquet(path: str | Path, asof: str | None = None) -> PriceData:
    """Load adjusted prices from a *local* long- or wide-format parquet.

    Long format accepts case-insensitive ``date``, ``ticker``, ``Close`` and
    optional ``Open`` columns.  Wide format is interpreted as close-only with a
    DatetimeIndex.  The adjusted-price basis cannot be inferred from numbers,
    so callers must supply a parquet whose provenance establishes that basis.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"price parquet not found: {source}")
    raw = pd.read_parquet(source)
    lower = {str(column).lower(): column for column in raw.columns}
    if {"date", "ticker", "close"}.issubset(lower):
        date_col = lower["date"]
        ticker_col = lower["ticker"]
        close_col = lower["close"]
        frame = raw[[date_col, ticker_col, close_col]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame[ticker_col] = frame[ticker_col].astype(str).str.upper().str.strip()
        if frame.duplicated([date_col, ticker_col]).any():
            raise ValueError("price parquet contains duplicate date/ticker rows")
        close = frame.pivot(index=date_col, columns=ticker_col, values=close_col)
        open_panel = None
        if "open" in lower:
            open_frame = raw[[date_col, ticker_col, lower["open"]]].copy()
            open_frame[date_col] = pd.to_datetime(open_frame[date_col])
            open_frame[ticker_col] = open_frame[ticker_col].astype(str).str.upper().str.strip()
            open_panel = open_frame.pivot(
                index=date_col, columns=ticker_col, values=lower["open"]
            )
    else:
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise ValueError(
                "wide price parquet needs a DatetimeIndex; long format needs "
                "date/ticker/Close columns"
            )
        close = raw
        open_panel = None

    close = _clean_panel(close, "close prices")
    open_panel = _clean_panel(open_panel, "open prices") if open_panel is not None else None
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        close = close.loc[close.index <= cutoff]
        if open_panel is not None:
            open_panel = open_panel.loc[open_panel.index <= cutoff]
    if close.empty:
        raise ValueError("no price observations remain after the as-of filter")
    return PriceData(close=close, open=open_panel, source=str(source))


def _month_end(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.resample("ME").last()


def frozen_benchmark_targets(
    close: pd.DataFrame,
    spec: BenchmarkSpec = FROZEN_BENCHMARK,
) -> TargetResult:
    """Reproduce the frozen production 12-ETF signal and slot-weight rules.

    The target stamped at month-end ``t`` is never applied to month ``t``.
    ``backtest_next_period`` shifts it into the following holding period.
    """
    daily = _clean_panel(close, "close prices")
    missing = sorted(set(spec.universe) - set(daily.columns))
    if missing:
        raise ValueError(f"frozen benchmark tickers missing: {missing}")
    daily = daily.loc[:, list(spec.universe)]
    monthly = _month_end(daily)
    momentum = (
        monthly.shift(spec.momentum_skip_months)
        / monthly.shift(spec.momentum_lookback_months)
        - 1.0
    )
    above_ma = monthly > monthly.rolling(spec.moving_average_months).mean()
    eligible = (
        monthly.notna()
        .rolling(spec.min_monthly_closes)
        .count()
        .ge(spec.min_monthly_closes)
    )
    signal = momentum.gt(0.0) & above_ma & eligible

    volatility = (
        daily.pct_change(fill_method=None)
        .rolling(spec.volatility_days)
        .std()
        .mul(np.sqrt(252.0))
        .resample("ME")
        .last()
        .clip(lower=spec.volatility_floor)
    )
    inverse = (1.0 / volatility).where(eligible, 0.0)
    slots = inverse.div(inverse.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    slots = slots.clip(upper=spec.asset_weight_cap)
    targets = slots.mul(signal.astype(float)).fillna(0.0)
    return TargetResult(targets=targets, signal_state=signal)


def votes_to_hysteresis(
    votes: pd.DataFrame,
    eligible: pd.DataFrame,
    enter_votes: int,
    exit_votes: int,
) -> pd.DataFrame:
    """Convert vote counts to long/flat states with separate entry/exit bars."""
    if enter_votes <= exit_votes:
        raise ValueError("enter_votes must be greater than exit_votes")
    if not votes.index.equals(eligible.index) or not votes.columns.equals(eligible.columns):
        raise ValueError("votes and eligibility must have identical axes")
    state = pd.DataFrame(False, index=votes.index, columns=votes.columns)
    previous = pd.Series(False, index=votes.columns)
    for when in votes.index:
        available = eligible.loc[when].fillna(False).astype(bool)
        current_votes = votes.loc[when].fillna(0.0)
        now = previous.copy()
        now.loc[~available] = False
        now.loc[available & ~previous & current_votes.ge(enter_votes)] = True
        now.loc[available & previous & current_votes.le(exit_votes)] = False
        state.loc[when] = now
        previous = now
    return state.astype(bool)


def _channel_state(monthly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Long after a prior-window high; exit after a half-window low."""
    prior_high = monthly.rolling(horizon, min_periods=horizon).max().shift(1)
    exit_window = max(2, horizon // 2)
    prior_low = monthly.rolling(exit_window, min_periods=exit_window).min().shift(1)
    enter = monthly.gt(prior_high)
    exit_ = monthly.lt(prior_low)
    available = monthly.notna() & prior_high.notna() & prior_low.notna()
    state = pd.DataFrame(False, index=monthly.index, columns=monthly.columns)
    previous = pd.Series(False, index=monthly.columns)
    for when in monthly.index:
        now = previous.copy()
        valid = available.loc[when]
        now.loc[~valid] = False
        now.loc[valid & ~previous & enter.loc[when]] = True
        now.loc[valid & previous & exit_.loc[when]] = False
        state.loc[when] = now
        previous = now
    return state.astype(bool)


def apply_turnover_controls(
    desired: pd.DataFrame,
    rebalance_band: float,
    max_turnover: float,
) -> pd.DataFrame:
    """Apply a no-trade band and a soft turnover cap to a target path.

    Full exits are never throttled merely to satisfy a turnover budget.  When
    forced exits alone exceed ``max_turnover``, reported turnover can therefore
    exceed the soft cap; discretionary entries/rebalances receive no budget in
    that period.
    """
    if rebalance_band < 0.0 or max_turnover <= 0.0:
        raise ValueError("rebalance_band must be nonnegative and max_turnover positive")
    wanted = desired.fillna(0.0).astype(float)
    controlled = pd.DataFrame(0.0, index=wanted.index, columns=wanted.columns)
    previous = pd.Series(0.0, index=wanted.columns)
    for when in wanted.index:
        goal = wanted.loc[when]
        delta = goal - previous
        forced_exit = goal.abs().lt(1e-14) & previous.abs().ge(1e-14)
        forced_delta = delta.where(forced_exit, 0.0)
        discretionary = delta.where(~forced_exit, 0.0)
        discretionary = discretionary.where(discretionary.abs().ge(rebalance_band), 0.0)
        forced_turnover = float(forced_delta.abs().sum())
        room = max(0.0, max_turnover - forced_turnover)
        discretionary_turnover = float(discretionary.abs().sum())
        if discretionary_turnover > room and discretionary_turnover > 0.0:
            discretionary *= room / discretionary_turnover
        current = previous + forced_delta + discretionary
        current = current.where(current.abs().ge(1e-14), 0.0)
        controlled.loc[when] = current
        previous = current
    return controlled


def _vol_scaled_targets(
    daily: pd.DataFrame,
    active: pd.DataFrame,
    spec: MultiSpeedSpec,
) -> pd.DataFrame:
    returns = daily.pct_change(fill_method=None)
    vol_monthly = (
        returns.rolling(spec.volatility_days)
        .std()
        .mul(np.sqrt(252.0))
        .resample("ME")
        .last()
        .clip(lower=spec.volatility_floor)
        .reindex(active.index)
    )
    desired = pd.DataFrame(0.0, index=active.index, columns=active.columns)
    for when in active.index:
        on = active.loc[when] & vol_monthly.loc[when].notna()
        tickers = list(on.index[on])
        if not tickers:
            continue
        inverse = 1.0 / vol_monthly.loc[when, tickers]
        weights = inverse / inverse.sum() * spec.gross_weight_cap
        weights = weights.clip(upper=spec.asset_weight_cap)

        history = returns.loc[returns.index <= when, tickers].tail(spec.covariance_days)
        covariance = history.cov(min_periods=max(20, spec.volatility_days // 2)).fillna(0.0)
        covariance = covariance.reindex(index=tickers, columns=tickers).fillna(0.0) * 252.0
        for ticker in tickers:
            if covariance.loc[ticker, ticker] <= 0.0:
                covariance.loc[ticker, ticker] = float(vol_monthly.loc[when, ticker] ** 2)
        vector = weights.reindex(tickers).to_numpy(dtype=float)
        portfolio_variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
        portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))
        if portfolio_volatility > 0.0:
            scale = min(1.0, spec.portfolio_target_volatility / portfolio_volatility)
            weights *= scale
        desired.loc[when, tickers] = weights
    return desired


def multispeed_targets(close: pd.DataFrame, spec: MultiSpeedSpec) -> TargetResult:
    """Build one preregistered multi-speed time-series trend target path."""
    daily = _clean_panel(close, "close prices")
    monthly = _month_end(daily)
    max_horizon = max(spec.horizons_months)
    eligible = monthly.notna() & monthly.shift(max_horizon).notna()
    states: list[pd.DataFrame] = []
    if spec.signal_kind == "return_sign":
        for horizon in spec.horizons_months:
            states.append(monthly.div(monthly.shift(horizon)).sub(1.0).gt(0.0))
    elif spec.signal_kind == "channel_breakout":
        states = [_channel_state(monthly, horizon) for horizon in spec.horizons_months]
    else:  # pragma: no cover - dataclass Literal plus defensive runtime check
        raise ValueError(f"unsupported signal kind: {spec.signal_kind}")
    votes = sum(state.astype(int) for state in states)
    signal = votes_to_hysteresis(
        votes=votes,
        eligible=eligible,
        enter_votes=spec.enter_votes,
        exit_votes=spec.exit_votes,
    )
    desired = _vol_scaled_targets(daily=daily, active=signal, spec=spec)
    targets = apply_turnover_controls(
        desired,
        rebalance_band=spec.rebalance_band,
        max_turnover=spec.max_monthly_turnover,
    )
    return TargetResult(targets=targets, signal_state=signal, votes=votes)


def next_period_asset_returns(
    close: pd.DataFrame,
    open_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return monthly holding-period returns with no same-period signal use.

    With opens, row ``M`` is first-open(M+1)/first-open(M)-1.  A target formed
    at month-end M-1 is shifted into row M.  Without opens, the conservative
    fallback is close(M)/close(M-1)-1, again earned by the prior target.
    """
    daily_close = _clean_panel(close, "close prices")
    if open_prices is not None:
        daily_open = _clean_panel(open_prices, "open prices").reindex(columns=daily_close.columns)
        first_open = daily_open.resample("ME").first()
        return first_open.shift(-1).div(first_open).sub(1.0), "next_open_to_next_open"
    monthly = _month_end(daily_close)
    return monthly.pct_change(fill_method=None), "next_close_to_next_close"


def backtest_next_period(
    targets: pd.DataFrame,
    close: pd.DataFrame,
    open_prices: pd.DataFrame | None = None,
    cost_bps_per_side: float = 5.0,
    cash_returns: pd.Series | None = None,
) -> BacktestResult:
    """Apply every target one period later and subtract turnover costs."""
    if cost_bps_per_side < 0.0:
        raise ValueError("cost_bps_per_side cannot be negative")
    weights = targets.fillna(0.0).astype(float).copy()
    weights.index = pd.to_datetime(weights.index)
    weights = weights.sort_index()
    asset_returns, execution = next_period_asset_returns(close, open_prices)
    asset_returns = asset_returns.reindex(index=weights.index, columns=weights.columns)
    held = weights.shift(1).fillna(0.0)

    changes = weights.diff()
    if len(changes):
        changes.iloc[0] = weights.iloc[0]
    turnover = changes.abs().sum(axis=1).shift(1).fillna(0.0)
    missing_held_return = (held.abs().gt(1e-14) & asset_returns.isna()).any(axis=1)
    usable = asset_returns.notna().any(axis=1) & ~missing_held_return
    # Warm-up rows before the first actionable target are not observations of
    # the strategy.  Keep later all-cash months, which are genuine signal
    # outcomes, but do not let years of pre-eligibility zeroes flatter risk.
    actionable = weights.abs().sum(axis=1).gt(1e-14)
    if actionable.any():
        first_target_position = int(np.flatnonzero(actionable.to_numpy())[0])
        first_holding_position = first_target_position + 1
        if first_holding_position < len(weights):
            usable.iloc[:first_holding_position] = False
        else:
            usable.iloc[:] = False
    else:
        usable.iloc[:] = False

    cash = (
        pd.Series(0.0, index=weights.index, name="cash_return")
        if cash_returns is None
        else cash_returns.reindex(weights.index).fillna(0.0).rename("cash_return")
    )
    asset_pnl = held.mul(asset_returns.fillna(0.0)).sum(axis=1)
    cash_weight = 1.0 - held.sum(axis=1)
    gross = asset_pnl + cash_weight * cash
    cost = turnover * (cost_bps_per_side / 10_000.0)
    monthly = pd.DataFrame(
        {
            "gross_return": gross,
            "net_return": gross - cost,
            "cash_return": cash,
            "turnover": turnover,
            "cost": cost,
            "gross_exposure": held.abs().sum(axis=1),
            "net_exposure": held.sum(axis=1),
            "cash_weight": cash_weight,
            "execution": execution,
        }
    ).loc[usable]
    return BacktestResult(monthly=monthly, targets=weights, held_weights=held)


def _compound_cagr(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    terminal = float((1.0 + clean).prod())
    years = len(clean) / 12.0
    return terminal ** (1.0 / years) - 1.0 if terminal > 0.0 and years > 0.0 else np.nan


def _max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    curve = (1.0 + clean).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    volatility = float(clean.std())
    return float(clean.mean() / volatility * np.sqrt(12.0)) if volatility > 0.0 else np.nan


def performance_summary(
    result: BacktestResult,
    benchmark_net: pd.Series | None = None,
) -> dict[str, Any]:
    """Gross/net, drawdown, turnover, cost and benchmark-comparison metrics."""
    monthly = result.monthly
    gross = monthly["gross_return"]
    net = monthly["net_return"]
    cash = monthly.get("cash_return", pd.Series(0.0, index=monthly.index))
    summary: dict[str, Any] = {
        "months": int(len(monthly)),
        "start": str(monthly.index.min().date()) if len(monthly) else None,
        "end": str(monthly.index.max().date()) if len(monthly) else None,
        "gross_cagr": _compound_cagr(gross),
        "net_cagr": _compound_cagr(net),
        "gross_annual_volatility": float(gross.std() * np.sqrt(12.0)),
        "net_annual_volatility": float(net.std() * np.sqrt(12.0)),
        "gross_sharpe": _sharpe(gross - cash),
        "net_sharpe": _sharpe(net - cash),
        "gross_max_drawdown": _max_drawdown(gross),
        "net_max_drawdown": _max_drawdown(net),
        "monthly_hit_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
        "average_monthly_turnover": float(monthly["turnover"].mean()),
        "annualized_turnover": float(monthly["turnover"].mean() * 12.0),
        "cumulative_cost": float(monthly["cost"].sum()),
        "average_gross_exposure": float(monthly["gross_exposure"].mean()),
        "worst_month": float(net.min()) if len(net) else np.nan,
    }
    if benchmark_net is not None:
        comparison = pd.concat(
            [net.rename("trial"), benchmark_net.rename("benchmark")], axis=1
        ).dropna()
        summary["correlation_to_frozen_benchmark"] = (
            float(comparison.corr().iloc[0, 1]) if len(comparison) >= 2 else np.nan
        )
        summary["common_benchmark_months"] = int(len(comparison))
    else:
        summary["correlation_to_frozen_benchmark"] = np.nan
        summary["common_benchmark_months"] = 0
    return summary


def residualized_returns(
    stock_close: pd.DataFrame,
    market_close: pd.Series,
    beta_lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PIT residuals using betas estimated strictly through the prior close."""
    stocks = _clean_panel(stock_close, "stock closes")
    market = pd.to_numeric(market_close, errors="coerce").copy()
    market.index = pd.to_datetime(market.index)
    if getattr(market.index, "tz", None) is not None:
        market.index = market.index.tz_localize(None)
    market = market.sort_index().reindex(stocks.index)
    stock_returns = stocks.pct_change(fill_method=None)
    market_returns = market.pct_change(fill_method=None)
    market_variance = market_returns.rolling(
        beta_lookback_days, min_periods=beta_lookback_days
    ).var().shift(1)
    betas = pd.DataFrame(index=stocks.index, columns=stocks.columns, dtype=float)
    for ticker in stocks.columns:
        covariance = stock_returns[ticker].rolling(
            beta_lookback_days, min_periods=beta_lookback_days
        ).cov(market_returns).shift(1)
        betas[ticker] = covariance.div(market_variance.replace(0.0, np.nan))
    residuals = stock_returns - betas.mul(market_returns, axis=0)
    return residuals, betas


def normalize_sector_history(
    sectors: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """Materialize PIT sector membership at signal dates without backfilling.

    Accepted inputs are dated snapshots (long ``date/ticker/sector`` or wide
    DatetimeIndex) or effective intervals
    (``ticker/sector/effective_from/effective_to``).  Static undated maps are
    rejected because they silently apply today's classification to history.
    """
    if not isinstance(sectors, pd.DataFrame) or sectors.empty:
        raise ValueError("sector history must be a non-empty dated DataFrame")
    lower = {str(column).lower(): column for column in sectors.columns}
    signal_dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).sort_values()
    ticker_set = {ticker.upper() for ticker in tickers}

    if {"date", "ticker", "sector"}.issubset(lower):
        frame = sectors[[lower["date"], lower["ticker"], lower["sector"]]].copy()
        frame.columns = ["date", "ticker", "sector"]
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame = frame[frame["ticker"].isin(ticker_set)]
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError("sector snapshots contain duplicate date/ticker rows")
        wide = frame.pivot(index="date", columns="ticker", values="sector")
        combined = wide.reindex(wide.index.union(signal_dates)).sort_index().ffill()
        return combined.reindex(index=signal_dates, columns=tickers)

    if {"ticker", "sector", "effective_from"}.issubset(lower):
        panel = pd.DataFrame(index=signal_dates, columns=tickers, dtype=object)
        history = sectors.copy()
        for _, row in history.iterrows():
            ticker = str(row[lower["ticker"]]).upper().strip()
            if ticker not in ticker_set:
                continue
            start = pd.Timestamp(row[lower["effective_from"]]).tz_localize(None).normalize()
            end_value = row[lower["effective_to"]] if "effective_to" in lower else pd.NaT
            end = (
                pd.Timestamp(end_value).tz_localize(None).normalize()
                if pd.notna(end_value)
                else signal_dates.max()
            )
            mask = (signal_dates >= start) & (signal_dates <= end)
            existing = panel.loc[mask, ticker].dropna()
            if not existing.empty:
                raise ValueError(f"overlapping sector intervals for {ticker}")
            panel.loc[mask, ticker] = row[lower["sector"]]
        return panel

    if isinstance(sectors.index, pd.DatetimeIndex):
        wide = sectors.copy()
        wide.index = pd.to_datetime(wide.index).tz_localize(None).normalize()
        wide.columns = [str(column).upper().strip() for column in wide.columns]
        combined = wide.reindex(wide.index.union(signal_dates)).sort_index().ffill()
        return combined.reindex(index=signal_dates, columns=tickers)

    raise ValueError(
        "sector history must be dated snapshots, effective intervals, or a wide dated panel"
    )


def sector_neutral_percentile_ranks(
    scores: pd.DataFrame,
    sector_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Rank each stock only against contemporaneous members of its sector."""
    if not scores.index.equals(sector_panel.index) or not scores.columns.equals(
        sector_panel.columns
    ):
        raise ValueError("scores and sector_panel must have identical axes")
    ranks = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    for when in scores.index:
        row = pd.DataFrame(
            {"score": scores.loc[when], "sector": sector_panel.loc[when]}
        ).dropna()
        for _, group in row.groupby("sector", sort=False):
            ranks.loc[when, group.index] = group["score"].rank(
                method="average", pct=True
            )
    return ranks.astype(float)


def cross_sectional_targets(
    stock_close: pd.DataFrame,
    market_close: pd.Series,
    sector_history: pd.DataFrame,
    spec: CrossSectionalSpec,
) -> TargetResult:
    """Build the separate stock residual-trend, sector-neutral research family."""
    daily = _clean_panel(stock_close, "stock closes")
    residuals, _ = residualized_returns(
        stock_close=daily,
        market_close=market_close,
        beta_lookback_days=spec.beta_lookback_days,
    )
    score_daily = residuals.rolling(
        spec.residual_momentum_days,
        min_periods=spec.residual_momentum_days,
    ).sum()
    scores = score_daily.resample("ME").last()
    sector_panel = normalize_sector_history(
        sectors=sector_history,
        dates=scores.index,
        tickers=list(scores.columns),
    )
    ranks = sector_neutral_percentile_ranks(scores, sector_panel)
    history_ok = (
        daily.notna()
        .rolling(spec.min_history_days)
        .sum()
        .ge(spec.min_history_days)
        .resample("ME")
        .last()
        .reindex(scores.index)
    )
    vol = (
        daily.pct_change(fill_method=None)
        .rolling(spec.volatility_days)
        .std()
        .mul(np.sqrt(252.0))
        .resample("ME")
        .last()
        .reindex(scores.index)
    )
    desired = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    signal = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    for when in scores.index:
        valid = history_ok.loc[when] & ranks.loc[when].notna() & vol.loc[when].gt(0.0)
        top = valid & ranks.loc[when].ge(spec.top_quantile)
        bottom = valid & ranks.loc[when].le(spec.bottom_quantile)
        sectors_with_longs = sorted(set(sector_panel.loc[when, top].dropna()))
        if spec.long_short:
            sectors_with_shorts = set(sector_panel.loc[when, bottom].dropna())
            active_sectors = [sector for sector in sectors_with_longs if sector in sectors_with_shorts]
            side_budget = spec.gross_weight_cap / 2.0
        else:
            active_sectors = sectors_with_longs
            side_budget = spec.gross_weight_cap
        if not active_sectors:
            continue
        sector_budget = side_budget / len(active_sectors)
        for sector in active_sectors:
            long_names = list(top.index[top & sector_panel.loc[when].eq(sector)])
            if long_names:
                inv = 1.0 / vol.loc[when, long_names]
                weights = (inv / inv.sum() * sector_budget).clip(
                    upper=spec.asset_weight_cap
                )
                desired.loc[when, long_names] = weights
                signal.loc[when, long_names] = True
            if spec.long_short:
                short_names = list(bottom.index[bottom & sector_panel.loc[when].eq(sector)])
                if short_names:
                    inv = 1.0 / vol.loc[when, short_names]
                    weights = (inv / inv.sum() * sector_budget).clip(
                        upper=spec.asset_weight_cap
                    )
                    desired.loc[when, short_names] = -weights
                    signal.loc[when, short_names] = True
    targets = apply_turnover_controls(
        desired,
        rebalance_band=spec.rebalance_band,
        max_turnover=spec.max_monthly_turnover,
    )
    return TargetResult(
        targets=targets,
        signal_state=signal,
        scores=scores,
        ranks=ranks,
    )
