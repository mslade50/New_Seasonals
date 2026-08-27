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

from trading_calendar import TRADING_DAY

from .config import (
    FROZEN_BENCHMARK,
    BenchmarkSpec,
    CrossSectionalSpec,
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
    desired_targets: pd.DataFrame
    signal_state: pd.DataFrame
    votes: pd.DataFrame | None = None
    scores: pd.DataFrame | None = None
    ranks: pd.DataFrame | None = None
    sector_panel: pd.DataFrame | None = None
    membership_panel: pd.DataFrame | None = None

    @property
    def targets(self) -> pd.DataFrame:
        """Compatibility alias; these are desired, not executed, weights."""
        return self.desired_targets


@dataclass(frozen=True)
class BacktestResult:
    monthly: pd.DataFrame
    desired_targets: pd.DataFrame
    executed_weights: pd.DataFrame
    pretrade_weights: pd.DataFrame

    @property
    def targets(self) -> pd.DataFrame:
        return self.desired_targets

    @property
    def held_weights(self) -> pd.DataFrame:
        return self.executed_weights


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


def exact_month_end_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Sample only the exact final NYSE session of each calendar month.

    ``resample(...).last()`` silently substitutes an earlier observation when
    the actual month-end value is missing. Month-end signals must not inherit
    that stale-data fallback.
    """

    if panel.empty:
        return panel.copy()
    periods = pd.period_range(
        panel.index.min().to_period("M"), panel.index.max().to_period("M"), freq="M"
    )
    labels = periods.to_timestamp("M")
    sessions = pd.DatetimeIndex(
        [_session_for_month(period, first=False) for period in periods]
    )
    sampled = panel.reindex(sessions).copy()
    sampled.index = labels
    return sampled


def validate_required_daily_sessions(panel: pd.DataFrame, label: str) -> None:
    """Fail closed on missing NYSE-session closes inside a required panel."""

    if panel.empty:
        raise ValueError(f"{label} is empty")
    required_end = pd.Timestamp(panel.index.max()).normalize()
    failures: list[str] = []
    for ticker in panel.columns:
        series = panel[ticker].dropna()
        if series.empty:
            failures.append(f"{ticker}: no observations")
            continue
        expected = pd.date_range(series.index.min(), required_end, freq=TRADING_DAY)
        missing = expected.difference(series.index)
        if len(missing):
            preview = ",".join(str(stamp.date()) for stamp in missing[:3])
            failures.append(f"{ticker}: {len(missing)} missing ({preview})")
    if failures:
        raise ValueError(
            f"{label} has missing required daily closes; refusing stale/forward-filled "
            "signal inputs: " + "; ".join(failures)
        )


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
    validate_required_daily_sessions(daily, "frozen benchmark prices")
    monthly = exact_month_end_panel(daily)
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
        .pipe(exact_month_end_panel)
        .clip(lower=spec.volatility_floor)
    )
    inverse = (1.0 / volatility).where(eligible, 0.0)
    slots = inverse.div(inverse.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    slots = slots.clip(upper=spec.asset_weight_cap)
    targets = slots.mul(signal.astype(float)).fillna(0.0)
    return TargetResult(desired_targets=targets, signal_state=signal)


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


def _project_hard_caps(
    weights: pd.Series,
    asset_weight_cap: float,
    gross_weight_cap: float,
) -> pd.Series:
    """Project weights into hard name/gross limits without changing signs."""
    if asset_weight_cap <= 0.0 or gross_weight_cap <= 0.0:
        raise ValueError("hard name and gross caps must be positive")
    projected = weights.fillna(0.0).astype(float).clip(
        lower=-asset_weight_cap, upper=asset_weight_cap
    )
    gross = float(projected.abs().sum())
    if gross > gross_weight_cap:
        projected *= gross_weight_cap / gross
    return projected.where(projected.abs().ge(1e-14), 0.0)


def transition_from_drifted_weights(
    pretrade: pd.Series,
    desired: pd.Series,
    rebalance_band: float,
    max_turnover: float | None,
    asset_weight_cap: float,
    gross_weight_cap: float,
    force_signal_flips: bool = True,
) -> tuple[pd.Series, float, float]:
    """Execute toward a desired target from drifted weights.

    Hard name/gross-limit corrections happen first and may exceed the soft
    turnover budget.  The no-trade band and remaining turnover budget then
    apply to the discretionary move toward the hard-capped desired target.
    Returns ``(posttrade, total_turnover, mandatory_cap_turnover)``.
    """
    if rebalance_band < 0.0:
        raise ValueError("rebalance_band cannot be negative")
    if max_turnover is not None and max_turnover <= 0.0:
        raise ValueError("max_turnover must be positive or None")
    if not pretrade.index.equals(desired.index):
        raise ValueError("pretrade and desired weights must have identical tickers")

    raw_pretrade = pretrade.fillna(0.0).astype(float)
    base = _project_hard_caps(raw_pretrade, asset_weight_cap, gross_weight_cap)
    mandatory_delta = base - raw_pretrade
    mandatory_turnover = float(mandatory_delta.abs().sum())

    goal = _project_hard_caps(desired, asset_weight_cap, gross_weight_cap)
    discretionary = goal - base
    flips = base.abs().lt(1e-14) != goal.abs().lt(1e-14)
    discretionary = discretionary.where(
        discretionary.abs().ge(rebalance_band) | (flips if force_signal_flips else False),
        0.0,
    )
    discretionary_turnover = float(discretionary.abs().sum())
    if max_turnover is not None:
        room = max(0.0, max_turnover - mandatory_turnover)
        if discretionary_turnover > room and discretionary_turnover > 0.0:
            discretionary *= room / discretionary_turnover
    posttrade = base + discretionary
    # Convex movement between two valid points is cap-safe, but project again
    # for floating-point protection and count any correction as mandatory.
    final = _project_hard_caps(posttrade, asset_weight_cap, gross_weight_cap)
    final_correction = final - posttrade
    mandatory_turnover += float(final_correction.abs().sum())
    turnover = float((final - raw_pretrade).abs().sum())
    if float(final.abs().max()) > asset_weight_cap + 1e-12:
        raise AssertionError("posttrade name cap violated")
    if float(final.abs().sum()) > gross_weight_cap + 1e-12:
        raise AssertionError("posttrade gross cap violated")
    return final, turnover, mandatory_turnover


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
        .pipe(exact_month_end_panel)
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
    validate_required_daily_sessions(daily, "multi-speed benchmark prices")
    monthly = exact_month_end_panel(daily)
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
    return TargetResult(desired_targets=desired, signal_state=signal, votes=votes)


def next_period_asset_returns(
    close: pd.DataFrame,
    open_prices: pd.DataFrame | None = None,
    period_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return exact-session monthly returns without silently using later bars."""
    returns, execution, _ = _exact_period_returns(
        close=close,
        open_prices=open_prices,
        period_index=period_index,
    )
    return returns, execution


def _session_for_month(month: pd.Period, first: bool) -> pd.Timestamp:
    sessions = pd.date_range(
        month.start_time.normalize(), month.end_time.normalize(), freq=TRADING_DAY
    )
    if sessions.empty:  # pragma: no cover - defensive against a broken calendar
        raise ValueError(f"NYSE calendar has no session in {month}")
    return pd.Timestamp(sessions[0] if first else sessions[-1]).normalize()


def _normalized_contiguous_months(index: pd.Index) -> pd.DatetimeIndex:
    raw = pd.DatetimeIndex(pd.to_datetime(index))
    if raw.has_duplicates:
        raise ValueError("desired targets contain duplicate months")
    periods = raw.to_period("M")
    if periods.duplicated().any():
        raise ValueError("desired targets contain multiple rows in one month")
    if len(periods) > 1:
        expected = pd.period_range(periods.min(), periods.max(), freq="M")
        if not periods.equals(expected):
            raise ValueError("desired target months must be contiguous; no time compression")
    return periods.to_timestamp("M")


def _exact_period_returns(
    close: pd.DataFrame,
    open_prices: pd.DataFrame | None,
    period_index: pd.DatetimeIndex | None,
) -> tuple[pd.DataFrame, str, pd.Series]:
    daily_close = _clean_panel(close, "close prices")
    months = (
        _normalized_contiguous_months(period_index)
        if period_index is not None
        else _normalized_contiguous_months(exact_month_end_panel(daily_close).index)
    )
    periods = months.to_period("M")
    if open_prices is not None:
        panel = _clean_panel(open_prices, "open prices").reindex(
            columns=daily_close.columns
        )
        execution = "next_open_to_next_open"
        first = True
    else:
        panel = daily_close
        execution = "next_close_to_next_close"
        first = False

    current_dates = pd.Series(
        [_session_for_month(month, first=first) for month in periods], index=months
    )
    next_dates = pd.Series(
        [_session_for_month(month + 1, first=first) for month in periods], index=months
    )
    # A trailing period whose next boundary lies beyond the source is simply
    # not complete yet. Any missing boundary inside source coverage is retained
    # as NaN and will fail if the simulator needs that held return.
    complete = next_dates.le(panel.index.max())
    if (~complete).any():
        first_incomplete = int(np.flatnonzero((~complete).to_numpy())[0])
        if complete.iloc[first_incomplete:].any():
            raise ValueError("non-terminal incomplete months would compress time")
    complete_months = months[complete.to_numpy()]
    current_dates = current_dates.loc[complete_months]
    next_dates = next_dates.loc[complete_months]
    current_prices = panel.reindex(pd.DatetimeIndex(current_dates.values))
    next_prices = panel.reindex(pd.DatetimeIndex(next_dates.values))
    current_prices.index = complete_months
    next_prices.index = complete_months
    returns = next_prices.div(current_prices).sub(1.0)
    boundary = pd.Series(
        [f"{a.date()}->{b.date()}" for a, b in zip(current_dates, next_dates)],
        index=complete_months,
        name="execution_boundary",
    )
    return returns, execution, boundary


def backtest_next_period(
    targets: pd.DataFrame,
    close: pd.DataFrame,
    open_prices: pd.DataFrame | None = None,
    cost_bps_per_side: float = 5.0,
    cash_returns: pd.Series | None = None,
    rebalance_band: float = 0.0,
    max_turnover: float | None = None,
    asset_weight_cap: float = 1.0,
    gross_weight_cap: float = 1.0,
    force_signal_flips: bool = True,
) -> BacktestResult:
    """Drift-aware next-period execution and cost simulator.

    Desired target ``M-1`` executes at the boundary starting month ``M``.
    Existing positions first drift through their prior holding-period returns;
    band/soft-turnover decisions are made against those drifted weights. Hard
    name/gross caps always win over the soft turnover budget.
    """
    if cost_bps_per_side < 0.0:
        raise ValueError("cost_bps_per_side cannot be negative")
    desired_all = targets.fillna(0.0).astype(float).copy().sort_index()
    desired_all.index = _normalized_contiguous_months(desired_all.index)
    desired_all.columns = [str(column).upper().strip() for column in desired_all.columns]
    if desired_all.columns.duplicated().any():
        raise ValueError("desired targets contain duplicate tickers")
    asset_returns, execution, boundaries = _exact_period_returns(
        close=close,
        open_prices=open_prices,
        period_index=desired_all.index,
    )
    asset_returns = asset_returns.reindex(columns=desired_all.columns)
    complete_months = asset_returns.index
    desired = desired_all.reindex(complete_months)
    cash = (
        pd.Series(0.0, index=complete_months, name="cash_return")
        if cash_returns is None
        else cash_returns.reindex(complete_months).fillna(0.0).rename("cash_return")
    )

    def empty_result() -> BacktestResult:
        empty_index = pd.DatetimeIndex([], name=desired.index.name)
        empty_monthly = pd.DataFrame(
            columns=[
                "gross_return", "net_return", "cash_return", "turnover",
                "mandatory_cap_turnover", "cost", "gross_exposure",
                "net_exposure", "cash_weight", "execution",
                "execution_boundary",
            ],
            index=empty_index,
        )
        blank = pd.DataFrame(index=empty_index, columns=desired.columns, dtype=float)
        return BacktestResult(empty_monthly, desired_all, blank, blank.copy())

    # Start every strategy at the first next-period boundary. Pre-signal
    # months stay in cash, preserving delayed activation as opportunity cost
    # and giving the ETF family a common evaluation clock.
    first_holding_position = 1
    if first_holding_position >= len(complete_months):
        return empty_result()

    simulation_months = complete_months[first_holding_position:]
    executed = pd.DataFrame(0.0, index=simulation_months, columns=desired.columns)
    pretrade_frame = pd.DataFrame(0.0, index=simulation_months, columns=desired.columns)
    records: list[dict[str, Any]] = []
    pretrade = pd.Series(0.0, index=desired.columns)
    rate = cost_bps_per_side / 10_000.0
    for position, month in enumerate(simulation_months, start=first_holding_position):
        goal = desired.iloc[position - 1]
        posttrade, turnover, mandatory_turnover = transition_from_drifted_weights(
            pretrade=pretrade,
            desired=goal,
            rebalance_band=rebalance_band,
            max_turnover=max_turnover,
            asset_weight_cap=asset_weight_cap,
            gross_weight_cap=gross_weight_cap,
            force_signal_flips=force_signal_flips,
        )
        period_returns = asset_returns.loc[month]
        missing = posttrade.abs().gt(1e-14) & period_returns.isna()
        if missing.any():
            names = list(missing.index[missing])
            boundary = boundaries.loc[month]
            kind = "first-session open" if open_prices is not None else "month-end close"
            raise ValueError(
                f"missing exact {kind}/held return for {names} at {boundary}; "
                "refusing delayed-bar substitution or month dropping"
            )
        period_returns = period_returns.fillna(0.0)
        cash_return = float(cash.loc[month])
        cash_weight = 1.0 - float(posttrade.sum())
        asset_pnl = float((posttrade * period_returns).sum())
        gross_return = asset_pnl + cash_weight * cash_return
        cost = turnover * rate
        net_return = gross_return - cost
        ending_nav = 1.0 + net_return
        if ending_nav <= 0.0:
            raise ValueError(f"portfolio NAV is non-positive in {month.date()}")

        executed.loc[month] = posttrade
        pretrade_frame.loc[month] = pretrade
        records.append(
            {
                "month": month,
                "gross_return": gross_return,
                "net_return": net_return,
                "cash_return": cash_return,
                "turnover": turnover,
                "mandatory_cap_turnover": mandatory_turnover,
                "cost": cost,
                "gross_exposure": float(posttrade.abs().sum()),
                "net_exposure": float(posttrade.sum()),
                "cash_weight": cash_weight,
                "execution": execution,
                "execution_boundary": boundaries.loc[month],
            }
        )
        # Cost is paid from cash at execution. Asset notionals then drift with
        # their realized returns and are normalized by ending net NAV.
        pretrade = posttrade.mul(1.0 + period_returns).div(ending_nav)

    monthly = pd.DataFrame(records).set_index("month")
    return BacktestResult(
        monthly=monthly,
        desired_targets=desired_all,
        executed_weights=executed,
        pretrade_weights=pretrade_frame,
    )


def _compound_cagr(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    terminal = float((1.0 + clean).prod())
    first_start = clean.index[0].to_period("M").start_time
    last_end = (clean.index[-1].to_period("M") + 1).start_time
    years = (last_end - first_start).days / 365.2425
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
        "months": len(monthly),
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
        "annualized_turnover": (
            float(
                monthly["turnover"].sum()
                / max(
                    (
                        (monthly.index[-1].to_period("M") + 1).start_time
                        - monthly.index[0].to_period("M").start_time
                    ).days
                    / 365.2425,
                    1.0 / 12.0,
                )
            )
            if len(monthly)
            else np.nan
        ),
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
        summary["common_benchmark_months"] = len(comparison)
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


def normalize_membership_history(
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """Materialize explicit historical universe membership without backfill."""
    if not isinstance(membership, pd.DataFrame) or membership.empty:
        raise ValueError("historical membership must be a non-empty dated DataFrame")
    lower = {str(column).lower(): column for column in membership.columns}
    signal_dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).sort_values()
    ticker_set = {ticker.upper() for ticker in tickers}
    value_name = next(
        (name for name in ("in_universe", "member", "eligible") if name in lower),
        None,
    )

    def coerce_bool(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        def one(value: Any) -> bool | float:
            if pd.isna(value):
                return np.nan
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(value)
            normalized = str(value).strip().lower()
            if normalized in {"true", "t", "yes", "y", "1"}:
                return True
            if normalized in {"false", "f", "no", "n", "0"}:
                return False
            raise ValueError(f"unrecognized membership boolean: {value!r}")

        return values.map(one)

    if {"date", "ticker"}.issubset(lower) and value_name is not None:
        frame = membership[
            [lower["date"], lower["ticker"], lower[value_name]]
        ].copy()
        frame.columns = ["date", "ticker", "in_universe"]
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame["in_universe"] = coerce_bool(frame["in_universe"])
        frame = frame[frame["ticker"].isin(ticker_set)]
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError("membership snapshots contain duplicate date/ticker rows")
        wide = frame.pivot(
            index="date", columns="ticker", values="in_universe"
        ).astype("boolean")
        combined = wide.reindex(wide.index.union(signal_dates)).sort_index().ffill()
        return (
            combined.reindex(index=signal_dates, columns=tickers)
            .eq(True)
            .fillna(False)
            .astype(bool)
        )

    if {"ticker", "effective_from"}.issubset(lower):
        panel = pd.DataFrame(False, index=signal_dates, columns=tickers)
        for _, row in membership.iterrows():
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
            if panel.loc[mask, ticker].any():
                raise ValueError(f"overlapping membership intervals for {ticker}")
            panel.loc[mask, ticker] = True
        return panel.astype(bool)

    if isinstance(membership.index, pd.DatetimeIndex):
        wide = membership.copy()
        wide.index = pd.to_datetime(wide.index).tz_localize(None).normalize()
        wide.columns = [str(column).upper().strip() for column in wide.columns]
        wide = coerce_bool(wide).astype("boolean")
        combined = wide.reindex(wide.index.union(signal_dates)).sort_index().ffill()
        return (
            combined.reindex(index=signal_dates, columns=tickers)
            .eq(True)
            .fillna(False)
            .astype(bool)
        )

    raise ValueError(
        "membership must be dated bool snapshots, effective intervals, or a wide dated panel"
    )


def capped_proportional_weights(
    preference: pd.Series,
    budget: float,
    asset_cap: float,
) -> pd.Series:
    """Water-fill a positive budget without clip-and-abandon distortion."""
    if budget < -1e-14 or asset_cap <= 0.0:
        raise ValueError("budget must be nonnegative and asset cap positive")
    preference = preference.dropna().astype(float)
    preference = preference.where(preference > 0.0, 0.0)
    result = pd.Series(0.0, index=preference.index)
    feasible = min(max(0.0, budget), len(preference) * asset_cap)
    remaining = feasible
    active = list(preference.index)
    while active and remaining > 1e-14:
        pref = preference.loc[active]
        if float(pref.sum()) <= 0.0:
            pref = pd.Series(1.0, index=active)
        proposal = pref / pref.sum() * remaining
        capped = list(proposal.index[proposal.gt(asset_cap + 1e-14)])
        if not capped:
            result.loc[active] += proposal
            remaining = 0.0
            break
        for ticker in capped:
            room = asset_cap - result.loc[ticker]
            add = max(0.0, room)
            result.loc[ticker] += add
            remaining -= add
            active.remove(ticker)
    if abs(float(result.sum()) - feasible) > 1e-10:
        raise AssertionError("capped allocation failed to use feasible budget")
    if len(result) and float(result.max()) > asset_cap + 1e-12:
        raise AssertionError("capped allocation violated asset cap")
    return result


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
    membership_history: pd.DataFrame | None = None,
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
    scores = exact_month_end_panel(score_daily)
    sector_panel = normalize_sector_history(
        sectors=sector_history,
        dates=scores.index,
        tickers=list(scores.columns),
    )
    membership_panel = (
        normalize_membership_history(
            membership=membership_history,
            dates=scores.index,
            tickers=list(scores.columns),
        )
        if membership_history is not None
        else pd.DataFrame(True, index=scores.index, columns=scores.columns)
    )
    ranks = sector_neutral_percentile_ranks(
        scores.where(membership_panel), sector_panel
    )
    history_ok = (
        daily.notna()
        .rolling(spec.min_history_days)
        .sum()
        .ge(spec.min_history_days)
        .pipe(exact_month_end_panel)
        .reindex(scores.index)
    )
    vol = (
        daily.pct_change(fill_method=None)
        .rolling(spec.volatility_days)
        .std()
        .mul(np.sqrt(252.0))
        .pipe(exact_month_end_panel)
        .reindex(scores.index)
    )
    desired = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    signal = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    for when in scores.index:
        valid = (
            history_ok.loc[when]
            & membership_panel.loc[when]
            & ranks.loc[when].notna()
            & vol.loc[when].gt(0.0)
        )
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
        long_groups = {
            sector: list(top.index[top & sector_panel.loc[when].eq(sector)])
            for sector in active_sectors
        }
        short_groups = {
            sector: list(bottom.index[bottom & sector_panel.loc[when].eq(sector)])
            for sector in active_sectors
        }
        nominal_sector_budget = side_budget / len(active_sectors)
        capacities = [len(long_groups[sector]) * spec.asset_weight_cap for sector in active_sectors]
        if spec.long_short:
            capacities.extend(
                len(short_groups[sector]) * spec.asset_weight_cap
                for sector in active_sectors
            )
        # Every active sector receives the same feasible budget. A narrow
        # sector reduces all sectors instead of letting broad sectors dominate.
        sector_budget = min(nominal_sector_budget, min(capacities))
        for sector in active_sectors:
            long_names = long_groups[sector]
            if long_names:
                inv = 1.0 / vol.loc[when, long_names]
                weights = capped_proportional_weights(
                    inv, budget=sector_budget, asset_cap=spec.asset_weight_cap
                )
                desired.loc[when, long_names] = weights
                signal.loc[when, long_names] = True
            if spec.long_short:
                short_names = short_groups[sector]
                if short_names:
                    inv = 1.0 / vol.loc[when, short_names]
                    weights = capped_proportional_weights(
                        inv, budget=sector_budget, asset_cap=spec.asset_weight_cap
                    )
                    desired.loc[when, short_names] = -weights
                    signal.loc[when, short_names] = True
    return TargetResult(
        desired_targets=desired,
        signal_state=signal,
        scores=scores,
        ranks=ranks,
        sector_panel=sector_panel,
        membership_panel=membership_panel if membership_history is not None else None,
    )
