"""Streaming, local-file runner for the fixed intraday v0 hypotheses.

The runner loads market and sector proxies once, reduces them to daily event
features, and then processes exactly one candidate parquet at a time.  It has
no network, R2, broker, order, production-state, or promotion dependency.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from trading_calendar import TRADING_DAY

from .data import BAR_DELTA, normalize_bars
from .diagnostics import (
    DEFAULT_COST_GRID_BPS,
    PRIMARY_COST_BPS,
    annual_diagnostics,
    capacity_overlays,
    concentration_summaries,
    day_cluster_statistics,
    make_daily_returns,
    materialize_cost_grid,
    rolling_five_year_train_one_year_test,
    validate_cost_grid,
)
from .eligibility import EligibilityConfig
from .simulator import simulate_fixed_time_signals_audited
from .templates import (
    DEFAULT_SECTOR_PROXIES,
    GAP_FIRST_HOUR_TEMPLATE_ID,
    INTRADAY_SHOCK_TEMPLATE_ID,
    GapFirstHourConfig,
    IntradayShockConfig,
)

FULL_SESSION_BAR_COUNT: Final[int] = 26
EARLY_CLOSE_BAR_COUNT: Final[int] = 14
SESSION_OPEN_MINUTE: Final[int] = 9 * 60 + 30
FULL_SESSION_LAST_MINUTE: Final[int] = 15 * 60 + 45
EARLY_CLOSE_LAST_MINUTE: Final[int] = 12 * 60 + 45
GAP_FEATURE_LAST_MINUTE: Final[int] = 10 * 60 + 15
SHOCK_FEATURE_LAST_MINUTE: Final[int] = 13 * 60

SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "template_id",
    "ticker",
    "sector",
    "sector_proxy",
    "trade_date",
    "side",
    "decision_ts",
    "feature_bar_ts",
    "feature_available_ts",
    "entry_bar_ts",
    "entry_ts",
    "exit_bar_ts",
    "exit_ts",
    "asset_gap",
    "market_gap",
    "sector_gap",
    "residual_gap",
    "asset_first_hour",
    "market_first_hour",
    "sector_first_hour",
    "residual_first_hour",
    "asset_shock",
    "market_shock",
    "sector_shock",
    "residual_shock",
    "signal_strength",
    "price_proxy",
    "median_dollar_volume",
    "data_completeness",
)


@dataclass(frozen=True)
class RawPriceDiscontinuityConfig:
    """Conservative common split-factor heuristic for unadjusted bars.

    A current open/prior scheduled-close ratio is flagged only when it is near
    a common forward or reverse split factor.  The flag is deliberately not a
    corporate-action assertion: it filters gap-template observations until a
    trusted split/action source is available.
    """

    common_factors: tuple[float, ...] = (
        0.1,
        0.2,
        0.25,
        1.0 / 3.0,
        0.5,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
    )
    relative_tolerance: float = 0.12

    def __post_init__(self) -> None:
        if not self.common_factors or any(
            not np.isfinite(factor) or factor <= 0 or np.isclose(factor, 1.0)
            for factor in self.common_factors
        ):
            raise ValueError("common split factors must be finite, positive, and not 1")
        if not np.isfinite(self.relative_tolerance) or not 0 < self.relative_tolerance < 0.5:
            raise ValueError("relative_tolerance must be in (0, 0.5)")


@dataclass
class StreamingIntradayResearchResult:
    data_dir: Path
    requested_tickers: tuple[str, ...]
    loaded_candidate_tickers: tuple[str, ...]
    loaded_proxy_tickers: tuple[str, ...]
    expected_sessions: pd.DatetimeIndex
    full_sessions: pd.DatetimeIndex
    signals: pd.DataFrame
    signal_rejections: pd.DataFrame
    signal_generation_audit: pd.DataFrame
    eligibility_summary: pd.DataFrame
    trades: pd.DataFrame
    execution_rejections: pd.DataFrame
    cost_grid_trades: pd.DataFrame
    cost_grid_summary: pd.DataFrame
    daily_returns: pd.DataFrame
    day_cluster_stats: pd.DataFrame
    annual_stats: pd.DataFrame
    rolling_diagnostics: pd.DataFrame
    capacity_daily_returns: pd.DataFrame
    capacity_summary: pd.DataFrame
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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_path(data_dir: Path, ticker: str) -> Path:
    return data_dir / f"{ticker.upper()}_15min.parquet"


def _minute_number(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def _exact_bar_series(
    work: pd.DataFrame,
    *,
    minute: int,
    column: str,
) -> pd.Series:
    selected = work.loc[work["minute"].eq(minute), ["trade_date", column]]
    return selected.set_index("trade_date")[column]


def reduce_bars_to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Reduce normalized bars to exact event-clock and quality features."""

    work = bars.copy()
    work["trade_date"] = work["ts"].dt.normalize()
    work["minute"] = _minute_number(work["ts"])
    work["dollar_volume"] = work["close"] * work["volume"]
    grouped = work.groupby("trade_date", sort=True, observed=True)
    daily = grouped.agg(
        bars_in_session=("ts", "size"),
        first_bar_minute=("minute", "min"),
        last_bar_minute=("minute", "max"),
        session_dollar_volume=("dollar_volume", "sum"),
        zero_volume_bars=("volume", lambda values: int(values.eq(0).sum())),
    )
    exact_fields = {
        "open_0930": (SESSION_OPEN_MINUTE, "open"),
        "close_1015": (GAP_FEATURE_LAST_MINUTE, "close"),
        "open_1045": (10 * 60 + 45, "open"),
        "volume_1045": (10 * 60 + 45, "volume"),
        "close_1300": (SHOCK_FEATURE_LAST_MINUTE, "close"),
        "open_1330": (13 * 60 + 30, "open"),
        "volume_1330": (13 * 60 + 30, "volume"),
        "close_1545": (FULL_SESSION_LAST_MINUTE, "close"),
        "volume_1545": (FULL_SESSION_LAST_MINUTE, "volume"),
    }
    for output_column, (minute, source_column) in exact_fields.items():
        daily[output_column] = _exact_bar_series(
            work, minute=minute, column=source_column
        )

    gap_window = work["minute"].between(
        SESSION_OPEN_MINUTE, GAP_FEATURE_LAST_MINUTE
    )
    shock_window = work["minute"].between(
        SESSION_OPEN_MINUTE, SHOCK_FEATURE_LAST_MINUTE
    )
    for name, mask, expected_count in (
        ("gap_feature", gap_window, 4),
        ("shock_feature", shock_window, 15),
    ):
        window = work.loc[mask]
        counts = window.groupby("trade_date", observed=True).size()
        min_volume = window.groupby("trade_date", observed=True)["volume"].min()
        daily[f"{name}_bar_count"] = counts
        daily[f"{name}_min_volume"] = min_volume
        daily[f"{name}_quality_ok"] = (
            daily[f"{name}_bar_count"].eq(expected_count)
            & daily[f"{name}_min_volume"].gt(0)
        )

    daily["is_exact_full_session"] = (
        daily["bars_in_session"].eq(FULL_SESSION_BAR_COUNT)
        & daily["first_bar_minute"].eq(SESSION_OPEN_MINUTE)
        & daily["last_bar_minute"].eq(FULL_SESSION_LAST_MINUTE)
    )
    daily["is_exact_observed_early_close"] = (
        daily["bars_in_session"].eq(EARLY_CLOSE_BAR_COUNT)
        & daily["first_bar_minute"].eq(SESSION_OPEN_MINUTE)
        & daily["last_bar_minute"].eq(EARLY_CLOSE_LAST_MINUTE)
    )
    return daily.sort_index()


def _normalize_strict_metadata(
    candidates: tuple[str, ...],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise ValueError("non-empty sector metadata is required; SPY fallback is disabled")
    work = metadata.copy()
    work.columns = [str(column).strip().lower() for column in work.columns]
    if "ticker" not in work.columns or "sector" not in work.columns:
        raise ValueError("sector metadata must contain ticker and sector columns")
    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    if work["ticker"].duplicated().any():
        duplicate = work.loc[work["ticker"].duplicated(), "ticker"].iloc[0]
        raise ValueError(f"duplicate sector metadata for {duplicate}")
    work = work.set_index("ticker").reindex(candidates)
    if "sector_proxy" not in work.columns:
        work["sector_proxy"] = pd.NA
    sectors = work["sector"].where(work["sector"].notna(), "").astype(str).str.strip()
    mapped = sectors.str.upper().map(DEFAULT_SECTOR_PROXIES)
    proxies = work["sector_proxy"].where(work["sector_proxy"].notna(), mapped)
    proxies = proxies.where(proxies.notna(), "").astype(str).str.upper().str.strip()
    normalized = pd.DataFrame(
        {"ticker": candidates, "sector": sectors.to_numpy(), "sector_proxy": proxies.to_numpy()}
    )
    exclusions: dict[str, str] = {}
    for row in normalized.itertuples(index=False):
        if not row.sector:
            exclusions[row.ticker] = "missing_sector_metadata"
        elif not row.sector_proxy:
            exclusions[row.ticker] = "missing_sector_proxy_mapping"
    return normalized, exclusions


def _load_daily(path: Path, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    bars = normalize_bars(pd.read_parquet(path), ticker=ticker)
    return bars, reduce_bars_to_daily(bars), _sha256_file(path)


def _expected_sessions(market_daily: pd.DataFrame) -> pd.DatetimeIndex:
    if market_daily.empty:
        raise ValueError("market proxy has no sessions")
    start = pd.Timestamp(market_daily.index.min()).normalize()
    end = pd.Timestamp(market_daily.index.max()).normalize()
    sessions = pd.date_range(start, end, freq=TRADING_DAY).normalize()
    if sessions.empty:
        raise ValueError("market proxy span contains no NYSE sessions")
    return pd.DatetimeIndex(sessions)


def _market_status(
    market_daily: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex,
) -> pd.Series:
    reindexed = market_daily.reindex(expected_sessions)
    status = pd.Series("missing_market_proxy_session", index=expected_sessions, dtype=object)
    status.loc[reindexed["is_exact_full_session"].eq(True)] = "full_session"
    status.loc[
        reindexed["is_exact_observed_early_close"].eq(True)
    ] = (
        "observed_early_close_excluded"
    )
    observed = reindexed["bars_in_session"].notna()
    unresolved = observed & status.eq("missing_market_proxy_session")
    status.loc[unresolved] = "incomplete_market_proxy_session"
    status.index.name = "trade_date"
    return status


def _calculate_streaming_eligibility(
    daily: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex,
    config: EligibilityConfig,
) -> pd.DataFrame:
    work = daily.reindex(expected_sessions).copy()
    work.index.name = "trade_date"
    work["session_dollar_volume"] = work["session_dollar_volume"].fillna(0.0)
    work["bars_in_session"] = work["bars_in_session"].fillna(0).astype(int)
    work["session_completeness"] = (
        work["bars_in_session"] / config.expected_bars_per_session
    ).clip(upper=1.0)
    prior_dollar_volume = work["session_dollar_volume"].shift(1)
    prior_completeness = work["session_completeness"].shift(1)
    work["price_proxy"] = work["close_1545"].shift(1)
    work["median_dollar_volume"] = prior_dollar_volume.rolling(
        config.lookback_sessions,
        min_periods=config.min_history_sessions,
    ).median()
    work["data_completeness"] = prior_completeness.rolling(
        config.lookback_sessions,
        min_periods=config.min_history_sessions,
    ).mean()
    work["history_sessions"] = np.arange(len(work), dtype=int)
    work["eligible"] = (
        work["history_sessions"].ge(config.min_history_sessions)
        & work["price_proxy"].ge(config.min_price)
        & work["median_dollar_volume"].ge(config.min_median_dollar_volume)
        & work["data_completeness"].ge(config.min_data_completeness)
    ).fillna(False)
    return work[
        [
            "eligible",
            "history_sessions",
            "price_proxy",
            "median_dollar_volume",
            "data_completeness",
            "bars_in_session",
        ]
    ]


def _eligibility_summary(
    ticker: str,
    eligibility: pd.DataFrame,
    config: EligibilityConfig,
) -> dict[str, int | str]:
    return {
        "ticker": ticker,
        "n_expected_sessions": len(eligibility),
        "n_eligible_sessions": int(eligibility["eligible"].sum()),
        "n_history_gate_fail": int(
            eligibility["history_sessions"].lt(config.min_history_sessions).sum()
        ),
        "n_price_gate_fail": int(
            eligibility["price_proxy"].lt(config.min_price).fillna(True).sum()
        ),
        "n_dollar_volume_gate_fail": int(
            eligibility["median_dollar_volume"]
            .lt(config.min_median_dollar_volume)
            .fillna(True)
            .sum()
        ),
        "n_completeness_gate_fail": int(
            eligibility["data_completeness"]
            .lt(config.min_data_completeness)
            .fillna(True)
            .sum()
        ),
    }


def _nearest_common_factor(
    ratios: pd.Series,
    config: RawPriceDiscontinuityConfig,
) -> tuple[pd.Series, pd.Series]:
    ratio_values = ratios.to_numpy(dtype=float)
    factors = np.asarray(config.common_factors, dtype=float)
    finite = np.isfinite(ratio_values) & (ratio_values > 0)
    distances = np.full((len(ratio_values), len(factors)), np.inf)
    distances[finite] = np.abs(
        ratio_values[finite, np.newaxis] / factors[np.newaxis, :] - 1.0
    )
    nearest_index = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(ratio_values)), nearest_index]
    flagged = finite & (nearest_distance <= config.relative_tolerance)
    factor_values = np.where(flagged, factors[nearest_index], np.nan)
    return (
        pd.Series(flagged, index=ratios.index, dtype=bool),
        pd.Series(factor_values, index=ratios.index, dtype=float),
    )


def _clock_index(index: pd.Index, clock: str) -> pd.DatetimeIndex:
    hour, minute = (int(part) for part in clock.split(":"))
    return pd.DatetimeIndex(index).normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _residual(
    asset_return: pd.Series,
    market_return: pd.Series,
    sector_return: pd.Series,
    *,
    market_weight: float,
    same_proxy: bool,
) -> pd.Series:
    reference = (
        market_return
        if same_proxy
        else market_weight * market_return + (1.0 - market_weight) * sector_return
    )
    return asset_return - reference


def _signal_frame(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.reindex(columns=SIGNAL_COLUMNS).reset_index(drop=True)
    if not output.empty:
        output = output.sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
    return output


def _gap_signals(
    *,
    ticker: str,
    sector: str,
    sector_proxy: str,
    asset: pd.DataFrame,
    market: pd.DataFrame,
    proxy: pd.DataFrame,
    eligibility: pd.DataFrame,
    market_status: pd.Series,
    discontinuity_config: RawPriceDiscontinuityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    config = GapFirstHourConfig()
    index = eligibility.index
    a = asset.reindex(index)
    m = market.reindex(index)
    s = proxy.reindex(index)
    same_proxy = sector_proxy in {"SPY", ticker}

    feature_quality = (
        a["gap_feature_quality_ok"].eq(True)
        & m["gap_feature_quality_ok"].eq(True)
        & s["gap_feature_quality_ok"].eq(True)
    )
    required = pd.concat(
        [
            a[["open_0930", "close_1015"]],
            a[["close_1545"]].shift(1).add_prefix("prior_asset_"),
            m[["open_0930", "close_1015"]].add_prefix("market_"),
            m[["close_1545"]].shift(1).add_prefix("prior_market_"),
            s[["open_0930", "close_1015"]].add_prefix("sector_"),
            s[["close_1545"]].shift(1).add_prefix("prior_sector_"),
        ],
        axis=1,
    )
    feature_complete = required.notna().all(axis=1) & feature_quality
    quality = (
        eligibility["eligible"]
        & market_status.reindex(index).eq("full_session")
        & feature_complete
    )

    asset_gap_ratio = a["open_0930"] / a["close_1545"].shift(1)
    market_gap_ratio = m["open_0930"] / m["close_1545"].shift(1)
    sector_gap_ratio = s["open_0930"] / s["close_1545"].shift(1)
    asset_gap = asset_gap_ratio - 1.0
    market_gap = market_gap_ratio - 1.0
    sector_gap = sector_gap_ratio - 1.0
    asset_first_hour = a["close_1015"] / a["open_0930"] - 1.0
    market_first_hour = m["close_1015"] / m["open_0930"] - 1.0
    sector_first_hour = s["close_1015"] / s["open_0930"] - 1.0
    residual_gap = _residual(
        asset_gap,
        market_gap,
        sector_gap,
        market_weight=config.market_weight,
        same_proxy=same_proxy,
    )
    residual_first_hour = _residual(
        asset_first_hour,
        market_first_hour,
        sector_first_hour,
        market_weight=config.market_weight,
        same_proxy=same_proxy,
    )
    threshold = (
        residual_gap.mul(residual_first_hour).gt(0)
        & residual_gap.abs().ge(config.min_abs_residual_gap)
        & residual_first_hour.abs().ge(config.min_abs_residual_first_hour)
    )
    threshold_pass = quality & threshold
    asset_flag, asset_factor = _nearest_common_factor(
        asset_gap_ratio, discontinuity_config
    )
    market_flag, market_factor = _nearest_common_factor(
        market_gap_ratio, discontinuity_config
    )
    sector_flag, sector_factor = _nearest_common_factor(
        sector_gap_ratio, discontinuity_config
    )
    discontinuity = asset_flag | market_flag | sector_flag
    selected = threshold_pass & ~discontinuity

    rejection_rows: list[dict[str, object]] = []
    for day in index[threshold_pass & discontinuity]:
        components = []
        factors = []
        for name, flag, factor in (
            ("asset", asset_flag, asset_factor),
            ("market", market_flag, market_factor),
            ("sector", sector_flag, sector_factor),
        ):
            if bool(flag.loc[day]):
                components.append(name)
                factors.append(f"{name}:{factor.loc[day]:g}")
        rejection_rows.append(
            {
                "template_id": GAP_FIRST_HOUR_TEMPLATE_ID,
                "ticker": ticker,
                "trade_date": day,
                "signal_rejection_reason": "raw_price_common_split_factor",
                "flagged_components": "|".join(components),
                "nearest_common_factors": "|".join(factors),
                "asset_open_prior_close_ratio": asset_gap_ratio.loc[day],
                "market_open_prior_close_ratio": market_gap_ratio.loc[day],
                "sector_open_prior_close_ratio": sector_gap_ratio.loc[day],
            }
        )

    selected_index = index[selected]
    rows = pd.DataFrame(index=selected_index)
    rows["template_id"] = GAP_FIRST_HOUR_TEMPLATE_ID
    rows["ticker"] = ticker
    rows["sector"] = sector
    rows["sector_proxy"] = sector_proxy
    rows["trade_date"] = selected_index
    rows["side"] = np.sign(residual_gap.loc[selected_index]).astype(int)
    rows["decision_ts"] = _clock_index(selected_index, config.decision_time)
    rows["feature_bar_ts"] = rows["decision_ts"] - BAR_DELTA
    rows["feature_available_ts"] = rows["feature_bar_ts"] + BAR_DELTA
    rows["entry_bar_ts"] = _clock_index(selected_index, config.entry_time)
    rows["entry_ts"] = rows["entry_bar_ts"]
    rows["exit_bar_ts"] = _clock_index(selected_index, config.exit_bar_time)
    rows["exit_ts"] = rows["exit_bar_ts"] + BAR_DELTA
    for column, series in (
        ("asset_gap", asset_gap),
        ("market_gap", market_gap),
        ("sector_gap", sector_gap),
        ("residual_gap", residual_gap),
        ("asset_first_hour", asset_first_hour),
        ("market_first_hour", market_first_hour),
        ("sector_first_hour", sector_first_hour),
        ("residual_first_hour", residual_first_hour),
    ):
        rows[column] = series.loc[selected_index].to_numpy()
    rows["signal_strength"] = (
        residual_gap.loc[selected_index].abs()
        + residual_first_hour.loc[selected_index].abs()
    ).to_numpy()
    for column in ("price_proxy", "median_dollar_volume", "data_completeness"):
        rows[column] = eligibility.loc[selected_index, column].to_numpy()

    eligible = eligibility["eligible"]
    full_market = market_status.reindex(index).eq("full_session")
    audit = {
        "template_id": GAP_FIRST_HOUR_TEMPLATE_ID,
        "ticker": ticker,
        "n_expected_sessions": len(index),
        "n_eligible_sessions": int(eligible.sum()),
        "n_market_not_full_after_eligibility": int((eligible & ~full_market).sum()),
        "n_feature_quality_fail_after_calendar": int(
            (eligible & full_market & ~feature_complete).sum()
        ),
        "n_threshold_fail_after_quality": int((quality & ~threshold).sum()),
        "n_raw_discontinuity_filtered": int((threshold_pass & discontinuity).sum()),
        "n_signals": int(selected.sum()),
    }
    return _signal_frame(rows), pd.DataFrame(rejection_rows), audit


def _shock_signals(
    *,
    ticker: str,
    sector: str,
    sector_proxy: str,
    asset: pd.DataFrame,
    market: pd.DataFrame,
    proxy: pd.DataFrame,
    eligibility: pd.DataFrame,
    market_status: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    config = IntradayShockConfig()
    index = eligibility.index
    a = asset.reindex(index)
    m = market.reindex(index)
    s = proxy.reindex(index)
    same_proxy = sector_proxy in {"SPY", ticker}
    feature_quality = (
        a["shock_feature_quality_ok"].eq(True)
        & m["shock_feature_quality_ok"].eq(True)
        & s["shock_feature_quality_ok"].eq(True)
    )
    required = pd.concat(
        [
            a[["open_0930", "close_1300"]],
            m[["open_0930", "close_1300"]].add_prefix("market_"),
            s[["open_0930", "close_1300"]].add_prefix("sector_"),
        ],
        axis=1,
    )
    feature_complete = required.notna().all(axis=1) & feature_quality
    full_market = market_status.reindex(index).eq("full_session")
    quality = eligibility["eligible"] & full_market & feature_complete

    asset_shock = a["close_1300"] / a["open_0930"] - 1.0
    market_shock = m["close_1300"] / m["open_0930"] - 1.0
    sector_shock = s["close_1300"] / s["open_0930"] - 1.0
    residual_shock = _residual(
        asset_shock,
        market_shock,
        sector_shock,
        market_weight=config.market_weight,
        same_proxy=same_proxy,
    )
    threshold = residual_shock.abs().ge(config.min_abs_residual_shock)
    selected = quality & threshold
    selected_index = index[selected]

    rows = pd.DataFrame(index=selected_index)
    rows["template_id"] = INTRADAY_SHOCK_TEMPLATE_ID
    rows["ticker"] = ticker
    rows["sector"] = sector
    rows["sector_proxy"] = sector_proxy
    rows["trade_date"] = selected_index
    rows["side"] = -np.sign(residual_shock.loc[selected_index]).astype(int)
    rows["feature_bar_ts"] = _clock_index(
        selected_index, config.evaluation_bar_time
    )
    rows["feature_available_ts"] = rows["feature_bar_ts"] + BAR_DELTA
    rows["decision_ts"] = rows["feature_available_ts"]
    rows["entry_bar_ts"] = _clock_index(selected_index, config.entry_time)
    rows["entry_ts"] = rows["entry_bar_ts"]
    rows["exit_bar_ts"] = _clock_index(selected_index, config.exit_bar_time)
    rows["exit_ts"] = rows["exit_bar_ts"] + BAR_DELTA
    for column, series in (
        ("asset_shock", asset_shock),
        ("market_shock", market_shock),
        ("sector_shock", sector_shock),
        ("residual_shock", residual_shock),
    ):
        rows[column] = series.loc[selected_index].to_numpy()
    rows["signal_strength"] = residual_shock.loc[selected_index].abs().to_numpy()
    for column in ("price_proxy", "median_dollar_volume", "data_completeness"):
        rows[column] = eligibility.loc[selected_index, column].to_numpy()

    eligible = eligibility["eligible"]
    audit = {
        "template_id": INTRADAY_SHOCK_TEMPLATE_ID,
        "ticker": ticker,
        "n_expected_sessions": len(index),
        "n_eligible_sessions": int(eligible.sum()),
        "n_market_not_full_after_eligibility": int((eligible & ~full_market).sum()),
        "n_feature_quality_fail_after_calendar": int(
            (eligible & full_market & ~feature_complete).sum()
        ),
        "n_threshold_fail_after_quality": int((quality & ~threshold).sum()),
        "n_raw_discontinuity_filtered": 0,
        "n_signals": int(selected.sum()),
    }
    return _signal_frame(rows), audit


def _frame_coverage_row(
    *,
    ticker: str,
    role: str,
    path: Path,
    input_hash: str,
    daily: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex,
    exclusion_reason: str = "",
    discontinuity_config: RawPriceDiscontinuityConfig,
) -> dict[str, object]:
    observed = pd.DatetimeIndex(daily.index).normalize()
    expected_observed = observed.intersection(expected_sessions)
    unexpected = observed.difference(expected_sessions)
    aligned = daily.reindex(expected_sessions)
    ratios = aligned["open_0930"] / aligned["close_1545"].shift(1)
    discontinuity, _ = _nearest_common_factor(ratios, discontinuity_config)
    return {
        "ticker": ticker,
        "role": role,
        "status": "excluded" if exclusion_reason else "loaded",
        "exclusion_reason": exclusion_reason,
        "input_path": str(path.resolve()),
        "input_file_exists": True,
        "input_sha256": input_hash,
        "n_rows": int(daily["bars_in_session"].sum()),
        "first_session": observed.min() if len(observed) else pd.NaT,
        "last_session": observed.max() if len(observed) else pd.NaT,
        "n_expected_sessions": len(expected_sessions),
        "n_observed_expected_sessions": len(expected_observed),
        "n_missing_expected_sessions": len(expected_sessions.difference(observed)),
        "n_unexpected_non_nyse_sessions": len(unexpected),
        "n_exact_full_sessions": int(daily["is_exact_full_session"].sum()),
        "n_observed_early_closes": int(
            daily["is_exact_observed_early_close"].sum()
        ),
        "n_partial_sessions": int(
            (
                ~daily["is_exact_full_session"]
                & ~daily["is_exact_observed_early_close"]
            ).sum()
        ),
        "n_zero_volume_bars": int(daily["zero_volume_bars"].sum()),
        "n_zero_volume_1045_bars": int(aligned["volume_1045"].eq(0).sum()),
        "n_zero_volume_1330_bars": int(aligned["volume_1330"].eq(0).sum()),
        "n_zero_volume_1545_bars": int(aligned["volume_1545"].eq(0).sum()),
        "n_common_split_factor_discontinuities": int(discontinuity.sum()),
        "coverage_fraction": (
            float(len(expected_observed) / len(expected_sessions))
            if len(expected_sessions)
            else np.nan
        ),
    }


def _missing_coverage_row(
    ticker: str,
    role: str,
    path: Path,
    reason: str,
    n_expected_sessions: int,
) -> dict[str, object]:
    exists = path.is_file()
    unknown_or_zero: float = np.nan if exists else 0.0
    return {
        "ticker": ticker,
        "role": role,
        "status": "excluded" if role == "candidate" else "missing",
        "exclusion_reason": reason,
        "input_path": str(path.resolve()),
        "input_file_exists": exists,
        "input_sha256": "",
        "n_rows": unknown_or_zero,
        "first_session": pd.NaT,
        "last_session": pd.NaT,
        "n_expected_sessions": n_expected_sessions,
        "n_observed_expected_sessions": unknown_or_zero,
        "n_missing_expected_sessions": (
            np.nan if exists else n_expected_sessions
        ),
        "n_unexpected_non_nyse_sessions": unknown_or_zero,
        "n_exact_full_sessions": unknown_or_zero,
        "n_observed_early_closes": unknown_or_zero,
        "n_partial_sessions": unknown_or_zero,
        "n_zero_volume_bars": unknown_or_zero,
        "n_zero_volume_1045_bars": unknown_or_zero,
        "n_zero_volume_1330_bars": unknown_or_zero,
        "n_zero_volume_1545_bars": unknown_or_zero,
        "n_common_split_factor_discontinuities": unknown_or_zero,
        "coverage_fraction": np.nan if exists else 0.0,
    }


def _cost_grid_summary(cost_grid_trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "template_id",
        "cost_bps",
        "primary_cost_case",
        "n_trades",
        "n_days",
        "n_tickers",
        "mean_net_return",
        "median_net_return",
        "win_rate_net",
    ]
    if cost_grid_trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (template_id, cost_bps), group in cost_grid_trades.groupby(
        ["template_id", "cost_bps"], sort=True, observed=True
    ):
        returns = group["net_return"].astype(float)
        rows.append(
            {
                "template_id": template_id,
                "cost_bps": float(cost_bps),
                "primary_cost_case": float(cost_bps) == PRIMARY_COST_BPS,
                "n_trades": len(group),
                "n_days": int(pd.to_datetime(group["trade_date"]).nunique()),
                "n_tickers": int(group["ticker"].nunique()),
                "mean_net_return": float(returns.mean()),
                "median_net_return": float(returns.median()),
                "win_rate_net": float(returns.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_streaming_intraday_research(
    data_dir: str | Path,
    metadata: pd.DataFrame,
    candidates: Iterable[str],
    *,
    market_ticker: str = "SPY",
    eligibility_config: EligibilityConfig | None = None,
    discontinuity_config: RawPriceDiscontinuityConfig | None = None,
    cost_grid_bps: tuple[float, ...] = DEFAULT_COST_GRID_BPS,
    bootstrap_reps: int = 2_000,
) -> StreamingIntradayResearchResult:
    """Run both locked v0 templates from an explicit local parquet directory."""

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"intraday data directory does not exist: {root}")
    market_ticker = str(market_ticker).upper().strip()
    if market_ticker != "SPY":
        raise ValueError("streaming v0 runner is locked to SPY as the market proxy")
    requested = tuple(sorted({str(ticker).upper().strip() for ticker in candidates}))
    if not requested or any(not ticker for ticker in requested):
        raise ValueError("explicit non-empty candidate universe is required")
    costs = validate_cost_grid(tuple(cost_grid_bps))
    eligibility_config = eligibility_config or EligibilityConfig()
    discontinuity_config = discontinuity_config or RawPriceDiscontinuityConfig()
    normalized_metadata, exclusions = _normalize_strict_metadata(requested, metadata)
    metadata_fingerprint = sha256(
        normalized_metadata.sort_values("ticker").to_csv(index=False).encode("utf-8")
    ).hexdigest()
    universe_fingerprint = sha256(("\n".join(requested) + "\n").encode("utf-8")).hexdigest()

    market_path = _parquet_path(root, market_ticker)
    if not market_path.is_file():
        raise FileNotFoundError(f"required market proxy parquet is missing: {market_path}")
    market_bars, market_daily, market_hash = _load_daily(market_path, market_ticker)
    expected_sessions = _expected_sessions(market_daily)
    market_status = _market_status(market_daily, expected_sessions)
    full_sessions = pd.DatetimeIndex(
        market_status.index[market_status.eq("full_session")]
    )
    if full_sessions.empty:
        raise ValueError("market proxy has no exact full sessions")

    observed_counts: Counter[pd.Timestamp] = Counter()

    def register_observed(daily: pd.DataFrame) -> None:
        for day in pd.DatetimeIndex(daily.index).normalize().intersection(expected_sessions):
            observed_counts[pd.Timestamp(day)] += 1

    register_observed(market_daily)
    coverage_rows: list[dict[str, object]] = [
        _frame_coverage_row(
            ticker=market_ticker,
            role="market_proxy",
            path=market_path,
            input_hash=market_hash,
            daily=market_daily,
            expected_sessions=expected_sessions,
            discontinuity_config=discontinuity_config,
        )
    ]
    del market_bars

    metadata_by_ticker = normalized_metadata.set_index("ticker").to_dict("index")
    required_proxies = sorted(
        {
            str(details["sector_proxy"]).upper()
            for ticker, details in metadata_by_ticker.items()
            if ticker not in exclusions and details["sector_proxy"]
        }
        - {market_ticker}
    )
    proxy_daily: dict[str, pd.DataFrame] = {market_ticker: market_daily}
    missing_proxies: set[str] = set()
    for proxy_ticker in required_proxies:
        path = _parquet_path(root, proxy_ticker)
        if not path.is_file():
            missing_proxies.add(proxy_ticker)
            coverage_rows.append(
                _missing_coverage_row(
                    proxy_ticker,
                    "sector_proxy",
                    path,
                    "missing_sector_proxy_file",
                    len(expected_sessions),
                )
            )
            continue
        proxy_bars, daily, input_hash = _load_daily(path, proxy_ticker)
        proxy_daily[proxy_ticker] = daily
        register_observed(daily)
        coverage_rows.append(
            _frame_coverage_row(
                ticker=proxy_ticker,
                role="sector_proxy",
                path=path,
                input_hash=input_hash,
                daily=daily,
                expected_sessions=expected_sessions,
                discontinuity_config=discontinuity_config,
            )
        )
        del proxy_bars

    for ticker, details in metadata_by_ticker.items():
        proxy_ticker = str(details["sector_proxy"]).upper()
        if ticker not in exclusions and proxy_ticker in missing_proxies:
            exclusions[ticker] = f"missing_sector_proxy_file:{proxy_ticker}"
        if ticker in proxy_daily:
            exclusions[ticker] = "candidate_is_required_proxy"

    signal_frames: list[pd.DataFrame] = []
    signal_rejection_frames: list[pd.DataFrame] = []
    generation_rows: list[dict[str, int | str]] = []
    eligibility_rows: list[dict[str, int | str]] = []
    trade_frames: list[pd.DataFrame] = []
    execution_rejection_frames: list[pd.DataFrame] = []
    loaded_candidates: list[str] = []

    for ticker in requested:
        path = _parquet_path(root, ticker)
        if ticker in exclusions:
            coverage_rows.append(
                _missing_coverage_row(
                    ticker,
                    "candidate",
                    path,
                    exclusions[ticker],
                    len(expected_sessions),
                )
            )
            continue
        if not path.is_file():
            exclusions[ticker] = "missing_candidate_file"
            coverage_rows.append(
                _missing_coverage_row(
                    ticker,
                    "candidate",
                    path,
                    exclusions[ticker],
                    len(expected_sessions),
                )
            )
            continue

        bars, daily, input_hash = _load_daily(path, ticker)
        register_observed(daily)
        coverage_rows.append(
            _frame_coverage_row(
                ticker=ticker,
                role="candidate",
                path=path,
                input_hash=input_hash,
                daily=daily,
                expected_sessions=expected_sessions,
                discontinuity_config=discontinuity_config,
            )
        )
        loaded_candidates.append(ticker)
        details = metadata_by_ticker[ticker]
        sector = str(details["sector"])
        sector_proxy = str(details["sector_proxy"]).upper()
        eligibility = _calculate_streaming_eligibility(
            daily, expected_sessions, eligibility_config
        )
        eligibility_rows.append(
            _eligibility_summary(ticker, eligibility, eligibility_config)
        )

        gap_signals, gap_rejections, gap_audit = _gap_signals(
            ticker=ticker,
            sector=sector,
            sector_proxy=sector_proxy,
            asset=daily,
            market=market_daily,
            proxy=proxy_daily[sector_proxy],
            eligibility=eligibility,
            market_status=market_status,
            discontinuity_config=discontinuity_config,
        )
        shock_signals, shock_audit = _shock_signals(
            ticker=ticker,
            sector=sector,
            sector_proxy=sector_proxy,
            asset=daily,
            market=market_daily,
            proxy=proxy_daily[sector_proxy],
            eligibility=eligibility,
            market_status=market_status,
        )
        candidate_signals = pd.concat(
            [gap_signals, shock_signals], ignore_index=True
        ).sort_values(["trade_date", "template_id"], ignore_index=True)
        signal_frames.append(candidate_signals)
        if not gap_rejections.empty:
            signal_rejection_frames.append(gap_rejections)
        generation_rows.extend([gap_audit, shock_audit])
        simulation = simulate_fixed_time_signals_audited(
            candidate_signals,
            {ticker: bars},
            round_trip_cost_bps=PRIMARY_COST_BPS,
            frames_are_normalized=True,
        )
        if not simulation.trades.empty:
            trade_frames.append(simulation.trades)
        if not simulation.execution_rejections.empty:
            execution_rejection_frames.append(simulation.execution_rejections)
        del bars, daily, eligibility, candidate_signals

    if not loaded_candidates:
        reasons = Counter(exclusions.values())
        raise ValueError(f"no candidate could be evaluated: {dict(reasons)}")

    signals = (
        _signal_frame(pd.concat(signal_frames, ignore_index=True))
        if signal_frames
        else _signal_frame(pd.DataFrame())
    )
    signal_rejections = (
        pd.concat(signal_rejection_frames, ignore_index=True).sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
        if signal_rejection_frames
        else pd.DataFrame(
            columns=[
                "template_id",
                "ticker",
                "trade_date",
                "signal_rejection_reason",
                "flagged_components",
                "nearest_common_factors",
                "asset_open_prior_close_ratio",
                "market_open_prior_close_ratio",
                "sector_open_prior_close_ratio",
            ]
        )
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True).sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
        if trade_frames
        else pd.DataFrame(columns=[*SIGNAL_COLUMNS, "gross_return", "net_return"])
    )
    execution_rejections = (
        pd.concat(execution_rejection_frames, ignore_index=True).sort_values(
            ["trade_date", "template_id", "ticker"], ignore_index=True
        )
        if execution_rejection_frames
        else pd.DataFrame(
            columns=[
                *SIGNAL_COLUMNS,
                "execution_status",
                "execution_rejection_reason",
            ]
        )
    )
    cost_grid_trades = materialize_cost_grid(trades, costs)
    daily_returns = make_daily_returns(cost_grid_trades)
    day_stats = day_cluster_statistics(
        daily_returns, cost_grid_bps=costs, bootstrap_reps=bootstrap_reps
    )
    capacity_daily, capacity_summary = capacity_overlays(
        cost_grid_trades, full_sessions
    )
    ticker_summary, sector_summary = concentration_summaries(trades)

    calendar_rows: list[dict[str, object]] = []
    market_aligned = market_daily.reindex(expected_sessions)
    for day in expected_sessions:
        calendar_rows.append(
            {
                "trade_date": day,
                "expected_nyse_session": True,
                "market_session_status": market_status.loc[day],
                "market_bars_in_session": market_aligned.loc[day, "bars_in_session"],
                "n_loaded_inputs_observed": observed_counts[day],
                "missing_from_all_loaded_inputs": observed_counts[day] == 0,
            }
        )

    return StreamingIntradayResearchResult(
        data_dir=root,
        requested_tickers=requested,
        loaded_candidate_tickers=tuple(loaded_candidates),
        loaded_proxy_tickers=tuple(sorted(proxy_daily)),
        expected_sessions=expected_sessions,
        full_sessions=full_sessions,
        signals=signals,
        signal_rejections=signal_rejections,
        signal_generation_audit=pd.DataFrame(generation_rows),
        eligibility_summary=pd.DataFrame(eligibility_rows),
        trades=trades,
        execution_rejections=execution_rejections,
        cost_grid_trades=cost_grid_trades,
        cost_grid_summary=_cost_grid_summary(cost_grid_trades),
        daily_returns=daily_returns,
        day_cluster_stats=day_stats,
        annual_stats=annual_diagnostics(daily_returns, full_sessions),
        rolling_diagnostics=rolling_five_year_train_one_year_test(daily_returns),
        capacity_daily_returns=capacity_daily,
        capacity_summary=capacity_summary,
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
    )


def write_streaming_research_artifacts(
    result: StreamingIntradayResearchResult,
    output_dir: str | Path,
) -> Path:
    """Write a fresh research bundle under this worktree's ignored artifacts."""

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
        "trades_primary_10bps.parquet": result.trades,
        "execution_rejections.parquet": result.execution_rejections,
        "cost_grid_trades.parquet": result.cost_grid_trades,
        "daily_returns.parquet": result.daily_returns,
        "capacity_daily_returns.parquet": result.capacity_daily_returns,
    }
    csv_outputs = {
        "signal_generation_audit.csv": result.signal_generation_audit,
        "eligibility_summary.csv": result.eligibility_summary,
        "cost_grid_summary.csv": result.cost_grid_summary,
        "day_cluster_stats.csv": result.day_cluster_stats,
        "annual_stats.csv": result.annual_stats,
        "rolling_5y_train_1y_test.csv": result.rolling_diagnostics,
        "capacity_summary.csv": result.capacity_summary,
        "ticker_summary.csv": result.ticker_summary,
        "sector_summary.csv": result.sector_summary,
        "coverage_audit.csv": result.coverage_audit,
        "market_calendar_audit.csv": result.market_calendar_audit,
    }
    for filename, frame in parquet_outputs.items():
        frame.to_parquet(target / filename, index=False)
    for filename, frame in csv_outputs.items():
        frame.to_csv(target / filename, index=False)

    manifest = {
        "schema_version": "intraday_streaming_research.v1",
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_dir": str(result.data_dir),
        "requested_tickers": list(result.requested_tickers),
        "loaded_candidate_tickers": list(result.loaded_candidate_tickers),
        "loaded_proxy_tickers": list(result.loaded_proxy_tickers),
        "universe_sha256": result.universe_fingerprint,
        "metadata_sha256": result.metadata_fingerprint,
        "input_files": result.coverage_audit[
            ["ticker", "role", "status", "input_path", "input_sha256"]
        ].to_dict("records"),
        "calendar_start": str(result.expected_sessions.min().date()),
        "calendar_end": str(result.expected_sessions.max().date()),
        "n_expected_sessions": len(result.expected_sessions),
        "n_exact_full_sessions": len(result.full_sessions),
        "templates": [GAP_FIRST_HOUR_TEMPLATE_ID, INTRADAY_SHOCK_TEMPLATE_ID],
        "thresholds_or_directions_selected_from_results": False,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "cost_grid_bps": list(result.cost_grid_bps),
        "capacity_slots": [1, 3, 5, 10],
        "capacity_overlay_is_slot_based_not_share_or_broker_model": True,
        "bootstrap_reps": result.bootstrap_reps,
        "bootstrap_seed_policy": "sha256(template_id|cost_bps)",
        "eligibility_config": asdict(result.eligibility_config),
        "raw_price_discontinuity_config": asdict(result.discontinuity_config),
        "n_signals": len(result.signals),
        "n_signal_rejections": len(result.signal_rejections),
        "n_primary_trades": len(result.trades),
        "n_execution_rejections": len(result.execution_rejections),
        "manifest_written_last": True,
    }
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return target
