"""Point-in-time execution-grade eligibility proxies for intraday research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from .data import normalize_frame_map


@dataclass(frozen=True)
class EligibilityConfig:
    """Pre-registered, quote-free eligibility proxy defaults.

    All metrics for session T use completed sessions through T-1.  These are
    research gates, not assertions that a security is actually executable.
    """

    lookback_sessions: int = 20
    min_history_sessions: int = 10
    min_price: float = 5.0
    min_median_dollar_volume: float = 25_000_000.0
    min_data_completeness: float = 0.95
    expected_bars_per_session: int = 26

    def __post_init__(self) -> None:
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError("min_history_sessions must be in [1, lookback_sessions]")
        if (
            not isfinite(self.min_price)
            or not isfinite(self.min_median_dollar_volume)
            or self.min_price <= 0
            or self.min_median_dollar_volume < 0
        ):
            raise ValueError("price must be positive and dollar volume non-negative")
        if not isfinite(self.min_data_completeness) or not (
            0 <= self.min_data_completeness <= 1
        ):
            raise ValueError("min_data_completeness must be in [0, 1]")
        if self.expected_bars_per_session < 1:
            raise ValueError("expected_bars_per_session must be positive")


def _failure_reason(row: pd.Series, config: EligibilityConfig) -> str:
    reasons: list[str] = []
    if row["history_sessions"] < config.min_history_sessions:
        reasons.append("history")
    if pd.isna(row["price_proxy"]) or row["price_proxy"] < config.min_price:
        reasons.append("price")
    if (
        pd.isna(row["median_dollar_volume"])
        or row["median_dollar_volume"] < config.min_median_dollar_volume
    ):
        reasons.append("dollar_volume")
    if (
        pd.isna(row["data_completeness"])
        or row["data_completeness"] < config.min_data_completeness
    ):
        reasons.append("completeness")
    return "|".join(reasons)


def calculate_eligibility(
    frames: Mapping[str, pd.DataFrame],
    config: EligibilityConfig | None = None,
    *,
    calendar_ticker: str | None = None,
    frames_are_normalized: bool = False,
) -> pd.DataFrame:
    """Return one point-in-time eligibility row per ticker and session.

    Dollar volume is approximated as ``sum(close * volume)`` within each
    session.  Price is the prior session close.  Median dollar volume and bar
    completeness are rolling statistics shifted by one full session, so a
    day's eligibility cannot benefit from that day's eventual volume/bars.
    """

    config = config or EligibilityConfig()
    bars_by_ticker = (
        dict(frames) if frames_are_normalized else normalize_frame_map(frames)
    )
    canonical_sessions: pd.DatetimeIndex | None = None
    if calendar_ticker is not None:
        calendar_ticker = calendar_ticker.upper()
        if calendar_ticker not in bars_by_ticker:
            raise ValueError(f"calendar ticker is missing: {calendar_ticker}")
        canonical_sessions = pd.DatetimeIndex(
            sorted(
                {
                    day
                    for bars in bars_by_ticker.values()
                    for day in bars["ts"].dt.normalize().unique()
                }
            )
        )
    rows: list[pd.DataFrame] = []
    for ticker, bars in bars_by_ticker.items():
        work = bars.assign(trade_date=bars["ts"].dt.normalize())
        work = work.assign(dollar_volume=work["close"] * work["volume"])
        scheduled_closes = (
            work.loc[work["ts"].dt.time.eq(pd.Timestamp("15:45").time())]
            .groupby("trade_date", sort=True)["close"]
            .last()
        )
        daily = work.groupby("trade_date", sort=True).agg(
            session_dollar_volume=("dollar_volume", "sum"),
            bars_in_session=("ts", "size"),
        )
        daily["session_close"] = scheduled_closes
        if canonical_sessions is not None:
            daily = daily.reindex(canonical_sessions)
            daily.index.name = "trade_date"
            daily["session_dollar_volume"] = daily["session_dollar_volume"].fillna(0.0)
            daily["bars_in_session"] = daily["bars_in_session"].fillna(0).astype(int)
        daily = daily.reset_index()
        daily["session_completeness"] = (
            daily["bars_in_session"] / config.expected_bars_per_session
        ).clip(upper=1.0)
        prior_dollar_volume = daily["session_dollar_volume"].shift(1)
        prior_completeness = daily["session_completeness"].shift(1)
        daily["price_proxy"] = daily["session_close"].shift(1)
        daily["median_dollar_volume"] = prior_dollar_volume.rolling(
            config.lookback_sessions,
            min_periods=config.min_history_sessions,
        ).median()
        daily["data_completeness"] = prior_completeness.rolling(
            config.lookback_sessions,
            min_periods=config.min_history_sessions,
        ).mean()
        daily["history_sessions"] = np.arange(len(daily), dtype=int)
        daily["ticker"] = ticker
        eligible = (
            daily["history_sessions"].ge(config.min_history_sessions)
            & daily["price_proxy"].ge(config.min_price)
            & daily["median_dollar_volume"].ge(config.min_median_dollar_volume)
            & daily["data_completeness"].ge(config.min_data_completeness)
        )
        daily["eligible"] = eligible.fillna(False)
        daily["failure_reason"] = daily.apply(_failure_reason, axis=1, config=config)
        rows.append(
            daily[
                [
                    "ticker",
                    "trade_date",
                    "eligible",
                    "failure_reason",
                    "history_sessions",
                    "price_proxy",
                    "median_dollar_volume",
                    "data_completeness",
                    "bars_in_session",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["trade_date", "ticker"], ignore_index=True
    )
