"""Frozen benchmark and preregistered Trend V2 trial specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FROZEN_BENCHMARK_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "FXI",
    "VNQ",
    "GLD",
    "SLV",
    "DBC",
    "TLT",
    "LQD",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Research copy of the production sleeve's rules as of 2026-08-27."""

    name: str = "frozen_prod_12etf_12_1_ma10"
    universe: tuple[str, ...] = FROZEN_BENCHMARK_UNIVERSE
    momentum_skip_months: int = 1
    momentum_lookback_months: int = 12
    moving_average_months: int = 10
    min_monthly_closes: int = 13
    volatility_days: int = 63
    volatility_floor: float = 0.04
    asset_weight_cap: float = 0.20
    gross_weight_cap: float = 1.00
    rebalance_band: float = 0.01
    max_monthly_turnover: float | None = None
    cost_bps_per_side: float = 5.0
    execution: str = "next_period"
    frozen_asof: str = "2026-08-27"


FROZEN_BENCHMARK = BenchmarkSpec()


@dataclass(frozen=True)
class MultiSpeedSpec:
    """One global, never-per-ticker multi-speed time-series trial."""

    name: str
    signal_kind: Literal["return_sign", "channel_breakout"]
    horizons_months: tuple[int, ...] = (3, 6, 12)
    enter_votes: int = 2
    exit_votes: int = 1
    volatility_days: int = 63
    covariance_days: int = 126
    volatility_floor: float = 0.04
    portfolio_target_volatility: float = 0.10
    asset_weight_cap: float = 0.20
    gross_weight_cap: float = 1.00
    rebalance_band: float = 0.01
    max_monthly_turnover: float = 0.50
    cost_bps_per_side: float = 5.0


# Four candidate trials, fixed before looking at results.  Adding or changing a
# row is a new preregistration/version and increases the family trial count.
PREREGISTERED_MULTISPEED_SPECS = (
    MultiSpeedSpec(
        name="ms_sign_2in_1out",
        signal_kind="return_sign",
        enter_votes=2,
        exit_votes=1,
    ),
    MultiSpeedSpec(
        name="ms_sign_unanimous_in",
        signal_kind="return_sign",
        enter_votes=3,
        exit_votes=1,
    ),
    MultiSpeedSpec(
        name="ms_breakout_2in_1out",
        signal_kind="channel_breakout",
        enter_votes=2,
        exit_votes=1,
    ),
    MultiSpeedSpec(
        name="ms_breakout_unanimous_in",
        signal_kind="channel_breakout",
        enter_votes=3,
        exit_votes=1,
    ),
)


@dataclass(frozen=True)
class CrossSectionalSpec:
    """Separate stock-level residual-trend family (not an ETF sleeve variant)."""

    name: str = "stock_residual_trend_sector_neutral_v0"
    beta_lookback_days: int = 126
    residual_momentum_days: int = 126
    volatility_days: int = 63
    min_history_days: int = 252
    top_quantile: float = 0.80
    bottom_quantile: float = 0.20
    long_short: bool = False
    gross_weight_cap: float = 1.00
    asset_weight_cap: float = 0.05
    rebalance_band: float = 0.0025
    max_monthly_turnover: float = 0.75
    cost_bps_per_side: float = 10.0


PREREGISTERED_CROSS_SECTIONAL_SPEC = CrossSectionalSpec()
