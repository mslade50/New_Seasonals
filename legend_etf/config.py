"""Pinned production configuration for the original Legend EMA rule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Final

STRATEGY_NAME: Final = "Legend EMA ETF"
STRATEGY_VERSION: Final = "legend-etf-spy-qqq-v2"
NY_TZ: Final = "America/New_York"
DATABENTO_DATASET: Final = "GLBX.MDP3"
DATABENTO_SCHEMA: Final = "ohlcv-1m"


@dataclass(frozen=True)
class Market:
    root: str
    continuous_symbol: str
    etf: str
    trusted_from: str


MARKETS: Final[tuple[Market, ...]] = (
    Market("ES", "ES.v.0", "SPY", "2016-01-01"),
    Market("NQ", "NQ.v.0", "QQQ", "2016-01-01"),
    Market("RTY", "RTY.v.0", "IWM", "2018-01-01"),
)
MARKET_BY_ROOT: Final = {market.root: market for market in MARKETS}
MARKET_BY_ETF: Final = {market.etf: market for market in MARKETS}


@dataclass(frozen=True)
class RuleConfig:
    trend_ratio_min: float = 0.75
    ema_span: int = 20
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)
    decision_time: time = time(9, 31)
    decision_deadline: time = time(9, 31, 20)
    time_exit: time = time(10, 30)
    penny: float = 0.01
    atr_period: int = 14
    atr_stress_multiple: float = 1.25


RULES: Final = RuleConfig()


@dataclass(frozen=True)
class RiskProfile:
    """Stress-distance sizing inputs, expressed in NLV basis points.

    These are not stop-loss amounts: the strategy deliberately has no stop.
    They are a consistent shock-distance denominator used to bound gross size.
    """

    long_bps: float
    short_bps: float
    cluster_bps: float
    max_shares_per_root: int
    max_notional_pct: float


# The short allocation is intentionally half the long allocation.  This is the
# user's stated launch preference and remains configurable via the runner's
# environment without changing the historical signal rule.
PRIMARY_RISK: Final = RiskProfile(
    long_bps=10.0,
    short_bps=5.0,
    cluster_bps=30.0,
    max_shares_per_root=5_000,
    max_notional_pct=0.25,
)
PA_RISK: Final = RiskProfile(
    long_bps=5.0,
    short_bps=2.5,
    cluster_bps=15.0,
    max_shares_per_root=1_000,
    max_notional_pct=0.15,
)


TRANSIENT_STATES: Final = frozenset(
    {"submitting", "entry_working", "exit_modifying", "time_exit_working"}
)
TERMINAL_STATES: Final = frozenset(
    {
        "complete",
        "complete_emergency",
        "skipped",
        "blocked",
        "dry_run",
        "critical_unflattened",
        "critical_connection",
        "critical_unprotected",
        "critical_working_order",
        "critical_over_exit",
    }
)
