"""Research-only intraday strategy laboratory.

Nothing in this package stages orders, mutates the strategy book, downloads
production data, or promotes a research result.
"""

from .capital import (
    AmbiguousCapitalTieError,
    CapitalFeasibilityResult,
    CapitalReuseConfig,
    apply_capital_feasibility,
)
from .data import IntradayDataError, LookaheadError, load_parquet_frames, normalize_bars
from .eligibility import EligibilityConfig, calculate_eligibility
from .lab import IntradayResearchResult, run_intraday_research, summarize_trades
from .simulator import (
    FixedTimeSimulationResult,
    MissingExecutionBarError,
    simulate_fixed_time_signals,
    simulate_fixed_time_signals_audited,
)
from .templates import (
    GAP_FIRST_HOUR_TEMPLATE_ID,
    INTRADAY_SHOCK_TEMPLATE_ID,
    GapFirstHourConfig,
    IntradayShockConfig,
    generate_gap_first_hour_signals,
    generate_intraday_shock_signals,
    prepare_metadata,
)

__all__ = [
    "GAP_FIRST_HOUR_TEMPLATE_ID",
    "INTRADAY_SHOCK_TEMPLATE_ID",
    "AmbiguousCapitalTieError",
    "CapitalFeasibilityResult",
    "CapitalReuseConfig",
    "EligibilityConfig",
    "FixedTimeSimulationResult",
    "GapFirstHourConfig",
    "IntradayDataError",
    "IntradayResearchResult",
    "IntradayShockConfig",
    "LookaheadError",
    "MissingExecutionBarError",
    "apply_capital_feasibility",
    "calculate_eligibility",
    "generate_gap_first_hour_signals",
    "generate_intraday_shock_signals",
    "load_parquet_frames",
    "normalize_bars",
    "prepare_metadata",
    "run_intraday_research",
    "simulate_fixed_time_signals",
    "simulate_fixed_time_signals_audited",
    "summarize_trades",
]
