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
from .gap_reversal import (
    GAP_DOWN_LONG_TEMPLATE_ID,
    GAP_REVERSAL_TEMPLATE_IDS,
    GAP_UP_SHORT_TEMPLATE_ID,
    GapReversalResearchResult,
    calculate_lagged_atr,
    candidate_slot_portfolios,
    run_gap_reversal_research,
    simulate_gap_reversal_signals,
    write_gap_reversal_artifacts,
)
from .lab import IntradayResearchResult, run_intraday_research, summarize_trades
from .simulator import (
    FixedTimeSimulationResult,
    MissingExecutionBarError,
    simulate_fixed_time_signals,
    simulate_fixed_time_signals_audited,
)
from .streaming import (
    RawPriceDiscontinuityConfig,
    StreamingIntradayResearchResult,
    reduce_bars_to_daily,
    run_streaming_intraday_research,
    write_streaming_research_artifacts,
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
    "GAP_DOWN_LONG_TEMPLATE_ID",
    "GAP_FIRST_HOUR_TEMPLATE_ID",
    "GAP_REVERSAL_TEMPLATE_IDS",
    "GAP_UP_SHORT_TEMPLATE_ID",
    "INTRADAY_SHOCK_TEMPLATE_ID",
    "AmbiguousCapitalTieError",
    "CapitalFeasibilityResult",
    "CapitalReuseConfig",
    "EligibilityConfig",
    "FixedTimeSimulationResult",
    "GapFirstHourConfig",
    "GapReversalResearchResult",
    "IntradayDataError",
    "IntradayResearchResult",
    "IntradayShockConfig",
    "LookaheadError",
    "MissingExecutionBarError",
    "RawPriceDiscontinuityConfig",
    "StreamingIntradayResearchResult",
    "apply_capital_feasibility",
    "calculate_eligibility",
    "calculate_lagged_atr",
    "candidate_slot_portfolios",
    "generate_gap_first_hour_signals",
    "generate_intraday_shock_signals",
    "load_parquet_frames",
    "normalize_bars",
    "prepare_metadata",
    "reduce_bars_to_daily",
    "run_gap_reversal_research",
    "run_intraday_research",
    "run_streaming_intraday_research",
    "simulate_fixed_time_signals",
    "simulate_fixed_time_signals_audited",
    "simulate_gap_reversal_signals",
    "summarize_trades",
    "write_gap_reversal_artifacts",
    "write_streaming_research_artifacts",
]
