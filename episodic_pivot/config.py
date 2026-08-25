"""Versioned policy for the Episodic Pivot shadow process.

The discovery rules are intentionally broad.  A candidate is not preview-eligible until
fresh market data, fetched news evidence, and execution-capacity gates all pass.
Nothing in this module can enable live actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from strategy_config import ACCOUNT_VALUE as BOOK_ACCOUNT_VALUE
except Exception:  # Keep research tools importable in isolated test contexts.
    BOOK_ACCOUNT_VALUE = 750_000.0


@dataclass(frozen=True)
class DiscoveryPolicy:
    min_price: float = 1.0
    min_abs_gap_pct: float = 2.0
    min_abs_move_dollars: float = 0.90
    min_premarket_volume: int = 100_000
    quote_max_age_seconds: int = 90
    premarket_metrics_max_age_seconds: int = 900
    future_timestamp_tolerance_seconds: int = 2
    max_prior_close_basis_mismatch_pct: float = 0.5
    max_candidates: int = 25
    # Both directions are research nominations.  Bearish movers are explicitly
    # barred from sizing previews until a separate borrow/SSR model exists.
    long_only: bool = False


@dataclass(frozen=True)
class NewsPolicy:
    # Covers Thursday-after-close events across Good Friday and the weekend.
    # Automatic qualification still requires verified causal timing, so this
    # broader discovery window does not make an old article decision-grade.
    lookback_hours: int = 96
    # Publication metadata can be a couple of seconds ahead because publisher
    # and broker clocks are not perfectly synchronized.  Retrieval itself is
    # never allowed after the decision clock.
    future_timestamp_tolerance_seconds: int = 2
    secondary_corroboration_window_hours: int = 12
    min_article_characters: int = 350
    max_documents_per_candidate: int = 8
    require_actual_document: bool = True
    # Secondary sources can support research but never automatic confirmation
    # in v0.  Search-result snippets never count as evidence.
    min_independent_secondary_sources: int = 2


@dataclass(frozen=True)
class ExecutionPolicy:
    account_value: float = float(BOOK_ACCOUNT_VALUE)
    classic_risk_bps: float = 10.0
    ep9m_catalyst_risk_bps: float = 7.5
    min_stage_price: float = 3.0
    min_stage_gap_pct: float = 4.0
    extension_warning_gap_pct: float = 20.0
    max_immediate_gap_pct: float = 25.0
    min_premarket_dollar_volume: float = 1_000_000.0
    max_spread_bps: float = 100.0
    max_chase_above_premarket_vwap_pct: float = 2.0
    max_stop_distance_pct: float = 20.0
    addv_participation: float = 0.0025
    premarket_volume_participation: float = 0.01
    displayed_size_participation: float = 0.25
    max_notional_pct_of_account: float = 2.0
    max_daily_risk_bps: float = 50.0
    max_daily_notional_pct_of_account: float = 10.0
    max_quantity: int = 25_000
    base_entry_slippage_bps: float = 8.0
    impact_coefficient_bps: float = 80.0
    max_entry_slippage_bps: float = 100.0
    stressed_exit_slippage_bps: float = 100.0
    event_gap_stress_pct: float = 2.0
    stop_buffer_bps: float = 10.0
    reference_entry_window_end_et: str = "09:35:00"


@dataclass(frozen=True)
class HistoricalPolicy:
    min_open_gap_pct: float = 10.0
    min_prior_close: float = 3.0
    min_prior_addv_63: float = 5_000_000.0
    min_prior_bars: int = 126
    max_prior_63d_return_pct: float = 20.0
    min_event_volume_rvol_20: float = 2.0
    first_event_lookback_sessions: int = 126
    horizons: tuple[int, ...] = (1, 5, 20, 60)


@dataclass(frozen=True)
class EPPolicy:
    policy_id: str = "ep-shadow-v0.2.0"
    policy_date: str = "2026-08-25"
    mode: str = "SHADOW_RESEARCH"
    live_actions_enabled: bool = False
    discovery: DiscoveryPolicy = field(default_factory=DiscoveryPolicy)
    news: NewsPolicy = field(default_factory=NewsPolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    historical: HistoricalPolicy = field(default_factory=HistoricalPolicy)

    def __post_init__(self) -> None:
        if self.live_actions_enabled:
            raise ValueError("EP shadow policy cannot enable live actions")
        if self.mode != "SHADOW_RESEARCH":
            raise ValueError("EP v0 only supports SHADOW_RESEARCH mode")
        if self.execution.max_daily_risk_bps <= 0:
            raise ValueError("daily risk cap must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = EPPolicy()
