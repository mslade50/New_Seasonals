"""Versioned doctrine and operating limits for the fundamental sleeve."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fundamental"
RAW_ROOT = DATA_ROOT / "raw"
SNAPSHOT_ROOT = DATA_ROOT / "snapshots"
CURRENT_ROOT = DATA_ROOT / "current"
RUN_ROOT = DATA_ROOT / "runs"
REPORT_ROOT = ROOT / "reports" / "fundamental"

POLICY_VERSION = "fundamental-sleeve.v1"
SCHEMA_VERSION = "fundamental-data.v1"


@dataclass(frozen=True)
class SleevePolicy:
    """Hard and draft controls expressed as percentages of total account NAV."""

    target_nav_pct: float = 27.0
    hard_nav_cap_pct: float = 30.0
    combined_slow_sleeves_hard_cap_pct: float = 30.0
    max_positions: int = 10
    normal_min_positions: int = 6
    normal_max_positions: int = 8
    starter_min_pct: float = 0.75
    starter_max_pct: float = 1.25
    core_min_pct: float = 2.50
    core_max_pct: float = 3.50
    single_name_hard_cap_pct: float = 4.50
    sector_hard_cap_pct: float = 9.0
    correlated_thesis_hard_cap_pct: float = 10.0
    single_name_scenario_loss_budget_bps: float = 75.0
    single_name_absolute_loss_review_bps: float = 100.0
    strict_cross_sleeve_ticker_lock: bool = True
    live_actions_enabled: bool = False


@dataclass(frozen=True)
class UniversePolicy:
    min_market_cap: float = 300_000_000.0
    min_dollar_volume_63d: float = 5_000_000.0
    min_price: float = 3.0
    min_annual_periods: int = 4
    max_filing_age_days: int = 550
    excluded_sectors: tuple[str, ...] = ("Financial Services", "Real Estate")
    excluded_industry_terms: tuple[str, ...] = (
        "biotechnology",
        "shell companies",
        "blank checks",
    )


@dataclass(frozen=True)
class BroadUniversePolicy:
    """Cheap discovery gates used before full statement enrichment.

    These limits define the research funnel, not the portfolio.  Specialist
    sectors stay visible in the broad universe even when the general-company
    score is not appropriate for them.
    """

    min_market_cap: float = 300_000_000.0
    min_price: float = 3.0
    min_current_volume: float = 100_000.0
    min_dollar_volume_63d: float = 5_000_000.0
    min_price_history_days: int = 252
    primary_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")
    default_enrichment_batch: int = 125
    max_enrichment_batch: int = 250
    refresh_after_days: int = 30


SLEEVE_POLICY = SleevePolicy()
UNIVERSE_POLICY = UniversePolicy()
BROAD_UNIVERSE_POLICY = BroadUniversePolicy()

# Cross-sectional research score.  The score prioritizes the diligence queue;
# it is never an investment recommendation or position-sizing instruction.
SCORE_WEIGHTS: dict[str, float] = {
    "roic": 12.0,
    "incremental_roic": 8.0,
    "gross_profitability": 6.0,
    "gross_margin_stability": 4.0,
    "revenue_cagr_3y": 5.0,
    "fcf_margin": 10.0,
    "cash_conversion": 5.0,
    "accrual_ratio": 5.0,
    "sbc_to_revenue": 4.0,
    "share_count_cagr_3y": 6.0,
    "net_debt_to_ebitda": 8.0,
    "fcf_positive_years": 7.0,
    "fcf_yield": 8.0,
    "earnings_yield": 4.0,
    "revenue_growth_change": 4.0,
    "fcf_margin_change": 4.0,
}

LOWER_IS_BETTER = {
    "gross_margin_stability",
    "accrual_ratio",
    "sbc_to_revenue",
    "share_count_cagr_3y",
    "net_debt_to_ebitda",
}

FMP_ENDPOINTS: tuple[str, ...] = (
    "profile",
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
    "key-metrics",
    "ratios",
    "analyst-estimates",
)


def policy_payload() -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "sleeve": asdict(SLEEVE_POLICY),
        "universe": asdict(UNIVERSE_POLICY),
        "broad_universe": asdict(BROAD_UNIVERSE_POLICY),
        "score_weights": SCORE_WEIGHTS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
    }


def validate_policy() -> None:
    if round(sum(SCORE_WEIGHTS.values()), 8) != 100.0:
        raise ValueError("SCORE_WEIGHTS must sum to 100")
    p = SLEEVE_POLICY
    if not (p.target_nav_pct < p.hard_nav_cap_pct <= p.combined_slow_sleeves_hard_cap_pct):
        raise ValueError("sleeve target/cap hierarchy is invalid")
    if p.normal_max_positions > p.max_positions:
        raise ValueError("normal position range exceeds max_positions")
    if p.core_max_pct > p.single_name_hard_cap_pct:
        raise ValueError("core size exceeds the single-name hard cap")
    if p.live_actions_enabled:
        raise ValueError("phase-one fundamental sleeve must remain research-only")
    u = BROAD_UNIVERSE_POLICY
    if u.default_enrichment_batch > u.max_enrichment_batch:
        raise ValueError("default enrichment batch exceeds the hard batch limit")


validate_policy()
