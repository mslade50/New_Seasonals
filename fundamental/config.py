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

POLICY_VERSION = "fundamental-sleeve.v2.1"
SCHEMA_VERSION = "fundamental-data.v2"
UNDERWRITE_SCHEMA_VERSION = "fundamental-underwrite.v2"
RUN_MANIFEST_SCHEMA_VERSION = "fundamental-run-manifest.v1"
TRIGGER_SCHEMA_VERSION = "fundamental-trigger.v1"
EVIDENCE_SCHEMA_VERSION = "fundamental-evidence.v1"
PORTFOLIO_SNAPSHOT_SCHEMA_VERSION = "fundamental-portfolio-snapshot.v1"
CONTROL_STATE_MAX_AGE_DAYS = 30


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
    max_price_age_days: int = 7
    primary_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")
    default_enrichment_batch: int = 125
    max_enrichment_batch: int = 250
    refresh_after_days: int = 30


@dataclass(frozen=True)
class UnderwritePolicy:
    """Minimum evidence and risk/reward gates for a reader-facing review.

    These are promotion gates for research, not instructions to allocate
    capital.  An archetype-specific underwrite may be more conservative, but
    it may not weaken these defaults without a versioned policy change.
    """

    max_visible_reviews: int = 3
    current_price_max_age_days: int = 7
    primary_source_max_age_days: int = 550
    estimate_snapshot_max_age_days: int = 45
    min_primary_sources: int = 2
    min_evidence_items: int = 3
    min_causal_links: int = 2
    min_proof_triggers: int = 1
    min_kill_conditions: int = 2
    min_base_case_cagr: float = 0.12
    min_discount_to_base_value: float = 0.20
    min_upside_downside_ratio: float = 1.50
    max_bear_case_downside: float = 0.40


SLEEVE_POLICY = SleevePolicy()
UNIVERSE_POLICY = UniversePolicy()
BROAD_UNIVERSE_POLICY = BroadUniversePolicy()
UNDERWRITE_POLICY = UnderwritePolicy()


# A screen is a recall device.  These independent axes expose why a company
# entered the diligence queue without pretending that one blended score proves
# a security-level mispricing.
SCREEN_AXIS_METRICS: dict[str, tuple[str, ...]] = {
    "business_quality": (
        "roic",
        "incremental_roic",
        "gross_profitability",
        "gross_margin_stability",
        "fcf_margin",
        "cash_conversion",
        "accrual_ratio",
        "fcf_positive_years",
    ),
    "owner_alignment": (
        "sbc_to_revenue",
        "share_count_cagr_3y",
        "net_debt_to_ebitda",
    ),
    "valuation_support": ("fcf_yield", "earnings_yield"),
    "fundamental_change": ("revenue_growth_change", "fcf_margin_change"),
}


ARCHETYPE_VALUATION_METHODS: dict[str, tuple[str, ...]] = {
    "quality_compounder": (
        "driver_dcf",
        "reverse_dcf",
        "normalized_owner_earnings",
    ),
    "derated_quality": (
        "driver_dcf",
        "reverse_dcf",
        "normalized_owner_earnings",
    ),
    "revision_inflection": (
        "scenario_dcf",
        "normalized_owner_earnings",
        "reverse_dcf",
    ),
    "self_help": (
        "scenario_dcf",
        "normalized_owner_earnings",
        "sum_of_parts",
    ),
    "capital_allocation": (
        "normalized_owner_earnings",
        "sum_of_parts",
        "reverse_dcf",
    ),
    "cash_yield_discount": (
        "normalized_owner_earnings",
        "reverse_dcf",
        "sum_of_parts",
    ),
    "cyclical_normalization": (
        "midcycle_earnings",
        "asset_nav",
        "scenario_dcf",
    ),
    "hidden_asset": ("sum_of_parts", "asset_nav", "event_value"),
    "post_event_overreaction": (
        "scenario_dcf",
        "normalized_owner_earnings",
        "event_value",
    ),
    "financial": ("excess_return", "tangible_book", "dividend_discount"),
    "reit": ("nav", "affo", "implied_cap_rate"),
    "biotech": ("risk_adjusted_npv", "net_cash", "dilution_scenario"),
    "commodity": ("asset_nav", "midcycle_earnings", "scenario_dcf"),
    "special_situation": ("event_value", "sum_of_parts", "liquidation_value"),
}

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
        "underwrite": asdict(UNDERWRITE_POLICY),
        "score_weights": SCORE_WEIGHTS,
        "screen_axes": SCREEN_AXIS_METRICS,
        "archetype_valuation_methods": ARCHETYPE_VALUATION_METHODS,
        "run_manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "trigger_schema_version": TRIGGER_SCHEMA_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "portfolio_snapshot_schema_version": PORTFOLIO_SNAPSHOT_SCHEMA_VERSION,
        "control_state_max_age_days": CONTROL_STATE_MAX_AGE_DAYS,
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
    if UNDERWRITE_POLICY.max_visible_reviews > 3:
        raise ValueError("reader-facing fundamental reviews may not exceed three")


validate_policy()
