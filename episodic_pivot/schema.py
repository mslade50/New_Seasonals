"""Typed records shared by the EP discovery and research-preview layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return dt.astimezone(timezone.utc)


def iso_utc(value: str | datetime) -> str:
    return parse_timestamp(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PremarketSnapshot:
    symbol: str
    observed_at: str
    previous_close: float
    last: float
    bid: float
    ask: float
    premarket_volume: int
    premarket_open: float
    premarket_high: float
    premarket_low: float
    premarket_vwap: float
    prior_two_day_low: float
    atr_14: float
    avg_volume_20: float
    addv_63: float
    company_name: str = ""
    bid_size: int = 0
    ask_size: int = 0
    market_cap: float | None = None
    float_shares: float | None = None
    prior_63d_return_pct: float | None = None
    sessions_since_prior_ep: int | None = None
    market_data_status: str = "UNKNOWN"
    halted: bool = False
    halt_status: str = "UNKNOWN"
    tradeable: bool = False
    source: str = "IBKR_READ_ONLY"
    price_basis: str = "IBKR_TRADES"
    daily_price_basis: str = "UNVERIFIED"
    atr_reference_close: float | None = None
    daily_data_status: str = "UNRESOLVED"
    daily_data_observed_at: str | None = None
    daily_source_session: str = ""
    daily_source_symbol: str = ""
    daily_repaired_bar_count: int = 0
    contract_con_id: int | None = None
    primary_exchange: str = ""
    contract_identity_status: str = "UNRESOLVED"
    resolved_symbol: str = ""
    contract_sec_type: str = ""
    contract_currency: str = ""
    valid_exchanges: str = ""
    allowed_order_types: str = ""
    quote_previous_close: float | None = None
    premarket_metrics_at: str | None = None
    first_trigger_at: str | None = None
    session: str = "premarket"
    provider: str = ""
    saved_screen_id: str = ""
    target_session_date: str = ""
    reported_result_count: int | None = None
    extracted_row_count: int | None = None
    source_file_sha256: str = ""
    screen_exchange: str = ""
    security_type: str = "STK"
    reported_change_pct: float | None = None
    reported_move_dollars: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PremarketSnapshot:
        fields = cls.__dataclass_fields__
        values = {k: raw[k] for k in fields if k in raw}
        values["symbol"] = str(values.get("symbol", "")).strip().upper()
        values["observed_at"] = iso_utc(values["observed_at"])
        if values.get("premarket_metrics_at"):
            values["premarket_metrics_at"] = iso_utc(values["premarket_metrics_at"])
        if values.get("first_trigger_at"):
            values["first_trigger_at"] = iso_utc(values["first_trigger_at"])
        if values.get("daily_data_observed_at"):
            values["daily_data_observed_at"] = iso_utc(
                values["daily_data_observed_at"]
            )
        return cls(**values)

    @property
    def gap_pct(self) -> float:
        return 100.0 * (self.last / self.previous_close - 1.0)

    @property
    def move_dollars(self) -> float:
        return self.last - self.previous_close

    @property
    def prior_atr_pct(self) -> float | None:
        reference_close = (
            float(self.atr_reference_close)
            if self.atr_reference_close is not None
            else self.previous_close
        )
        return (
            100.0 * self.atr_14 / reference_close
            if reference_close > 0 and self.atr_14 > 0
            else None
        )

    @property
    def discovery_gap_pct(self) -> float:
        if self.reported_change_pct is not None:
            return float(self.reported_change_pct)
        return self.gap_pct

    @property
    def discovery_move_dollars(self) -> float:
        if self.reported_move_dollars is not None:
            return float(self.reported_move_dollars)
        return self.move_dollars

    @property
    def premarket_dollar_volume(self) -> float:
        reference_price = self.premarket_vwap if self.premarket_vwap > 0 else self.last
        return float(self.premarket_volume) * float(reference_price)

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return 10_000.0 * (self.ask - self.bid) / mid if mid > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            gap_pct=self.gap_pct,
            move_dollars=self.move_dollars,
            prior_atr_pct=self.prior_atr_pct,
            discovery_gap_pct=self.discovery_gap_pct,
            discovery_move_dollars=self.discovery_move_dollars,
            premarket_dollar_volume=self.premarket_dollar_volume,
            spread_bps=self.spread_bps,
        )
        return data


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    snapshot: PremarketSnapshot
    discovery_reasons: tuple[str, ...]
    discovery_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "snapshot": self.snapshot.to_dict(),
            "discovery_reasons": list(self.discovery_reasons),
            "discovery_warnings": list(self.discovery_warnings),
        }


@dataclass(frozen=True)
class NewsHit:
    title: str
    url: str
    published_at: str | None
    publisher: str = ""
    snippet: str = ""
    search_provider: str = ""


@dataclass(frozen=True)
class NewsDocument:
    title: str
    url: str
    canonical_url: str
    publisher: str
    published_at: str | None
    retrieved_at: str
    text_excerpt: str
    text_sha256: str
    source_tier: str
    fetch_status: str
    catalyst_types: tuple[str, ...] = ()
    adverse_flags: tuple[str, ...] = ()
    published_at_provenance: str = "UNKNOWN"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NewsDocument:
        values = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        values["catalyst_types"] = tuple(values.get("catalyst_types") or ())
        values["adverse_flags"] = tuple(values.get("adverse_flags") or ())
        if values.get("published_at"):
            values["published_at"] = iso_utc(values["published_at"])
        values["retrieved_at"] = iso_utc(values["retrieved_at"])
        return cls(**values)

    @property
    def is_actual_document(self) -> bool:
        return self.fetch_status == "FETCHED" and bool(self.text_sha256)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_actual_document"] = self.is_actual_document
        return data


@dataclass(frozen=True)
class CatalystAssessment:
    status: str
    catalyst_type: str
    summary: str
    confidence: str
    materiality_score: int = 0
    materiality_signals: tuple[str, ...] = ()
    evidence_urls: tuple[str, ...] = ()
    evidence_published_at: tuple[str, ...] = ()
    adverse_flags: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    primary_source_confirmed: bool = False
    publication_time_verified: bool = False
    trajectory_change_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationDecision:
    candidate_id: str
    symbol: str
    decision: str
    setup_type: str
    catalyst: CatalystAssessment
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchSizingPreview:
    candidate_id: str
    symbol: str
    research_direction: str
    max_preview_shares: int
    reference_entry_price: float
    hypothetical_stop_price: float
    modeled_risk_per_share: float
    modeled_risk_dollars: float
    modeled_risk_bps: float
    hypothetical_notional: float
    expected_entry_slippage_bps: float
    max_entry_slippage_bps: float
    stressed_exit_slippage_bps: float
    binding_constraint: str
    setup_type: str
    catalyst_type: str
    catalyst_summary: str
    catalyst_confidence: str
    materiality_score: int
    evidence_urls: tuple[str, ...]
    evidence_published_at: tuple[str, ...]
    contract_con_id: int | None
    primary_exchange: str
    reference_activation_min_price: float
    reference_activation_max_price: float
    max_reference_gap_pct: float
    reference_entry_window_end_et: str
    target_session_date: str
    quote_revalidation_required: bool
    halt_revalidation_required: bool
    gap_revalidation_required: bool
    contract_revalidation_required: bool
    research_source: str = "EP_SHADOW"
    preview_only: bool = True
    executable: bool = False
    broker_route: str = "NONE"
    order_submission_allowed: bool = False
    human_review_required: bool = True
    production_eligible: bool = False
    live_actions_enabled: bool = False
    record_type: str = field(default="EP_RESEARCH_SIZING_PREVIEW_V1", init=False)

    def __post_init__(self) -> None:
        if not self.preview_only or self.executable:
            raise ValueError("EP research sizing must remain a non-executable preview")
        if self.broker_route != "NONE" or self.order_submission_allowed:
            raise ValueError("EP research sizing cannot name or enable a broker route")
        if not self.human_review_required or self.production_eligible:
            raise ValueError(
                "EP research sizing always requires review and is never production eligible"
            )
        if self.live_actions_enabled:
            raise ValueError("EP research sizing cannot enable live actions")
        if self.record_type != "EP_RESEARCH_SIZING_PREVIEW_V1":
            raise ValueError("invalid EP research sizing record type")
        if self.research_direction not in {"LONG", "SHORT"}:
            raise ValueError("research_direction must be LONG or SHORT")
        if self.max_preview_shares < 1:
            raise ValueError("max_preview_shares must be positive")
        if not self.quote_revalidation_required:
            raise ValueError("research previews must require a fresh quote")
        if not self.halt_revalidation_required:
            raise ValueError("research previews must require a halt recheck")
        if not self.gap_revalidation_required:
            raise ValueError("research previews must require a gap recheck")
        if not self.contract_revalidation_required:
            raise ValueError("research previews must require a contract recheck")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    generated_at: str
    candidates: list[Candidate] = field(default_factory=list)
    documents_by_candidate: dict[str, list[NewsDocument]] = field(default_factory=dict)
    decisions: list[QualificationDecision] = field(default_factory=list)
    previews: list[ResearchSizingPreview] = field(default_factory=list)
