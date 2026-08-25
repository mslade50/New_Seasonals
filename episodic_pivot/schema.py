"""Typed records shared by the EP discovery, research, and staging layers."""

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
    market_data_status: str = "LIVE"
    halted: bool = False
    halt_status: str = "UNKNOWN"
    tradeable: bool = False
    source: str = "IBKR_READ_ONLY"
    price_basis: str = "IBKR_TRADES"
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PremarketSnapshot":
        fields = cls.__dataclass_fields__
        values = {k: raw[k] for k in fields if k in raw}
        values["symbol"] = str(values.get("symbol", "")).strip().upper()
        values["observed_at"] = iso_utc(values["observed_at"])
        if values.get("premarket_metrics_at"):
            values["premarket_metrics_at"] = iso_utc(values["premarket_metrics_at"])
        return cls(**values)

    @property
    def gap_pct(self) -> float:
        return 100.0 * (self.last / self.previous_close - 1.0)

    @property
    def move_dollars(self) -> float:
        return self.last - self.previous_close

    @property
    def premarket_dollar_volume(self) -> float:
        return float(self.premarket_volume) * float(self.premarket_vwap)

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return 10_000.0 * (self.ask - self.bid) / mid if mid > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            gap_pct=self.gap_pct,
            move_dollars=self.move_dollars,
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NewsDocument":
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
class StagingPreview:
    candidate_id: str
    symbol: str
    action: str
    quantity: int
    order_type: str
    tif: str
    regular_hours_only: bool
    entry_limit: float
    initial_stop: float
    risk_per_share: float
    risk_dollars: float
    risk_bps: float
    notional: float
    expected_entry_slippage_bps: float
    max_entry_slippage_bps: float
    stressed_exit_slippage_bps: float
    binding_cap: str
    setup_type: str
    catalyst_type: str
    catalyst_summary: str
    catalyst_confidence: str
    materiality_score: int
    evidence_urls: tuple[str, ...]
    evidence_published_at: tuple[str, ...]
    contract_con_id: int | None
    primary_exchange: str
    activation_min_price: float
    activation_max_price: float
    max_opening_gap_pct: float
    entry_window_end_et: str
    requires_fresh_quote_at_release: bool
    requires_halt_recheck_at_release: bool
    requires_opening_gap_recheck: bool
    requires_contract_recheck_at_release: bool
    scan_source: str = "EP_SHADOW"
    approval: str = ""
    execute_on: str = ""
    live_eligible: bool = False

    def __post_init__(self) -> None:
        if self.live_eligible:
            raise ValueError("shadow staging previews cannot be live eligible")
        if self.order_type != "LMT":
            raise ValueError("shadow staging previews must use a limit order")
        if self.approval:
            raise ValueError("shadow staging previews must leave approval blank")
        if not self.requires_fresh_quote_at_release:
            raise ValueError("shadow previews must require a fresh release-time quote")
        if not self.requires_halt_recheck_at_release:
            raise ValueError("shadow previews must require a release-time halt recheck")
        if not self.requires_opening_gap_recheck:
            raise ValueError("shadow previews must require a release-time opening-gap recheck")
        if not self.requires_contract_recheck_at_release:
            raise ValueError("shadow previews must require a release-time contract recheck")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    generated_at: str
    candidates: list[Candidate] = field(default_factory=list)
    documents_by_candidate: dict[str, list[NewsDocument]] = field(default_factory=dict)
    decisions: list[QualificationDecision] = field(default_factory=list)
    previews: list[StagingPreview] = field(default_factory=list)
