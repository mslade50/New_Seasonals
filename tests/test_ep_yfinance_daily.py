from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.daily_prices import (
    YFINANCE_DAILY_PRICE_BASIS,
    calculate_prior_daily_metrics,
    enrich_snapshots_from_yfinance,
    extract_yfinance_symbol_frame,
)
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.email_delivery import morning_payload
from episodic_pivot.manifest import write_run_artifacts
from episodic_pivot.schema import NewsDocument, PremarketSnapshot
from scripts.capture_ep_daily_yfinance import main as yfinance_main
from trading_calendar import TRADING_DAY


TARGET_DATE = date(2026, 8, 25)
AS_OF = "2026-08-25T08:20:00-04:00"


def _snapshot(symbol: str, **overrides: object) -> PremarketSnapshot:
    values: dict[str, object] = {
        "symbol": symbol,
        "company_name": f"{symbol} Company",
        "observed_at": AS_OF,
        "previous_close": 10.0,
        "last": 11.0,
        "bid": 0.0,
        "ask": 0.0,
        "premarket_volume": 250_000,
        "premarket_open": 11.0,
        "premarket_high": 11.0,
        "premarket_low": 11.0,
        "premarket_vwap": 0.0,
        "prior_two_day_low": 0.0,
        "atr_14": 0.0,
        "avg_volume_20": 0.0,
        "addv_63": 0.0,
        "market_data_status": "BROWSER_EXPORT",
        "tradeable": False,
        "source": "TRADINGVIEW_BROWSER_EXPORT",
        "provider": "TRADINGVIEW",
        "session": "premarket",
        "saved_screen_id": "yftOvM3e",
        "target_session_date": TARGET_DATE.isoformat(),
        "reported_result_count": 1,
        "extracted_row_count": 1,
        "source_file_sha256": "a" * 64,
        "screen_exchange": "NYSE",
        "reported_change_pct": 10.0,
        "reported_move_dollars": 1.0,
    }
    values.update(overrides)
    return PremarketSnapshot(**values)  # type: ignore[arg-type]


def _daily_raw(symbols: list[str], *, include_target: bool = False) -> pd.DataFrame:
    previous_session = (pd.Timestamp(TARGET_DATE) - TRADING_DAY).date()
    dates = pd.date_range(end=previous_session, periods=126, freq=TRADING_DAY)
    if include_target:
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(TARGET_DATE)]))
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product(
        [fields, symbols], names=["Price", "Ticker"]
    )
    raw = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for symbol in symbols:
        raw[("Open", symbol)] = 10.0
        raw[("High", symbol)] = 10.25
        raw[("Low", symbol)] = 9.75
        raw[("Close", symbol)] = 10.0
        raw[("Volume", symbol)] = 1_000_000.0
    if include_target:
        raw.loc[pd.Timestamp(TARGET_DATE), ("High", symbols[0])] = 100.0
        raw.loc[pd.Timestamp(TARGET_DATE), ("Low", symbols[0])] = 1.0
    return raw


def _document() -> NewsDocument:
    return NewsDocument(
        title="Company reports results",
        url="https://www.sec.gov/Archives/example",
        canonical_url="https://www.sec.gov/Archives/example",
        publisher="SEC",
        published_at="2026-08-25T11:00:00Z",
        retrieved_at="2026-08-25T12:21:00Z",
        text_excerpt="Company reported quarterly results and raised full-year guidance. " * 8,
        text_sha256="b" * 64,
        source_tier="REGULATOR_PRIMARY",
        fetch_status="FETCHED",
        catalyst_types=("EARNINGS_GUIDANCE",),
        published_at_provenance="SEC_ACCEPTED_AT",
    )


def test_multiindex_dot_symbol_and_adjusted_atr_enrichment_preserve_tape_time():
    raw = _daily_raw(["ABC", "BRK-B"])
    calls: list[dict[str, object]] = []

    def download(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return raw

    result = enrich_snapshots_from_yfinance(
        [_snapshot("ABC"), _snapshot("BRK.B")],
        session_date=TARGET_DATE,
        download=download,
        fetched_at="2026-08-25T12:21:00Z",
    )

    assert result.verified_count == 2
    assert result.errors == ()
    assert calls[0]["auto_adjust"] is True
    assert calls[0]["repair"] is True
    assert calls[0]["end"] == TARGET_DATE
    by_symbol = {snapshot.symbol: snapshot for snapshot in result.snapshots}
    assert by_symbol["ABC"].observed_at == AS_OF
    assert by_symbol["BRK.B"].daily_source_symbol == "BRK-B"
    assert by_symbol["ABC"].daily_price_basis == YFINANCE_DAILY_PRICE_BASIS
    assert by_symbol["ABC"].atr_14 == pytest.approx(0.5)
    assert by_symbol["ABC"].prior_atr_pct == pytest.approx(5.0)
    assert by_symbol["ABC"].previous_close == pytest.approx(10.0)


def test_yfinance_repaired_source_bar_is_explicitly_stamped():
    raw = _daily_raw(["ABC"])
    raw[("Repaired?", "ABC")] = False
    raw.loc[raw.index[-3], ("Repaired?", "ABC")] = True
    result = enrich_snapshots_from_yfinance(
        [_snapshot("ABC")],
        session_date=TARGET_DATE,
        download=lambda **_kwargs: raw,
        fetched_at="2026-08-25T12:21:00Z",
    )
    snapshot = result.snapshots[0]
    assert snapshot.daily_repaired_bar_count == 1
    assert snapshot.daily_data_status == "VERIFIED_WITH_YFINANCE_REPAIR"


def test_event_session_bar_is_excluded_from_prior_atr():
    frame = extract_yfinance_symbol_frame(
        _daily_raw(["ABC"], include_target=True), "ABC"
    )
    metrics = calculate_prior_daily_metrics(frame, TARGET_DATE)
    assert metrics["atr_14"] == pytest.approx(0.5)
    assert metrics["daily_source_session"] == "2026-08-24"


def test_partial_yfinance_data_retains_unresolved_discovery_row():
    raw = _daily_raw(["GOOD", "MISS"])
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        raw[(field, "MISS")] = float("nan")

    result = enrich_snapshots_from_yfinance(
        [_snapshot("GOOD"), _snapshot("MISS")],
        session_date=TARGET_DATE,
        download=lambda **_kwargs: raw,
        fetched_at="2026-08-25T12:21:00Z",
    )

    assert result.requested_count == 2
    assert result.verified_count == 1
    assert {snapshot.symbol for snapshot in result.snapshots} == {"GOOD", "MISS"}
    missing = next(item for item in result.snapshots if item.symbol == "MISS")
    assert missing.atr_14 == 0.0
    assert missing.daily_price_basis == "UNVERIFIED"
    assert missing.daily_data_status == "NO_SYMBOL_DATA"
    assert result.errors[0].symbol == "MISS"


def test_yfinance_atr_qualifies_news_without_enabling_sizing():
    high = _snapshot(
        "HIGH",
        atr_14=0.5,
        atr_reference_close=10.0,
        daily_price_basis=YFINANCE_DAILY_PRICE_BASIS,
        daily_data_status="VERIFIED",
        daily_source_session="2026-08-24",
    )
    low = _snapshot(
        "LOW",
        atr_14=0.4,
        atr_reference_close=10.0,
        daily_price_basis=YFINANCE_DAILY_PRICE_BASIS,
        daily_data_status="VERIFIED",
        daily_source_session="2026-08-24",
    )
    result = run_shadow_pipeline(
        [high, low],
        as_of=AS_OF,
        target_session_date=TARGET_DATE,
        policy=DEFAULT_POLICY,
        offline_documents={"HIGH": [_document()], "LOW": [_document()]},
        offline_documents_verified=True,
    )
    candidates = {candidate.snapshot.symbol: candidate for candidate in result.candidates}
    decisions = {decision.symbol: decision for decision in result.decisions}
    assert result.documents_by_candidate[candidates["HIGH"].candidate_id]
    assert result.documents_by_candidate[candidates["LOW"].candidate_id] == []
    assert "NEWS_RESEARCH_SKIPPED_PRIOR_ATR" not in decisions["HIGH"].blockers
    assert "NEWS_RESEARCH_SKIPPED_PRIOR_ATR" in decisions["LOW"].blockers
    assert result.previews == []


def test_yfinance_research_run_renders_normal_focused_morning_email(tmp_path: Path):
    high = _snapshot(
        "HIGH",
        atr_14=0.5,
        atr_reference_close=10.0,
        daily_price_basis=YFINANCE_DAILY_PRICE_BASIS,
        daily_data_status="VERIFIED",
        daily_source_session="2026-08-24",
    )
    low = _snapshot(
        "LOW",
        atr_14=0.4,
        atr_reference_close=10.0,
        daily_price_basis=YFINANCE_DAILY_PRICE_BASIS,
        daily_data_status="VERIFIED",
        daily_source_session="2026-08-24",
    )
    result = run_shadow_pipeline(
        [high, low],
        as_of=AS_OF,
        target_session_date=TARGET_DATE,
        policy=DEFAULT_POLICY,
        offline_documents={"HIGH": [_document()], "LOW": [_document()]},
        offline_documents_verified=True,
    )
    run_dir = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["atr_qualified"] == 1
    assert manifest["counts"]["news_research_selected"] == 1
    assert manifest["counts"]["execution_data_verified"] == 0
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Execution data unavailable or unverified" in html
    assert "5.00%" in html
    payload = morning_payload(run_dir)
    assert "1 researched, 1 ATR-qualified" in payload.subject
    assert payload.metadata["research_sizing_previews"] == 0


def test_yfinance_capture_cli_is_no_network_no_write_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "premarket.json"
    snapshot = _snapshot("ABC")
    source.write_text(
        json.dumps(
            {
                "provider": "TRADINGVIEW",
                "captured_at": snapshot.observed_at,
                "target_session_date": TARGET_DATE.isoformat(),
                "reported_result_count": 1,
                "extracted_row_count": 1,
                "result_count_verified": True,
                "snapshots": [snapshot.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"

    def unexpected_download(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("dry run must not contact yfinance")

    monkeypatch.setattr("yfinance.download", unexpected_download)
    assert (
        yfinance_main(
            ["--snapshot", str(source), "--output", str(output)]
        )
        == 0
    )
    assert not output.exists()
