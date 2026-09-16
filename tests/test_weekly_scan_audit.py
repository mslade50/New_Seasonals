import copy
import datetime as dt
import json
from pathlib import Path

import pytest

from live_scan_universe import exclude_retired_symbols
from scan_audit import archive_scan
from scripts.weekly_scan_audit import collect


def test_live_exclusion_preserves_historical_book_and_is_date_gated(tmp_path):
    book = [{"universe_tickers": ["LIVE", "OLD"]}]
    original = copy.deepcopy(book)
    registry = tmp_path / "exclusions.json"
    row = dict(ticker="OLD", status="confirmed_delisted", reason="completed merger",
               evidence_url="https://www.sec.gov/example", effective_from="2026-09-12")
    registry.write_text(json.dumps(dict(schema_version=1, exclusions=[row])))
    assert exclude_retired_symbols(book, asof="2026-09-11", registry_path=registry) == (book, [])
    result, removed = exclude_retired_symbols(book, asof="2026-09-12", registry_path=registry)
    assert result[0]["universe_tickers"] == ["LIVE"]
    assert removed == ["OLD"] and book == original
    row["status"] = "provider_missing"
    registry.write_text(json.dumps(dict(schema_version=1, exclusions=[row])))
    with pytest.raises(ValueError, match="confirmed delisting"):
        exclude_retired_symbols(book, asof="2026-09-12", registry_path=registry)


def test_archived_email_retains_sizing_and_redacts_exception_credentials(tmp_path, capsys):
    coverage = dict(unavailable=["BAD"], stale={}, exceptions=[
        dict(ticker="BAD", reason="token=private-value https://example.com/private?key=secret")])
    target = archive_scan(coverage, [dict(Ticker="ABC", Entry_Offset_ATR=float("nan"), OLV_Signal_Number=5,
                          OLV_Recency_Mult=1.0, OLV_Risk_Budget=3937.5)],
                          scope="all", bookend="pm", email_ok=True, root=tmp_path)
    text = target.read_text()
    assert "private-value" not in text and "https://" not in text
    assert "private-value" not in capsys.readouterr().out
    data = json.loads(text)
    assert data["signals"][0]["OLV_Signal_Number"] == 5
    assert data["signals"][0]["OLV_Risk_Budget"] == 3937.5
    assert data["signals"][0]["Entry_Offset_ATR"] is None
    assert coverage["exceptions"][0]["reason"].startswith("token=private-value")
    report = collect(tmp_path, data["date_et"], data["date_et"])
    assert list(report["ticker_candidates"]) == ["BAD"]


def test_weekly_audit_covers_failures_and_trading_holiday_without_inventing_delisting(tmp_path):
    folder = tmp_path / "logs" / "2026-09-08"
    folder.mkdir(parents=True)
    (folder / "scan.log").write_text(
        "success scan_am (ok)\n[WARN] 1 stale: OLD@2026-08-01\n"
        "ERROR: Quote not found for symbol: TEMP\n"
        "[OLV-EXIT] WARNING: inventory unavailable\n")
    report = collect(tmp_path, "2026-09-07", "2026-09-08")
    assert set(report["ticker_candidates"]) == {"OLD", "TEMP"}
    assert report["scan_success_evidence_gaps"] == [["2026-09-08", "scan_pm"]]
    assert any("inventory unavailable" in f["message"] for f in report["operational_findings"])
    assert "not proof of delisting" in report["ticker_candidates"]["TEMP"][0]["reason"]
