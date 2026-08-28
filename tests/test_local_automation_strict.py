from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_sheet_missing_credentials_is_fatal_only_in_strict_mode(monkeypatch):
    import daily_portfolio_report as report

    monkeypatch.setattr(report, "get_google_client", lambda: None)
    monkeypatch.delenv("LOCAL_AUTOMATION_STRICT", raising=False)
    assert report.write_portfolio_to_sheet(None) is False

    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    with pytest.raises(RuntimeError, match="requires Google credentials"):
        report.write_portfolio_to_sheet(None)


def test_scan_staging_missing_credentials_is_fatal_in_strict_mode(monkeypatch):
    import daily_scan

    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    with pytest.raises(RuntimeError, match="No Google Sheets client"):
        daily_scan._staging_no_client("Order_Staging")


def test_event_upload_requires_matching_r2_size_in_strict_mode(tmp_path, monkeypatch):
    import event_sleeve

    payload = tmp_path / "state.json"
    payload.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    monkeypatch.setattr(event_sleeve, "upload_from_local", lambda *_: True)
    monkeypatch.setattr(
        event_sleeve,
        "head",
        lambda _key: {"ContentLength": payload.stat().st_size + 1},
    )

    with pytest.raises(RuntimeError, match="R2 verification failed"):
        event_sleeve._verified_automation_upload(payload, "event.json")


def test_indicator_cache_short_upload_is_fatal_in_strict_mode(monkeypatch):
    from scripts import build_indicator_cache as indicator

    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    monkeypatch.setattr(indicator, "_build_strategies", lambda *_: [])
    monkeypatch.setattr(indicator, "_build_master_dict", lambda *_: {})
    monkeypatch.setattr(indicator, "_snapshot_cache_mtimes", lambda: {})
    monkeypatch.setattr(indicator, "precompute_all_indicators", lambda *_: None)
    monkeypatch.setattr(indicator, "_detect_changed", lambda *_: ["one.parquet"])
    monkeypatch.setattr(indicator, "_upload_changed", lambda *_: 0)

    with pytest.raises(RuntimeError, match="upload incomplete"):
        indicator._run_pass("liquid", False, {}, {}, None, False)


def test_execution_email_false_is_fatal_in_strict_mode(monkeypatch):
    import daily_execution_report as report

    monkeypatch.setenv("LOCAL_AUTOMATION_STRICT", "1")
    monkeypatch.setenv("STATUS_TOKEN", "token")
    monkeypatch.setattr(
        report,
        "fetch_book",
        lambda *_: {"accounts": [{"key": "primary", "positions": [], "orders": []}]},
    )
    monkeypatch.setattr(report, "load_trend_state", lambda: {})
    monkeypatch.setattr(report, "load_event_state", lambda: {})
    monkeypatch.setattr(report, "send_email", lambda *_: False)
    monkeypatch.setattr(report.sys, "argv", ["daily_execution_report.py", "--force"])

    assert report.main() == 1


def test_all_local_producers_honor_strict_mode_contract():
    expected = (
        "daily_portfolio_report.py",
        "daily_execution_report.py",
        "daily_risk_report.py",
        "daily_scan.py",
        "verify_fills.py",
        "weekly_market_rundown.py",
        "event_sleeve.py",
        "trend_sleeve.py",
        "scripts/update_master_prices.py",
        "scripts/update_intraday_yfinance.py",
        "scripts/build_earnings_calendar.py",
        "scripts/build_analyst_grades.py",
        "scripts/build_macro_releases.py",
        "scripts/build_indicator_cache.py",
    )
    for relative in expected:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "LOCAL_AUTOMATION_STRICT" in text, relative
