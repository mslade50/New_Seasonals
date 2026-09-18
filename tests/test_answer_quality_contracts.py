"""Daily Pitch judges standalone quality; a portfolio replay is not an account."""
from email import message_from_string

import pandas as pd
import pytest

from scripts import build_pitch_state as pitch


@pytest.mark.parametrize("offline", [False, True])
def test_pitch_state_excludes_portfolio_inputs(tmp_path, monkeypatch, offline):
    monkeypatch.setattr(pitch, "ROOT", tmp_path)
    prices = tmp_path / "prices.parquet"
    prices.touch()
    monkeypatch.setattr(pitch, "PRICES_PATH", prices)
    monkeypatch.setattr(pitch.pd, "read_parquet", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(pitch, "build_tape", lambda *a: {"freshest_bar": "2026-09-04"})
    monkeypatch.setattr(pitch, "build_risk", lambda *a: {"fragility": {"ma10_63d": 88}})
    for name in ("build_calendar", "build_earnings", "build_seasonality",
                 "build_research_index", "build_history", "build_watchlist", "build_pipeline"):
        monkeypatch.setattr(pitch, name, lambda *a: {})
    # A portfolio source must never be consulted, even when --no-book is absent.
    def forbidden(*args, **kwargs):
        pytest.fail("Daily Pitch must not load the portfolio or strategy book")
    monkeypatch.setattr(pitch, "build_book", forbidden, raising=False)
    result = pitch.build_state("2026-09-08", offline=offline)
    assert "book" not in result
    assert result["evaluation_basis"] == "standalone_idea_quality"
    assert result["risk"]["fragility"]["ma10_63d"] == 88


def test_pitch_risk_does_not_read_portfolio_exposure(tmp_path, monkeypatch):
    monkeypatch.setattr(pitch, "ROOT", tmp_path)
    original = type(tmp_path).read_text
    def read(path, *args, **kwargs):
        if path.name in {"exposure_state.json", "event_sleeve_state.json", "trend_sleeve_state.json"}:
            pytest.fail("Portfolio state must not affect Daily Pitch")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(tmp_path), "read_text", read)
    import pc_fear
    monkeypatch.setattr(pc_fear, "fear_state_asof", lambda *a: {
        "state": "OFF", "pct": 50, "data_date": "2026-09-04", "age_bd": 1})
    result = pitch.build_risk(pd.Timestamp("2026-09-08"), [])
    assert "exposure_leg" not in result
    assert result["pc_fear"]["state"] == "OFF"


def test_portfolio_email_identifies_replay_and_last_model_session(monkeypatch):
    import daily_portfolio_report as report

    # Exercise the real HTML/MIME formatter with a fake transport, never a live send.
    messages = []
    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def starttls(self):
            pass
        def login(self, *args):
            pass
        def sendmail(self, sender, receiver, message):
            messages.append(message_from_string(message))

    monkeypatch.setattr(report.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("EMAIL_USER", "fixture@example.invalid")
    monkeypatch.setenv("EMAIL_PASS", "fixture-only")
    dates = pd.bdate_range(end="2026-09-04", periods=40)
    pnl = pd.Series([100.0] * len(dates), index=dates)
    equity = 750_000 + pnl.cumsum()
    analysis = report.generate_sizing_recommendations(equity, pnl, 750_000)
    metrics = analysis["metrics"]
    assert metrics["model_asof"] == "2026-09-04"
    assert report.send_portfolio_email(None, pd.DataFrame(), analysis, metrics)
    assert len(messages) == 1
    assert "Theoretical Portfolio" in str(messages[0]["Subject"])
    html = messages[0].get_payload(0).get_payload(decode=True).decode("utf-8")
    assert "not broker holdings or account returns" in html
    assert "2026-09-04" in html
    assert "No modeled open positions" in html
    assert "Latest modeled session P&amp;L" in html
    assert "Today P&L" not in html
