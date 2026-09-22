from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from episodic_pivot import short_watchlist as sw
from episodic_pivot import listed_universe as lu
from episodic_pivot import short_discovery as sd
from episodic_pivot.email_delivery import EmailPayload, EmailDeliveryError, with_short_watchlist
from trading_calendar import TRADING_DAY

TARGET = "2026-09-22"
NOW = datetime(2026, 9, 22, 12, 30, tzinfo=timezone.utc)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setattr(lu, "MIN_DIRECTORY_ROWS", 1)
    monkeypatch.setattr(sd, "MIN_ROWS", 1)
    monkeypatch.setattr(sw, "datetime", Clock)
    monkeypatch.setattr(sw, "capture_universe", lambda _target: universe())
    monkeypatch.setattr(sw, "capture_discovery", lambda _target, _settings: discovery())
    policy = sw.screen_from_discovery(universe(), discovery(), TARGET)
    monkeypatch.setattr(sw, "configured_screen", lambda: deepcopy(policy))
    return policy


def universe():
    bodies = {
        "nasdaqlisted.txt": "Symbol|Security Name|Test Issue|ETF\nTEST|Test Company Common Stock|N|N\nFile Creation Time: 0922202608:00|||",
        "otherlisted.txt": "ACT Symbol|Security Name|Exchange|NASDAQ Symbol|Test Issue|ETF\nFUND|Test ETF|N|FUND|N|Y\nFile Creation Time: 0922202608:00|||||",
    }
    return {"schema": lu.SCHEMA, "session_date": TARGET, "sources": {
        name: {"url": url, "content": bodies[name], "captured_at": "2026-09-22T12:15:00Z"}
        for name, url in lu.URLS.items()}}


def discovery():
    return {"schema": sd.SCHEMA, "session_date": TARGET, "url": sd.URL,
            "request": sd.request_for(sw.configured_screen()["settings"]),
            "captured_at": "2026-09-22T12:17:00Z",
            "response": {"totalCount": 1, "data": [{"s": "NASDAQ:TEST", "d": [
                "TEST", "Test Company", "NASDAQ", 68.85, 800000, 100000, 60,
                datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc).timestamp()]}]}}


def bars():
    dates = pd.date_range(end=pd.Timestamp(TARGET) - TRADING_DAY, periods=260, freq=TRADING_DAY)
    rows = []
    for i, day in enumerate(dates):
        close = 30 + i * 0.15
        rows.append({"date": str(day.date()), "Open": close - 0.05,
                     "High": close + 0.15, "Low": close - 0.15,
                     "Close": close, "Adj Close": close,
                     "Volume": 800_000 if i == 259 else 100_000})
    return rows


def news():
    return [{"symbol": "TEST", "company_name": "Test Company", "research_complete": True,
             "status": "CONTEXT_VERIFIED", "reviewed_at": "2026-09-22T12:28:00Z",
             "news_context": "The opened company release describes a new commercial contract.",
             "squeeze_risk": "New contract demand could sustain the rally; borrow was not checked.",
             "reason": "Company identity and the announcement were checked in the opened original.",
             "searches": [{"query": "TEST company news", "url": "https://www.google.com/search?q=TEST",
                           "observation_ref": "observed-search-1", "outcome": "RESULTS_READ",
                           "searched_at": "2026-09-22T12:22:00Z"}],
             "sources": [{"url": "https://example.com/release", "title": "Company contract release",
                          "capture_kind": "ARTICLE_BODY", "opened_at": "2026-09-22T12:25:00Z",
                          "published_at": "2026-09-21", "observation_ref": "observed-source-1",
                          "authority_basis": "The opened release identifies the company and contract.",
                          "content": "Test Company announced a new commercial contract. This is retained synthetic source evidence for a test, not live research."}]}]


def write_queue(run_dir, screen, *, source=None):
    run_dir.mkdir()
    source = source or {"TEST": {"bars": bars()}}
    candidates, coverage = sw.replay_prices(source, screen, TARGET)
    queue = {"schema": sw.SCHEMA, "session_date": TARGET, "captured_at": "2026-09-22T12:20:00Z",
             "source": "YFINANCE_AUTO_ADJUST_FALSE_REPAIR_TRUE_WITH_ADJ_CLOSE",
             "screen": screen, "prices_sha256": sw.digest(source), "universe_sha256": sw.digest(universe()),
             "discovery_sha256": sw.digest(discovery()),
             "candidates": candidates, "coverage": coverage}
    (run_dir / "universe.json").write_text(json.dumps(universe()))
    (run_dir / "discovery.json").write_text(json.dumps(discovery()))
    (run_dir / "prices.json").write_text(json.dumps(source))
    (run_dir / "queue.json").write_text(json.dumps(queue))
    return queue


def packet(tmp_path, screen, reviews=None):
    run_dir = tmp_path / "short"
    write_queue(run_dir, screen)
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps(news() if reviews is None else reviews))
    return sw.seal(run_dir, notes)


def base_payload(tmp_path):
    return EmailPayload(kind="morning", subject="EP report", html_body="<html><body>Validated EP</body></html>",
                        plain_body="Validated EP", attachments=(), receipt_path=tmp_path / "email_delivery.json",
                        source_sha256="a" * 64, metadata={"target_session_date": TARGET, "research_mode": "AGENT_GOOGLE_SEARCH_AND_READ"})


def test_exact_shared_strategy_metrics_without_trade_levels(screen):
    row = sw.analyze_bars("TEST", bars(), TARGET, screen)
    assert row["extension_score"] > 10
    assert row["relative_volume_63"] > 2
    assert row["status"] == "EXTENDED_SHORT_RESEARCH_WATCH"
    assert row["return_1d_pct"] > 0 and row["above_sma50_pct"] > 0
    assert row["borrow_status"] == "NOT_CHECKED"
    assert set(row).isdisjoint({"quantity", "order_type", "stop_price", "entry_price",
                               "open_confirmation_above", "reversal_watch_below_raw", "prior_high_raw"})


@pytest.mark.parametrize("field,metric", [("dist_min", "extension_score"), ("vol_thresh", "relative_volume_63")])
def test_strict_filter_boundaries(screen, field, metric):
    row = sw.analyze_bars("TEST", bars(), TARGET, screen)
    screen["settings"][field] = row[metric]
    assert sw.analyze_bars("TEST", bars(), TARGET, screen) is None


def test_split_dividend_basis_and_current_candle_do_not_change_setup(screen):
    source = bars()
    original = sw.analyze_bars("TEST", source, TARGET, screen)
    for row in source:
        row["Adj Close"] *= 0.9
    source.append({**source[-1], "date": TARGET, "Close": 999_999, "High": 999_999})
    adjusted = sw.analyze_bars("TEST", source, TARGET, screen)
    for field in ("extension_score", "atr_pct", "close_raw", "above_sma50_pct", "return_5d_pct"):
        assert adjusted[field] == pytest.approx(original[field])


@pytest.mark.parametrize("mutation", ["stale", "duplicate", "gap", "nan", "bad_ohlc"])
def test_unverified_history_rejected(screen, mutation):
    source = bars()
    if mutation == "stale": source.pop()
    elif mutation == "duplicate": source.append(source[-1])
    elif mutation == "gap": source.pop(-10)
    elif mutation == "nan": source[-1]["Adj Close"] = None
    else: source[-1]["High"] = 1
    with pytest.raises(ValueError):
        sw.analyze_bars("TEST", source, TARGET, screen)


def test_yfinance_multiindex_selects_ticker_before_flattening():
    a = pd.DataFrame(bars()).set_index("date")
    b = a.copy()
    b["Close"] += 100
    raw = pd.concat({"AAA": a, "BBB": b}, axis=1).swaplevel(0, 1, axis=1)
    raw.columns.names = ["Price", "Ticker"]
    assert sw.normalize_download(raw, "AAA")[-1]["Close"] == a.Close.iloc[-1]
    assert sw.normalize_download(raw, "BBB")[-1]["Close"] == b.Close.iloc[-1]


def test_missing_universe_rows_and_total_outage_not_empty(screen):
    with pytest.raises(ValueError, match="full frozen universe"):
        sw.replay_prices({}, screen, TARGET)
    with pytest.raises(ValueError, match="unavailable"):
        sw.replay_prices({"TEST": {"error": "download unavailable"}}, screen, TARGET)


def test_partial_coverage_disclosed_and_no_silent_truncation(screen):
    screen["universe"].append("MISSING")
    rows, coverage = sw.replay_prices({"TEST": {"bars": bars()}, "MISSING": {"error": "stale"}}, screen, TARGET)
    assert len(rows) == 1
    assert coverage["requested"] == 2 and coverage["verified"] == 1 and coverage["unverified"] == 1


def test_seal_replay_and_combined_mail(tmp_path, screen):
    path = packet(tmp_path, screen)
    combined = with_short_watchlist(base_payload(tmp_path), watchlist=path)
    assert "Validated EP" in combined.html_body
    assert "1 ATR short watch" in combined.subject
    assert "above SMA50" in combined.html_body and "1d" in combined.html_body
    assert "Opening-gap" not in combined.html_body and "Reversal level" not in combined.html_body
    assert combined.receipt_path == base_payload(tmp_path).receipt_path
    assert combined.source_sha256 != base_payload(tmp_path).source_sha256
    assert combined.metadata["short_watchlist"]["status"] == "REVIEWED"
    assert len(combined.attachments) == 2


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unfinished", "blocked", "snippet", "future", "no_source", "one_negative_search"])
def test_incomplete_or_unverified_news_blocked(tmp_path, screen, kind):
    reviews = news()
    if kind == "missing": reviews = []
    elif kind == "duplicate": reviews *= 2
    elif kind == "unfinished": reviews[0]["research_complete"] = False
    elif kind == "blocked": reviews[0]["searches"][0]["outcome"] = "BLOCKED"
    elif kind == "snippet": reviews[0]["sources"][0]["capture_kind"] = "SEARCH_SNIPPET"
    elif kind == "future": reviews[0]["sources"][0]["published_at"] = "2026-09-23"
    elif kind == "no_source": reviews[0]["sources"] = []
    else:
        reviews[0]["status"] = "NO_VERIFIED_NEWS"
        reviews[0]["sources"] = []
    with pytest.raises(ValueError):
        packet(tmp_path, screen, reviews)


@pytest.mark.parametrize("file", ["universe.json", "discovery.json", "prices.json", "queue.json", "watchlist.json", "watchlist.html", "watchlist.md"])
def test_tampering_fails_before_delivery(tmp_path, screen, file):
    path = packet(tmp_path, screen)
    target = path.parent / file
    if file.endswith(".json"):
        content = json.loads(target.read_text())
        if file == "universe.json": content["sources"].pop("otherlisted.txt")
        elif file == "discovery.json": content["response"]["data"] = []
        elif file == "prices.json": content["TEST"]["bars"][-1]["Close"] += 1
        elif file == "queue.json": content["candidates"] = []
        else: content["reviews"][0]["news_context"] = "Tampered unsupported claim about this company."
        target.write_text(json.dumps(content))
    else:
        target.write_text("unreviewed name")
    with pytest.raises(EmailDeliveryError):
        with_short_watchlist(base_payload(tmp_path), watchlist=path)


def test_session_mismatch_and_config_drift_rejected(tmp_path, screen):
    path = packet(tmp_path, screen)
    with pytest.raises(ValueError, match="session"):
        sw.load_watchlist(path, "2026-09-23")
    screen["settings"]["dist_min"] = 11
    with pytest.raises(ValueError, match="configured"):
        sw.load_watchlist(path, TARGET)


def test_unavailable_is_explicit_and_preserves_ep(tmp_path):
    result = with_short_watchlist(base_payload(tmp_path), unavailable=True)
    assert "Validated EP" in result.html_body
    assert "not a zero-candidate result" in result.html_body
    assert result.metadata["short_watchlist"]["status"] == "UNAVAILABLE"


def test_short_unavailable_cannot_bypass_original_ep_validator(tmp_path, monkeypatch):
    from scripts import send_episodic_pivot_email as sender

    def reject(*_args):
        raise EmailDeliveryError("EP review incomplete")
    monkeypatch.setattr(sender, "morning_payload", reject)
    assert sender.main(["--kind", "morning", "--artifact", str(tmp_path), "--require-agent-review",
                        "--short-screen-unavailable"]) == 2


def test_completed_no_verified_news_is_explicit(tmp_path, screen):
    reviews = news()
    review = reviews[0]
    review.update(status="NO_VERIFIED_NEWS", sources=[],
                  news_context="Completed issuer and announcement searches did not verify a catalyst.")
    review["searches"][0]["purpose"] = "COMPANY_NEWS"
    review["searches"].append({**review["searches"][0], "query": "Test Company investor relations release",
                              "purpose": "PRIMARY_ANNOUNCEMENT"})
    result = with_short_watchlist(base_payload(tmp_path), watchlist=packet(tmp_path, screen, reviews))
    assert "No news verified after completed searches" in result.html_body
    assert result.metadata["short_watchlist"]["count"] == 1


def test_verified_zero_candidate_report_and_live_capture_contract(tmp_path, screen):
    source = bars()
    source[-1]["Volume"] = 100_000
    frame = pd.DataFrame(source).set_index("date")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return frame

    run_dir = tmp_path / "capture"
    queue = sw.capture(run_dir, TARGET, download=download)
    assert calls[0]["end"] == TARGET and calls[0]["auto_adjust"] is False
    assert calls[0]["repair"] is True and calls[0]["tickers"] == ["TEST"]
    assert queue["candidates"] == [] and queue["coverage"]["verified"] == 1
    notes = tmp_path / "empty-notes.json"
    notes.write_text("[]")
    result = with_short_watchlist(base_payload(tmp_path), watchlist=sw.seal(run_dir, notes))
    assert "No setups among the verified histories" in result.html_body
    assert result.metadata["short_watchlist"]["status"] == "REVIEWED"


def test_capture_failure_is_retained_but_cannot_be_sealed(tmp_path, screen):
    def outage(**_kwargs):
        raise RuntimeError("source unavailable")

    run_dir = tmp_path / "outage"
    with pytest.raises(ValueError, match="unavailable"):
        sw.capture(run_dir, TARGET, download=outage)
    assert (run_dir / "prices.json").exists()
    assert not (run_dir / "queue.json").exists()


def test_shortcut_excludes_names_before_history_download(tmp_path, screen, monkeypatch):
    directory = universe()
    directory["sources"]["nasdaqlisted.txt"]["content"] = directory["sources"]["nasdaqlisted.txt"]["content"].replace(
        "TEST|", "LOW|Low Volume Company Common Stock|N|N\nTEST|")
    bulk = discovery()
    bulk["response"]["totalCount"] = 2
    bulk["response"]["data"].append({"s": "NASDAQ:LOW", "d": [
        "LOW", "Low Volume Company", "NASDAQ", 30, 200000, 1000000, 20,
        bulk["response"]["data"][0]["d"][-1]]})
    monkeypatch.setattr(sw, "capture_universe", lambda _target: directory)
    monkeypatch.setattr(sw, "capture_discovery", lambda *_args: bulk)

    def download(**kwargs):
        assert kwargs["tickers"] == ["TEST"]
        return pd.DataFrame(bars()).set_index("date")

    queue = sw.capture(tmp_path / "bounded", TARGET, download=download)
    assert queue["screen"]["universe_coverage"]["listed_equities"] == 2
    assert queue["screen"]["discovery_coverage"]["volume_filtered"] == 1
    assert len(queue["candidates"]) == 1


def test_empty_discovery_can_seal_without_yahoo_requests(tmp_path, screen, monkeypatch):
    bulk = discovery()
    bulk["response"]["data"][0]["d"][5] = 1000000
    monkeypatch.setattr(sw, "capture_discovery", lambda *_args: bulk)
    run_dir = tmp_path / "no-discovery-matches"
    queue = sw.capture(run_dir, TARGET, download=lambda **_kwargs: pytest.fail("Unnecessary history download"))
    assert queue["coverage"] == {"requested": 0, "verified": 0, "unverified": 0, "failures": {}}
    notes = tmp_path / "notes.json"
    notes.write_text("[]")
    assert sw.load_watchlist(sw.seal(run_dir, notes), TARGET)


def test_capacity_outage_retains_bulk_evidence_without_full_market_fallback(tmp_path, screen, monkeypatch):
    monkeypatch.setattr(sd, "MAX_HISTORY_TARGETS", 0)
    run_dir = tmp_path / "over-capacity"
    with pytest.raises(ValueError, match="exceeding capacity"):
        sw.capture(run_dir, TARGET, download=lambda **_kwargs: pytest.fail("Unbounded history download"))
    assert (run_dir / "discovery.json").exists()
    assert not (run_dir / "prices.json").exists()


def test_cli_refuses_outputs_outside_artifacts():
    from scripts import build_ep_short_watchlist as builder

    with pytest.raises(SystemExit) as error:
        builder.main(["--capture", "--run-dir", str(builder.ROOT / "tests" / "forbidden-output")])
    assert error.value.code == 2
