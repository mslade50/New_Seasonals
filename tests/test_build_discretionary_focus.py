import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_discretionary_focus as builder


def _prices(ticker: str, *, volatile: bool, end: str = "2026-08-25") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=270)
    if volatile:
        closes = np.full(len(dates), 10.0)
        closes[-22:-6] = np.linspace(10.2, 14.2, 16)
        closes[-6:] = [14.2, 15.0, 14.3, 15.1, 14.4, 14.8]
    else:
        closes = np.full(len(dates), 20.0)
    rows = []
    for index, (date, close) in enumerate(zip(dates, closes)):
        if volatile:
            spread = 0.035 if index < len(dates) - 5 else 0.03
        else:
            spread = 0.004
        rows.append(
            {
                "ticker": ticker,
                "date": date,
                "Open": close * 0.995,
                "High": close * (1.0 + spread),
                "Low": close * (1.0 - spread),
                "Close": close,
                "Volume": 2_500_000,
            }
        )
    return pd.DataFrame(rows)


def _symbols() -> dict[str, dict]:
    return {
        "FAST": {
            "ticker": "FAST",
            "company_name": "Fast Company",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "market_cap": 2_000_000_000,
        },
        "SLOW": {
            "ticker": "SLOW",
            "company_name": "Slow Company",
            "exchange": "NYSE",
            "sector": "Industrials",
            "market_cap": 2_000_000_000,
        },
    }


def test_tradingview_adr_and_weekly_volatility_formulas_are_mirrored() -> None:
    dates = pd.bdate_range(end="2026-08-25", periods=270)
    bars = pd.DataFrame(
        {
            "ticker": "FORMULA",
            "date": dates,
            "Open": 10.0,
            "High": 11.0,
            "Low": 9.0,
            "Close": 10.0,
            "Volume": 1_000_000,
        }
    )

    metrics = builder._ticker_metrics(bars, dt.date(2026, 8, 25))

    assert metrics is not None
    assert metrics["adr14_pct"] == pytest.approx(20.0)
    assert metrics["volatility1w_pct"] == pytest.approx((11.0 - 9.0) / 9.0 * 100)
    assert metrics["performance1w_pct"] == pytest.approx(0.0)
    assert metrics["performance1m_pct"] == pytest.approx(0.0)
    assert metrics["performance3m_pct"] == pytest.approx(0.0)


def test_recent_ipo_history_remains_eligible_for_the_armed_math() -> None:
    bars = _prices("IPO", volatile=True).tail(90).copy()
    metrics = builder._ticker_metrics(bars, dt.date(2026, 8, 25))

    assert metrics is not None
    assert metrics["high_lookback_sessions"] == 90
    assert metrics["distance_available_high_pct"] >= 0


def test_xnys_early_close_sets_the_focus_expiry() -> None:
    expiry = builder.session_expiry(dt.date(2026, 11, 27))  # Black Friday
    assert expiry.isoformat() == "2026-11-27T13:15:00-05:00"


def test_armed_screen_requires_volatility_chart_and_known_earnings(tmp_path) -> None:
    prices = pd.concat([_prices("FAST", volatile=True), _prices("SLOW", volatile=False)])
    earnings = tmp_path / "earnings.parquet"
    pd.DataFrame(
        {
            "ticker": ["FAST", "SLOW"],
            "date": [pd.Timestamp("2026-09-22"), pd.Timestamp("2026-09-22")],
        }
    ).to_parquet(earnings, index=False)

    rows, counts = builder.technical_screen(
        prices,
        _symbols(),
        earnings,
        as_of=dt.date(2026, 8, 25),
    )

    assert [row["ticker"] for row in rows] == ["FAST"]
    assert rows[0]["technical_state"] == "ARMED"
    assert rows[0]["earnings_td"] > 5
    assert "RVOL-at-Time" in rows[0]["trigger"]
    assert counts["identity_liquidity_gate"] == 2
    assert counts["earnings_gate"] == 1


def test_missing_earnings_fails_the_standard_lane(tmp_path) -> None:
    prices = _prices("FAST", volatile=True)
    earnings = tmp_path / "earnings.parquet"
    pd.DataFrame(
        {"ticker": ["SLOW"], "date": [pd.Timestamp("2026-09-22")]}
    ).to_parquet(earnings, index=False)

    rows, counts = builder.technical_screen(
        prices,
        _symbols(),
        earnings,
        as_of=dt.date(2026, 8, 25),
    )

    assert rows == []
    assert counts["armed_technical_gate"] == 1
    assert counts["earnings_gate"] == 0


def test_optional_overflow_calendar_extends_earnings_coverage(tmp_path) -> None:
    prices = _prices("FAST", volatile=True)
    earnings = tmp_path / "earnings.parquet"
    overflow = tmp_path / "earnings-overflow.parquet"
    pd.DataFrame(
        {"ticker": ["SLOW"], "date": [pd.Timestamp("2026-09-22")]}
    ).to_parquet(earnings, index=False)
    pd.DataFrame(
        {"ticker": ["FAST"], "date": [pd.Timestamp("2026-09-22")]}
    ).to_parquet(overflow, index=False)

    rows, counts = builder.technical_screen(
        prices,
        _symbols(),
        earnings,
        earnings_overflow_path=overflow,
        as_of=dt.date(2026, 8, 25),
    )

    assert [row["ticker"] for row in rows] == ["FAST"]
    assert counts["earnings_gate"] == 1


def test_earnings_blackout_is_measured_from_the_focus_session(tmp_path) -> None:
    prices = _prices("FAST", volatile=True)
    earnings = tmp_path / "earnings.parquet"
    # Six trading days from the Tuesday price cutoff, but only five from the
    # Wednesday session for which the shortlist is valid.
    pd.DataFrame(
        {"ticker": ["FAST"], "date": [pd.Timestamp("2026-09-02")]}
    ).to_parquet(earnings, index=False)

    rows, counts = builder.technical_screen(
        prices,
        _symbols(),
        earnings,
        as_of=dt.date(2026, 8, 25),
        valid_for=dt.date(2026, 8, 26),
    )

    assert rows == []
    assert counts["armed_technical_gate"] == 1
    assert counts["earnings_gate"] == 0


def _technical_row() -> dict:
    return {
        "ticker": "FAST",
        "company_name": "Fast Company",
        "causal_cluster": "Technology",
        "technical_state": "ARMED",
        "research_lane": "standard_company",
        "earnings_td": -18,
        "invalidation": "Pass on a failed breakout.",
        "technical": {
            "adr14_pct": 5.1,
            "performance1m_pct": 35.0,
            "performance3m_pct": 40.0,
            "performance1w_pct": 2.0,
        },
    }


def _fundamental(**overrides) -> dict:
    row = {
        "ticker": "FAST",
        "research_score": 82,
        "score_coverage_pct": 92,
        "statement_periods": 5,
        "latest_fiscal_date": "2025-12-31",
        "latest_fcf_positive": True,
        "source_current": True,
        "revenue_growth_change": 0.08,
        "fcf_margin_change": 0.01,
        "hard_exclusion_reason": "",
        "research_control": "",
        "research_suppressed": False,
        "share_count_cagr_3y": 0.01,
        "net_debt_to_ebitda": 0.5,
        "research_priority": "A - immediate research candidate",
        "issuer_cik": "1234567",
        "latest_accepted_at": "2026-08-20",
        "sec_source": {
            "source_id": "sec-fast-10q",
            "label": "FAST 10-Q filed 2026-08-20",
            "url": "https://www.sec.gov/Archives/edgar/data/1234567/fast-10q.htm",
            "as_of": "2026-08-20",
            "primary": True,
        },
    }
    row.update(overrides)
    return row


def test_fundamental_change_can_support_attention_without_inventing_news() -> None:
    output = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
    )

    assert set(output) == {"FAST"}
    assert "Annual revenue growth accelerated" in output["FAST"]["catalyst"]
    assert "already reflects" in output["FAST"]["priced_in"]
    assert output["FAST"]["sources"][0]["url"].startswith("https://")


def test_dilution_or_negative_news_removes_candidate() -> None:
    news = {
        "FAST": [
            {
                "symbol": "FAST",
                "publishedDate": "2026-08-24",
                "title": "Fast Company announces public offering",
                "url": "https://example.com/offering",
                "_endpoint": "news/press-releases",
            }
        ]
    }
    negative = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker=news,
    )
    dilution = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental(share_count_cagr_3y=0.12)},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
    )

    assert negative == {}
    assert dilution == {}


def test_generic_official_release_cannot_rescue_weak_fundamentals() -> None:
    news = {
        "FAST": [
            {
                "symbol": "FAST",
                "publishedDate": "2026-08-24",
                "title": "Fast Company to present at an investor conference",
                "url": "https://example.com/conference",
                "_endpoint": "news/press-releases",
            }
        ]
    }
    weak = _fundamental(
        research_score=40,
        revenue_growth_change=-0.05,
        fcf_margin_change=-0.01,
    )

    output = builder.research_overlay(
        [_technical_row()],
        {"FAST": weak},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker=news,
    )

    assert output == {}


def test_positive_official_release_can_support_attention() -> None:
    news = {
        "FAST": [
            {
                "symbol": "FAST",
                "publishedDate": "2026-08-24",
                "title": "Fast Company raises guidance after record revenue",
                "url": "https://example.com/guide",
                "_endpoint": "news/press-releases",
            }
        ]
    }
    weak = _fundamental(
        research_score=40,
        revenue_growth_change=-0.05,
        fcf_margin_change=-0.01,
    )

    output = builder.research_overlay(
        [_technical_row()],
        {"FAST": weak},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker=news,
    )

    assert set(output) == {"FAST"}
    assert output["FAST"]["sources"][0]["primary"] is True


def test_current_float_gate_is_fail_closed() -> None:
    accepted = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
        float_by_ticker={"FAST": 199_000_000},
        market_cap_by_ticker={"FAST": 2_000_000_000},
    )
    rejected = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
        float_by_ticker={"FAST": 201_000_000},
        market_cap_by_ticker={"FAST": 2_000_000_000},
    )

    assert set(accepted) == {"FAST"}
    assert rejected == {}


def test_current_market_cap_gate_is_fail_closed() -> None:
    accepted = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
        float_by_ticker={"FAST": 100_000_000},
        market_cap_by_ticker={"FAST": 2_000_000_000},
    )
    rejected = builder.research_overlay(
        [_technical_row()],
        {"FAST": _fundamental()},
        fundamental_as_of="2026-08-20",
        as_of=dt.date(2026, 8, 25),
        news_by_ticker={},
        float_by_ticker={"FAST": 100_000_000},
        market_cap_by_ticker={"FAST": 30_000_000_000},
    )

    assert set(accepted) == {"FAST"}
    assert rejected == {}


def test_fundamental_snapshot_staleness_fails_closed(tmp_path) -> None:
    path = tmp_path / "fundamentals.json"
    path.write_text(
        json.dumps(
            {
                "health": {"as_of": "2026-01-01"},
                "candidates": [_fundamental()],
            }
        ),
        encoding="utf-8",
    )

    try:
        builder.load_fundamental_research(path, as_of=dt.date(2026, 8, 25))
    except builder.FocusBuildError as exc:
        assert "max is" in str(exc)
    else:
        raise AssertionError("stale fundamental snapshot should fail closed")


def test_session_for_run_uses_today_before_market_and_next_after_latest_close() -> None:
    now = dt.datetime(2026, 8, 26, 12, 30, tzinfo=dt.timezone.utc)
    assert builder.session_for_run(dt.date(2026, 8, 25), now) == dt.date(2026, 8, 26)
    assert builder.session_for_run(dt.date(2026, 8, 26), now) == dt.date(2026, 8, 27)


def test_price_cutoff_must_be_immediately_prior_to_focus_session() -> None:
    builder.require_fresh_price_cutoff(
        dt.date(2026, 9, 4), dt.date(2026, 9, 8)
    )  # Labor Day is skipped.
    with pytest.raises(builder.FocusBuildError, match="expected 2026-08-25"):
        builder.require_fresh_price_cutoff(
            dt.date(2026, 8, 24), dt.date(2026, 8, 26)
        )
    with pytest.raises(builder.FocusBuildError, match="not an NYSE session"):
        builder.require_fresh_price_cutoff(
            dt.date(2026, 8, 28), dt.date(2026, 8, 29)
        )


def test_symbol_master_supplies_identity_not_current_market_cap(tmp_path) -> None:
    path = tmp_path / "symbols.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "FAST",
                "company_name": "Fast Company",
                "exchange": "NASDAQ",
                "market_cap": 2_000_000_000,
                "as_of": "2026-08-01",
            }
        ]
    ).to_parquet(path, index=False)

    symbols = builder.load_symbols(path)

    assert symbols["FAST"]["company_name"] == "Fast Company"
    assert symbols["FAST"]["market_cap"] == 2_000_000_000


def test_armed_mirror_requires_price_above_each_average_not_average_order(
    tmp_path, monkeypatch
) -> None:
    as_of = dt.date(2026, 8, 25)
    prices = pd.DataFrame([{"ticker": "FAST", "date": pd.Timestamp(as_of)}])
    metrics = {
        "price": 14.0,
        "adr14_pct": 5.0,
        "atr14_pct": 5.0,
        "avg_dollar_volume20": 30_000_000.0,
        "avg_volume60": 2_000_000.0,
        "latest_volume": 2_000_000.0,
        "relative_volume": 1.2,
        "performance1w_pct": 1.0,
        "performance1m_pct": 35.0,
        "performance3m_pct": 40.0,
        "volatility1w_pct": 5.0,
        "sma20": 12.0,
        "sma50": 13.0,
        "sma200": 10.0,
        "pivot_distance_pct": 2.0,
        "distance_available_high_pct": 3.0,
        "high_lookback_sessions": 252,
        "compression_ratio": 0.7,
        "close_location20": 0.8,
        "volume_ratio5_20": 1.0,
    }
    monkeypatch.setattr(
        builder, "_ticker_metrics", lambda *args, **kwargs: dict(metrics)
    )
    monkeypatch.setattr(
        builder,
        "load_earnings_dates_map",
        lambda **kwargs: {"FAST": np.array(["2026-09-22"], dtype="datetime64[D]")},
    )

    rows, _ = builder.technical_screen(
        prices,
        _symbols(),
        tmp_path / "unused.parquet",
        as_of=as_of,
    )

    assert [row["ticker"] for row in rows] == ["FAST"]


def test_attention_trim_rejects_a_loose_far_from_pivot_setup(tmp_path, monkeypatch) -> None:
    as_of = dt.date(2026, 8, 25)
    prices = pd.DataFrame([{"ticker": "FAST", "date": pd.Timestamp(as_of)}])
    metrics = {
        "price": 14.0,
        "adr14_pct": 6.0,
        "atr14_pct": 6.0,
        "avg_dollar_volume20": 30_000_000.0,
        "avg_volume60": 2_000_000.0,
        "latest_volume": 2_000_000.0,
        "relative_volume": 1.2,
        "performance1w_pct": 1.0,
        "performance1m_pct": 40.0,
        "performance3m_pct": 50.0,
        "volatility1w_pct": 6.0,
        "sma20": 12.0,
        "sma50": 13.0,
        "sma200": None,
        "pivot_distance_pct": 14.0,
        "distance_available_high_pct": 15.0,
        "high_lookback_sessions": 100,
        "compression_ratio": 1.2,
        "close_location20": 0.55,
        "volume_ratio5_20": 1.5,
    }
    monkeypatch.setattr(builder, "_ticker_metrics", lambda *a, **k: dict(metrics))
    monkeypatch.setattr(
        builder,
        "load_earnings_dates_map",
        lambda **kwargs: {"FAST": np.array(["2026-09-22"], dtype="datetime64[D]")},
    )

    rows, counts = builder.technical_screen(
        prices, _symbols(), tmp_path / "unused.parquet", as_of=as_of
    )

    assert rows == []
    assert counts["armed_technical_gate"] == 1
    assert counts["setup_readiness_gate"] == 0


def _payload_technical(ticker: str, cluster: str, rank: int) -> dict:
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Company",
        "causal_cluster": cluster,
        "technical_gate": "PASS",
        "liquidity_gate": "PASS",
        "setup_quality": 80 - rank,
        "screen_rank": rank,
        "observed_at": "2026-08-26T12:34:00Z",
        "earnings_td": 14,
        "event_date": "2026-09-15",
        "technical": {
            "adr14_pct": 5.2,
            "performance1w_pct": 1.0,
            "performance1m_pct": 36.0,
            "performance3m_pct": 45.0,
        },
        "setup": "The daily-bar Armed mirror is intact beneath a live breakout trigger.",
        "trigger": "TradingView confirms a new one-month high with RVOL-at-Time at least 2.0.",
        "invalidation": "The breakout loses its live pivot or relative volume never confirms.",
    }


def _payload_research(ticker: str, cluster: str, rank: int) -> dict:
    return {
        "ticker": ticker,
        "research_gate": "PASS",
        "attention_rank": rank,
        "company_name": f"{ticker} Company",
        "why_now": "A current operating change supports the technical setup.",
        "catalyst": "The issuer raised its forward operating outlook.",
        "variant_wedge": "Consensus does not reflect another period of acceleration.",
        "priced_in": "The recent momentum reflects the current guide, but not another raise.",
        "next_proof": "The next operating update must sustain the acceleration.",
        "source_current": True,
        "catalyst_reaches_economics": True,
        "unresolved_financing_risk": False,
        "unresolved_dilution_risk": False,
        "unresolved_restatement_risk": False,
        "kill_condition": "Forward growth falls below the market-implied path.",
        "causal_cluster": cluster,
        "sources": [
            {
                "source_id": f"{ticker.lower()}-ir",
                "label": f"{ticker} investor relations release",
                "url": f"https://example.com/{ticker.lower()}/release",
                "as_of": "2026-08-25",
                "primary": True,
            }
        ],
    }


@pytest.mark.parametrize(
    ("candidate_count", "expected_count", "expected_status"),
    [(0, 0, "NO_QUALIFIED_SETUP"), (1, 1, "READY"), (2, 2, "READY"), (3, 2, "READY")],
)
def test_build_payload_is_contract_valid_for_zero_one_or_two_names(
    candidate_count, expected_count, expected_status
) -> None:
    tickers = ["AAA", "BBB", "CCC"][:candidate_count]
    technical_rows = [
        _payload_technical(ticker, f"cluster-{ticker}", rank)
        for rank, ticker in enumerate(tickers, start=1)
    ]
    research = {
        ticker: _payload_research(ticker, f"cluster-{ticker}", rank)
        for rank, ticker in enumerate(tickers, start=1)
    }
    generated_at = dt.datetime(2026, 8, 26, 12, 35, tzinfo=dt.timezone.utc)

    payload = builder.build_payload(
        technical_rows=technical_rows,
        research_by_ticker=research,
        counts={"measured": candidate_count, "earnings_gate": candidate_count},
        as_of=dt.date(2026, 8, 25),
        valid_for=dt.date(2026, 8, 26),
        phase="FINAL",
        generated_at=generated_at,
        fundamental_as_of="2026-08-25",
        news_status="CURRENT",
    )

    assert payload["status"] == expected_status
    assert len(payload["focus"]) == expected_count
    assert builder.validate_payload(payload, now=generated_at) == payload
    assert payload["provenance"]["screen_captured_at"] == "2026-08-25T16:00:00-04:00"
    if candidate_count == 3:
        assert payload["screen_summary"]["rejected_counts"]["attention_cap"] == 1


def test_fmp_client_uses_current_news_and_share_structure_contract() -> None:
    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def get(self, url, *, params, timeout):
            calls.append((url, dict(params), timeout))
            if url.endswith("/shares-float"):
                return Response(
                    [{"floatShares": 125_000_000, "outstandingShares": 150_000_000}]
                )
            return Response(
                [
                    {
                        "symbol": "FAST",
                        "title": "Fast Company raises guidance",
                        "publishedDate": "2026-08-25",
                    }
                ]
            )

    client = builder.FMPNewsClient("test-key", session=Session())
    news = client.fetch("FAST")
    shares = client.fetch_share_structure("FAST")

    assert len(news) == 2
    assert shares == {
        "float_shares": 125_000_000,
        "outstanding_shares": 150_000_000,
    }
    assert all(call[1]["symbols"] == "FAST" for call in calls[:2])
    assert calls[2][1]["symbol"] == "FAST"


def test_fmp_news_requires_both_endpoints_to_complete() -> None:
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise builder.requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            return self.payload

    class Session:
        def get(self, url, *, params, timeout):
            if url.endswith("/news/stock"):
                return Response(503, [])
            return Response(200, [])

    client = builder.FMPNewsClient("test-key", session=Session(), retries=1)
    with pytest.raises(builder.FocusBuildError, match="news/stock"):
        client.fetch("FAST")


def test_fmp_financial_evidence_requires_statements_and_direct_sec_link(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def get(self, url, *, params, timeout):
            calls.append((url, dict(params)))
            if url.endswith("/sec-filings-search/symbol"):
                return Response(
                    [
                        {
                            "formType": "10-Q",
                            "filingDate": "2026-08-10",
                            "finalLink": "https://www.sec.gov/Archives/edgar/data/1/fast10q.htm",
                            "cik": "0000000001",
                            "accessionNumber": "0001-26-000001",
                        }
                    ]
                )
            return Response(
                [
                    {
                        "date": f"{year}-12-31",
                        "acceptedDate": f"{year + 1}-02-15T12:00:00Z",
                    }
                    for year in range(2022, 2026)
                ]
            )

    monkeypatch.setattr(
        builder,
        "calculate_ticker_metrics",
        lambda *args, **kwargs: {
            "statement_periods": 4,
            "latest_fcf_positive": True,
            "revenue_growth_change": 0.05,
            "fcf_margin_change": 0.02,
            "share_count_cagr_3y": 0.01,
            "net_debt_to_ebitda": 0.5,
        },
    )
    client = builder.FMPNewsClient("test-key", session=Session(), retries=1)
    evidence = client.fetch_financial_evidence(
        "FAST",
        market_cap=2_000_000_000,
        as_of=dt.date(2026, 8, 25),
        company_name="Fast Company",
        sector="Technology",
        industry="Software",
        research_lane="standard_company",
    )

    assert evidence["statement_periods"] == 4
    assert evidence["source_current"] is True
    assert evidence["sec_source"]["url"].startswith("https://www.sec.gov/Archives/")
    assert {url.rsplit("/", 1)[-1] for url, _ in calls} >= {
        "income-statement",
        "balance-sheet-statement",
        "cash-flow-statement",
        "symbol",
    }


def test_any_candidate_enrichment_failure_aborts_the_run() -> None:
    class Client:
        def fetch(self, ticker):
            if ticker == "BBB":
                raise builder.FocusBuildError("provider unavailable")
            return []

        def fetch_share_structure(self, ticker):
            return {"float_shares": 10_000_000, "outstanding_shares": 20_000_000}

    rows = [
        {"ticker": ticker, "screen_price": 10.0, "research_lane": "standard_company"}
        for ticker in ("AAA", "BBB")
    ]
    with pytest.raises(builder.FocusBuildError, match="enrichment incomplete"):
        builder.enrich_live_candidates(Client(), rows, as_of=dt.date(2026, 8, 26))


def test_production_coverage_floor_rejects_a_truncated_universe() -> None:
    tickers = [f"T{index}" for index in range(builder.MIN_PRODUCTION_UNIVERSE - 1)]
    prices = pd.DataFrame({"ticker": tickers})
    symbols = {ticker: {"ticker": ticker} for ticker in tickers}
    with pytest.raises(builder.FocusBuildError, match="price universe"):
        builder.validate_production_input_coverage(prices, symbols)


def test_pinned_tradingview_manifest_matches_the_cloud_mirror() -> None:
    manifest = builder.load_screen_manifest()
    assert manifest["armed"]["url"] == builder.TRADINGVIEW_ARMED_URL
    assert manifest["live"]["url"] == builder.TRADINGVIEW_LIVE_URL
