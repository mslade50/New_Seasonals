import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fundamental.config import (
    BROAD_UNIVERSE_POLICY,
    SCORE_WEIGHTS,
    SLEEVE_POLICY,
    policy_payload,
)
from fundamental.fmp import normalize_rows
from fundamental.metrics import calculate_ticker_metrics
from fundamental.report import render_candidate_report
from fundamental.sec import normalize_companyfacts
from fundamental.storage import (
    load_latest_snapshot_parts,
    point_in_time_latest,
    snapshot_part_path,
    write_immutable_parquet,
)
from fundamental.tearsheet import build_tearsheet_pack
from fundamental.underwrite import build_underwrite_pack, load_underwrite_decisions
from fundamental.triage import score_candidates
from fundamental.universe import (
    build_broad_universe,
    normalize_screener_rows,
    select_balanced_enrichment_batch,
    summarize_universe,
)
from scripts.build_fundamental_report import _research_eligible_symbols


def test_policy_is_research_only_and_limits_are_coherent():
    assert sum(SCORE_WEIGHTS.values()) == 100
    assert SLEEVE_POLICY.target_nav_pct < SLEEVE_POLICY.hard_nav_cap_pct
    assert SLEEVE_POLICY.hard_nav_cap_pct <= SLEEVE_POLICY.combined_slow_sleeves_hard_cap_pct
    assert SLEEVE_POLICY.core_max_pct <= SLEEVE_POLICY.single_name_hard_cap_pct
    assert SLEEVE_POLICY.live_actions_enabled is False
    assert BROAD_UNIVERSE_POLICY.min_market_cap < 2_000_000_000
    assert BROAD_UNIVERSE_POLICY.default_enrichment_batch <= BROAD_UNIVERSE_POLICY.max_enrichment_batch
    assert policy_payload()["policy_version"] == "fundamental-sleeve.v2.1"


def test_daily_report_excludes_companies_that_left_current_universe():
    symbols = pd.DataFrame({
        "ticker": ["KEEP", "DROP", "UNKNOWN"],
        "research_eligible": [True, False, None],
    })
    kept = _research_eligible_symbols(symbols)
    assert kept["ticker"].tolist() == ["KEEP"]


def test_point_in_time_selector_fails_closed_on_unaccepted_rows():
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "BBB"],
        "endpoint": ["income-statement"] * 4,
        "date": ["2024-12-31", "2024-12-31", "2025-12-31", "2024-12-31"],
        "accepted_at": ["2025-02-01T12:00:00Z", "2025-02-15T12:00:00Z", None, "2025-03-01T12:00:00Z"],
        "revenue": [100, 101, 150, 80],
    })
    selected = point_in_time_latest(frame, "2025-02-20T00:00:00Z")
    assert len(selected) == 1
    assert selected.iloc[0]["revenue"] == 101
    assert selected.iloc[0]["ticker"] == "AAA"


def _synthetic_bundle() -> pd.DataFrame:
    dates = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"])
    accepted = pd.to_datetime(["2022-02-01", "2023-02-01", "2024-02-01", "2025-02-01", "2026-02-01"], utc=True)
    common = {"ticker": "AAA", "date": dates, "accepted_at": accepted}
    income = pd.DataFrame({
        **common, "endpoint": "income-statement",
        "revenue": [100, 112, 126, 143, 165],
        "grossProfit": [60, 68, 78, 90, 106],
        "operatingIncome": [20, 23, 27, 33, 40],
        "netIncome": [15, 17, 20, 25, 31],
        "incomeBeforeTax": [19, 21, 25, 31, 38],
        "incomeTaxExpense": [4, 4, 5, 6, 7],
        "ebitda": [24, 28, 32, 39, 47],
        "weightedAverageShsOutDil": [10.0, 9.9, 9.8, 9.7, 9.6],
    })
    balance = pd.DataFrame({
        **common, "endpoint": "balance-sheet-statement",
        "totalAssets": [120, 128, 138, 150, 164],
        "totalDebt": [20, 19, 18, 17, 16],
        "cashAndShortTermInvestments": [12, 14, 17, 21, 26],
        "totalStockholdersEquity": [60, 66, 73, 82, 93],
    })
    cash = pd.DataFrame({
        **common, "endpoint": "cash-flow-statement",
        "operatingCashFlow": [19, 22, 26, 32, 39],
        "capitalExpenditure": [-5, -5, -6, -7, -8],
        "freeCashFlow": [14, 17, 20, 25, 31],
        "stockBasedCompensation": [2.0, 2.0, 2.1, 2.2, 2.3],
        "commonStockRepurchased": [-1, -2, -2, -3, -4],
        "commonStockIssued": [0, 0, 0, 0, 0],
    })
    return pd.concat([income, balance, cash], ignore_index=True, sort=False)


def test_metrics_measure_per_share_cash_quality_and_incremental_returns():
    result = calculate_ticker_metrics(
        _synthetic_bundle(), ticker="AAA", market_cap=250.0,
        company_name="Alpha", sector="Technology", sec_rows=50,
    )
    assert result["statement_periods"] == 5
    assert result["roic"] > 0
    assert result["incremental_roic"] > 0
    assert result["revenue_cagr_3y"] > 0.10
    assert result["fcf_margin"] > 0.15
    assert result["share_count_cagr_3y"] < 0
    assert result["fcf_positive_years"] == 5
    assert result["sec_rows"] == 50


def test_negative_invested_capital_does_not_publish_misleading_roic():
    bundle = _synthetic_bundle()
    balance = bundle["endpoint"].eq("balance-sheet-statement")
    bundle.loc[balance, "totalDebt"] = 0
    bundle.loc[balance, "totalStockholdersEquity"] = -100
    bundle.loc[balance, "cashAndShortTermInvestments"] = 20
    result = calculate_ticker_metrics(bundle, ticker="AAA", market_cap=250.0)
    assert pd.isna(result["roic"])


def _metric_universe() -> pd.DataFrame:
    rows = []
    for i in range(6):
        strength = 5 - i
        rows.append({
            "ticker": f"T{i}", "company_name": f"Company {i}", "sector": "Industrials",
            "industry": "Machinery", "market_cap": 10_000_000_000 + strength,
            "statement_periods": 5, "sec_rows": 0,
            "roic": 0.10 + strength * 0.03,
            "incremental_roic": 0.08 + strength * 0.025,
            "gross_profitability": 0.25 + strength * 0.04,
            "gross_margin_stability": 0.06 - strength * 0.008,
            "revenue_cagr_3y": 0.03 + strength * 0.025,
            "fcf_margin": 0.05 + strength * 0.025,
            "cash_conversion": 0.8 + strength * 0.08,
            "accrual_ratio": 0.08 - strength * 0.012,
            "sbc_to_revenue": 0.06 - strength * 0.008,
            "share_count_cagr_3y": 0.04 - strength * 0.01,
            "net_debt_to_ebitda": 3.0 - strength * 0.45,
            "fcf_positive_years": float(strength),
            "fcf_yield": 0.02 + strength * 0.012,
            "earnings_yield": 0.015 + strength * 0.01,
            "revenue_growth_change": -0.02 + strength * 0.01,
            "fcf_margin_change": -0.01 + strength * 0.006,
            "latest_fcf_positive": i < 5,
        })
    return pd.DataFrame(rows)


def _green_trends() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(6)],
        "price": [100.0] * 6,
        "sma200": [90.0] * 6,
        "above_sma200": [True] * 6,
        "sma200_slope_20d": [0.02] * 6,
        "return_12_1": [0.15] * 6,
        "relative_return_12_1": [0.05] * 6,
        "dollar_volume_63d": [50_000_000.0] * 6,
        "price_history_days": [500] * 6,
    })


def test_triage_advances_research_but_never_marks_a_trade_ready():
    candidates = score_candidates(_metric_universe(), _green_trends(), as_of="2026-08-04")
    top = candidates.iloc[0]
    assert top["ticker"] == "T0"
    assert top["research_priority"] == "A - immediate research candidate"
    assert top["research_score"] >= 80
    assert "not approved for capital" in top["actionability"].lower()
    assert candidates["implementation_readiness"].str.startswith("Not implementation-ready").all()
    assert candidates["variant_wedge"].str.startswith("UNTESTED").all()
    assert candidates["screen_can_surface_review"].eq(False).all()


def test_specialist_sector_is_covered_without_a_misleading_general_score():
    metrics = _metric_universe()
    metrics.loc[0, "sector"] = "Financial Services"
    candidates = score_candidates(metrics, _green_trends(), as_of="2026-08-04")
    row = candidates[candidates["ticker"] == "T0"].iloc[0]
    assert row["research_priority"] == "C - screen flag only"
    assert row["research_lane"] == "financials_specialist"
    assert pd.isna(row["research_score"])
    assert row["hard_exclusion_reason"] == ""
    assert "Baseline is current" in row["first_rejection"]
    assert "capital and credit" in row["next_workflow"]


def test_duplicate_issuer_defaults_to_more_liquid_share_class():
    metrics = _metric_universe()
    metrics.loc[0, "issuer_cik"] = "0001"
    metrics.loc[1, "issuer_cik"] = "0001"
    trends = _green_trends()
    trends.loc[0, "dollar_volume_63d"] = 100_000_000.0
    trends.loc[1, "dollar_volume_63d"] = 30_000_000.0
    candidates = score_candidates(metrics, trends, as_of="2026-08-04")
    lower_liquidity = candidates[candidates["ticker"] == "T1"].iloc[0]
    assert lower_liquidity["research_priority"] == "Reject"
    assert "Duplicate issuer share class" in lower_liquidity["hard_exclusion_reason"]


def test_fmp_normalization_keeps_acceptance_and_source_labels():
    frame = normalize_rows(
        [{"date": "2025-12-31", "acceptedDate": "2026-02-01T18:00:00Z", "revenue": 10}],
        ticker="AAA", endpoint="income-statement", snapshot_as_of="2026-08-04",
        digest="abc", fetched_at="2026-08-04T20:00:00Z",
    )
    assert str(frame.iloc[0]["accepted_at"]) == "2026-02-01 18:00:00+00:00"
    assert frame.iloc[0]["source_label"] == "fact_provider_standardized"


def test_profile_snapshot_drops_intraday_fields_but_keeps_raw_digest():
    frame = normalize_rows(
        [{"symbol": "AAA", "companyName": "Alpha", "cik": "1", "industry": "Software",
          "price": 123.45, "marketCap": 10_000_000_000, "beta": 1.2}],
        ticker="AAA", endpoint="profile", snapshot_as_of="2026-08-04",
        digest="raw123", fetched_at="2026-08-04T20:00:00Z",
    )
    assert frame.iloc[0]["cik"] == "1"
    assert frame.iloc[0]["raw_payload_digest"] == "raw123"
    assert "price" not in frame.columns
    assert "marketCap" not in frame.columns


def test_sec_companyfacts_join_accession_to_acceptance_time():
    submissions = {"filings": {"recent": {
        "accessionNumber": ["0001-26-000001"],
        "acceptanceDateTime": ["2026-02-01T18:00:00.000Z"],
    }}}
    facts = {"facts": {"us-gaap": {"Revenue": {
        "label": "Revenue", "description": "Revenue", "units": {"USD": [{
            "val": 100, "start": "2025-01-01", "end": "2025-12-31", "fy": 2025,
            "fp": "FY", "form": "10-K", "filed": "2026-02-01", "accn": "0001-26-000001"
        }]}
    }}}}
    frame = normalize_companyfacts(
        facts, submissions, ticker="AAA", cik=1, snapshot_as_of="2026-08-04",
        digest="def", fetched_at="2026-08-04T20:00:00Z",
    )
    assert len(frame) == 1
    assert frame.iloc[0]["source_label"] == "fact_source_reported"
    assert str(frame.iloc[0]["accepted_at"]) == "2026-02-01 18:00:00+00:00"


def test_immutable_parquet_rejects_changed_history(tmp_path):
    path = tmp_path / "snapshot.parquet"
    write_immutable_parquet(pd.DataFrame({"value": [1]}), path)
    write_immutable_parquet(pd.DataFrame({"value": [1]}), path)
    with pytest.raises(FileExistsError):
        write_immutable_parquet(pd.DataFrame({"value": [2]}), path)


def test_immutable_snapshot_retry_ignores_only_retrieval_time(tmp_path):
    path = tmp_path / "snapshot.parquet"
    first = pd.DataFrame({"value": [1], "payload_digest": ["abc"], "fetched_at": ["2026-08-04T12:00:00Z"]})
    retry = pd.DataFrame({"value": [1], "payload_digest": ["abc"], "fetched_at": ["2026-08-04T12:05:00Z"]})
    write_immutable_parquet(first, path)
    write_immutable_parquet(retry, path)
    assert pd.read_parquet(path).iloc[0]["fetched_at"] == "2026-08-04T12:00:00Z"


def test_immutable_snapshot_retry_ignores_raw_archive_identity_not_semantics(tmp_path):
    path = tmp_path / "snapshot.parquet"
    first = pd.DataFrame({"value": [1], "payload_digest": ["semantic"], "raw_payload_digest": ["raw1"]})
    retry = pd.DataFrame({"value": [1], "payload_digest": ["semantic"], "raw_payload_digest": ["raw2"]})
    write_immutable_parquet(first, path)
    write_immutable_parquet(retry, path)
    with pytest.raises(FileExistsError):
        write_immutable_parquet(
            pd.DataFrame({"value": [2], "payload_digest": ["changed"], "raw_payload_digest": ["raw3"]}),
            path,
        )


def test_report_has_no_execution_path_or_live_button(tmp_path):
    candidates = score_candidates(_metric_universe(), _green_trends(), as_of="2026-08-04")
    path = render_candidate_report(candidates, {
        "as_of": "2026-08-04", "gaps": ["SEC tie-out pending"], "sources": []
    }, Path(tmp_path) / "report.html")
    text = path.read_text(encoding="utf-8")
    assert "Research only" in text
    assert "Live actions are disabled" in text
    assert "Research engine diagnostics — optional" in text
    assert "The broad universe stays here for audit and debugging" in text
    assert "/exec-command" not in text
    assert "Send live" not in text


def test_screen_rank_alone_never_creates_user_quick_review(tmp_path):
    candidates = score_candidates(_metric_universe(), _green_trends(), as_of="2026-08-04")
    path = render_candidate_report(
        candidates,
        {"as_of": "2026-08-04", "gaps": [], "sources": []},
        Path(tmp_path) / "report.html",
    )
    text = path.read_text(encoding="utf-8")
    assert "Nothing needs your attention today" in text
    assert "<h2>Quick review</h2>" not in text
    assert "QUICK REVIEW" not in text
    assert "HYPOTHESIS TEST" in text
    assert '"quick_review_count": 0' in text.replace("&quot;", '"')


def test_completed_underwrite_controls_reader_facing_decision(tmp_path, v2_underwrite_factory):
    candidates = score_candidates(_metric_universe(), _green_trends(), as_of="2026-08-04")
    ticker = str(candidates.iloc[0]["ticker"])
    decisions = [v2_underwrite_factory(ticker=ticker, as_of="2026-08-04")]
    links = build_underwrite_pack(decisions, candidates, tmp_path / "underwrites")
    path = render_candidate_report(
        candidates,
        {"as_of": "2026-08-04", "gaps": [], "sources": []},
        Path(tmp_path) / "report.html",
        tearsheet_links=links,
        underwrite_decisions=decisions,
    )
    text = path.read_text(encoding="utf-8")
    assert "1 idea worth a quick check" in text
    assert "<h2>Quick review</h2>" in text
    assert "Open the completed underwrite" in text
    assert "A completed underwrite found a defined expectations gap" in text
    assert "No allocation, position, order, or broker action" in (
        tmp_path / "underwrites" / f"{ticker}.html"
    ).read_text(encoding="utf-8")


def test_underwrite_decision_loader_rejects_duplicates(tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"decisions": [
        {"ticker": "AAA", "decision": "PASS"},
        {"ticker": "AAA", "decision": "WAIT_FOR_PROOF"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_underwrite_decisions(path)


def test_advanced_candidate_gets_source_backed_research_packet(tmp_path):
    candidates = score_candidates(_metric_universe(), _green_trends(), as_of="2026-08-04")
    top = candidates.head(1).copy()
    fmp = _synthetic_bundle().copy()
    fmp["ticker"] = top.iloc[0]["ticker"]
    links = build_tearsheet_pack(top, fmp, pd.DataFrame(), tmp_path / "tearsheets")
    ticker = str(top.iloc[0]["ticker"])
    assert links[ticker] == f"tearsheets/{ticker}.html"
    text = (tmp_path / "tearsheets" / f"{ticker}.html").read_text(encoding="utf-8")
    assert "Research baseline only" in text
    assert "variant view" in text
    assert "Material evidence gaps" in text
    assert "No capital action" not in text  # no simulated action control is rendered
    assert "does not recommend a security, allocate capital, or enable an order" in text


def _broad_prices() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-08-04", periods=300)
    frames = []
    for ticker, start, end in (("AAA", 20, 40), ("BBB", 50, 65), ("SPY", 100, 120)):
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "Close": np.linspace(start, end, len(dates)),
            "Volume": 1_000_000,
        }))
    return pd.concat(frames, ignore_index=True)


def test_broad_universe_preserves_standard_and_specialist_lanes():
    payload = [
        {"symbol": "AAA", "companyName": "Alpha", "marketCap": 1_000_000_000,
         "sector": "Industrials", "industry": "Machinery", "price": 40,
         "volume": 1_000_000, "exchangeShortName": "NASDAQ", "country": "US",
         "isEtf": False, "isFund": False, "isActivelyTrading": True},
        {"symbol": "BBB", "companyName": "Beta Bank", "marketCap": 5_000_000_000,
         "sector": "Financial Services", "industry": "Banks - Regional", "price": 65,
         "volume": 1_000_000, "exchangeShortName": "NYSE", "country": "US",
         "isEtf": False, "isFund": False, "isActivelyTrading": True},
        {"symbol": "ETF", "companyName": "Fund", "marketCap": 5_000_000_000,
         "sector": "Financial Services", "industry": "ETF", "price": 20,
         "volume": 1_000_000, "exchangeShortName": "NYSE", "country": "US",
         "isEtf": True, "isFund": True, "isActivelyTrading": True},
    ]
    screener = normalize_screener_rows(payload, as_of="2026-08-04")
    universe = build_broad_universe(screener, _broad_prices(), as_of="2026-08-04")
    alpha = universe[universe["ticker"].eq("AAA")].iloc[0]
    bank = universe[universe["ticker"].eq("BBB")].iloc[0]
    fund = universe[universe["ticker"].eq("ETF")].iloc[0]
    assert bool(alpha["research_eligible"])
    assert alpha["market_cap_band"] == "small"
    assert alpha["research_lane"] == "standard_company"
    assert bool(bank["research_eligible"])
    assert bank["research_lane"] == "financials_specialist"
    assert not bool(fund["research_eligible"])
    summary = summarize_universe(universe)
    assert summary["research_eligible"] == 2
    assert summary["specialist_queue"] == 1


def test_broad_universe_excludes_stale_prices_even_when_provider_marks_active():
    payload = [{
        "symbol": "DEAD", "companyName": "Acquired Company", "marketCap": 5_000_000_000,
        "sector": "Financial Services", "industry": "Banks - Regional", "price": 75,
        "volume": 1_000_000, "exchangeShortName": "NYSE", "country": "US",
        "isEtf": False, "isFund": False, "isActivelyTrading": True,
    }]
    screener = normalize_screener_rows(payload, as_of="2026-08-28")
    dates = pd.bdate_range(end="2026-08-20", periods=300)
    prices = pd.DataFrame({
        "ticker": "DEAD", "date": dates, "Close": np.linspace(50, 75, len(dates)),
        "Volume": 1_000_000,
    })
    universe = build_broad_universe(screener, prices, as_of="2026-08-28")
    row = universe.iloc[0]
    assert not bool(row["research_eligible"])
    assert "stale" in row["eligibility_reason"].lower()


def test_balanced_enrichment_batch_does_not_default_to_mega_caps():
    rows = []
    for size, sector, base in (
        ("small", "Industrials", 10), ("mid", "Technology", 20),
        ("large", "Healthcare", 30), ("mega", "Technology", 40),
    ):
        for i in range(2):
            rows.append({
                "ticker": f"{size[:1].upper()}{base+i}",
                "research_eligible": True,
                "research_lane": "standard_company",
                "market_cap_band": size,
                "sector": sector,
                "dollar_volume_63d": 100_000_000 - i,
            })
    universe = pd.DataFrame(rows)
    selected = select_balanced_enrichment_batch(universe, 4)
    selected_sizes = set(universe.set_index("ticker").loc[selected, "market_cap_band"])
    assert len(selected) == 4
    assert {"small", "mid"}.issubset(selected_sizes)
    assert selected_sizes != {"mega"}


def test_latest_snapshot_loader_combines_incremental_batch_dates(tmp_path, monkeypatch):
    import fundamental.storage as storage

    monkeypatch.setattr(storage, "SNAPSHOT_ROOT", tmp_path)
    old_income = snapshot_part_path("fmp", "2026-08-01", "AAA", "income-statement")
    new_income = snapshot_part_path("fmp", "2026-08-04", "AAA", "income-statement")
    old_balance = snapshot_part_path("fmp", "2026-08-01", "AAA", "balance-sheet-statement")
    write_immutable_parquet(pd.DataFrame({"ticker": ["AAA"], "endpoint": ["income-statement"], "value": [1]}), old_income)
    write_immutable_parquet(pd.DataFrame({"ticker": ["AAA"], "endpoint": ["income-statement"], "value": [2]}), new_income)
    write_immutable_parquet(pd.DataFrame({"ticker": ["AAA"], "endpoint": ["balance-sheet-statement"], "value": [3]}), old_balance)
    loaded = load_latest_snapshot_parts("fmp", "2026-08-04")
    assert len(loaded) == 2
    assert loaded.loc[loaded["endpoint"].eq("income-statement"), "value"].iloc[0] == 2
    assert loaded.loc[loaded["endpoint"].eq("balance-sheet-statement"), "value"].iloc[0] == 3
