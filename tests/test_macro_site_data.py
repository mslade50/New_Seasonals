import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_universe import (
    IBKR_EQUIVALENTS,
    SECTOR_ETFS,
    TICKER_INFO,
    get_ibkr_label,
    get_ticker_label,
)
from scripts.macro_site_data import (
    export_macro_snapshot,
    extension_ranks,
    percentile_rank,
    sort_key,
    rank_session,
    validate_macro_rank_coverage,
)


def test_percentile_rank_matches_page_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    assert percentile_rank(s, 2.0) == pytest.approx(50.0)   # (s <= v) / size * 100
    assert percentile_rank(s, 4.0) == pytest.approx(100.0)
    assert percentile_rank(s, 0.5) == pytest.approx(0.0)
    assert percentile_rank(pd.Series(dtype=float), 1.0) is None


def test_ibkr_equivalents_cover_macro_universe_and_format_titles():
    assert set(IBKR_EQUIVALENTS) == set(SECTOR_ETFS)

    assert get_ibkr_label("^GSPC") == "ES FUT"
    assert get_ticker_label("^GSPC").endswith("IBKR: ES FUT")
    assert get_ticker_label("^GDAXI").endswith("IBKR: DAX (FDAX) FUT")
    assert get_ticker_label("^VIX").endswith("IBKR: VIX (VX) FUT")
    assert get_ticker_label("EURUSD=X").endswith("IBKR: EUR.USD FX")
    assert get_ticker_label("GLD").endswith("IBKR: GC FUT")

    # Directly tradeable names do not repeat themselves in the title.
    assert "IBKR:" not in get_ticker_label("CEF")
    assert "IBKR:" not in get_ticker_label("TLT")

    # Index/commodity research series prefer futures unless explicitly proxied.
    index_or_commodity = {
        ticker for ticker in SECTOR_ETFS
        if ticker.startswith("^") or ticker.endswith("=F")
        or ticker in {"GLD", "SLV", "UNG"}
    }
    allowed_proxy_etfs = {"^IXIC", "^DJT", "^SOX", "^BSESN", "^MXX"}
    for ticker in index_or_commodity - allowed_proxy_etfs:
        assert IBKR_EQUIVALENTS[ticker].sec_type == "FUT"
    for ticker in allowed_proxy_etfs:
        equivalent = IBKR_EQUIVALENTS[ticker]
        assert equivalent.sec_type == "STK" and equivalent.proxy


def test_extension_ranks_against_pandas():
    close = pd.Series(np.linspace(100, 130, 300),
                      index=pd.bdate_range("2024-01-02", periods=300))
    out = extension_ranks(close)
    assert out["price"] == pytest.approx(130.0)
    for window in (5, 20, 50, 200):
        ma = close.rolling(window).mean()
        dist = (close - ma) / ma * 100.0
        expected = float((dist.dropna() <= dist.dropna().iloc[-1]).sum()
                         / dist.dropna().size * 100.0)
        assert out[f"r{window}"] == pytest.approx(expected)


def _fixture_prices(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2023-01-02", periods=520)
    rows = []
    for ticker, base in [("GLD", 150.0), ("^VIX", 15.0)]:
        for i, date in enumerate(dates):
            close = base + i * 0.1
            rows.append({"ticker": ticker, "date": date, "Close": close,
                         "High": close + 1, "Low": close - 1})
    path = tmp_path / "master_prices.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _fixture_ranks(tmp_path: Path) -> Path:
    rows = []
    # GLD: strong seasonal (85 on the 5d window), history + forward dates.
    # ^VIX: mild (55). AGG: rank-only ticker (no prices in the fixture).
    for ticker, rank in [("GLD", 85.0), ("^VIX", 55.0), ("AGG", 20.0)]:
        for date in ["2026-07-10", "2026-07-13", "2026-09-01"]:
            row = {"ticker": ticker, "Date": pd.Timestamp(date)}
            for w in (5, 10, 21, 63, 126, 252):
                row[f"atr_sznl_{w}d"] = rank if date != "2026-09-01" else 99.0
            rows.append(row)
    path = tmp_path / "atr_seasonal_ranks.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_export_macro_snapshot(tmp_path: Path):
    prices = _fixture_prices(tmp_path)
    ranks = _fixture_ranks(tmp_path)
    out = tmp_path / "macro.json"
    before = hashlib.sha256(prices.read_bytes()).hexdigest()

    payload = export_macro_snapshot(prices, ranks, out, asof=pd.Timestamp("2026-07-14"))

    assert hashlib.sha256(prices.read_bytes()).hexdigest() == before
    assert payload["sznl_available"] is True
    last_fixture_day = pd.bdate_range("2023-01-02", periods=520)[-1]
    assert payload["asof"] == last_fixture_day.strftime("%Y-%m-%d")
    # caret tickers without price history are dropped; everything else stays
    expected = {t for t in set(SECTOR_ETFS) if not t.startswith("^")} | {"^VIX"}
    assert {row["ticker"] for row in payload["rows"]} == expected

    by_ticker = {row["ticker"]: row for row in payload["rows"]}
    assert "^GSPC" not in by_ticker  # caret, no prices in fixture -> dropped
    assert "^VIX" in by_ticker       # caret with prices survives
    gld = by_ticker["GLD"]
    assert gld["name"] == TICKER_INFO["GLD"][0]
    assert gld["price"] == pytest.approx(150.0 + 519 * 0.1)
    assert gld["file"] and gld["file"].endswith(".bin")
    # asof lookup takes the last row <= asof (2026-07-13), never the forward 99s
    assert gld["s5"] == pytest.approx(85.0)
    assert gld["ibkr"] == "GC FUT"
    assert gld["chart_label"].endswith("IBKR: GC FUT")
    assert by_ticker["AGG"]["s5"] == pytest.approx(20.0)
    # AGG has ranks but no prices: table-only row
    agg = by_ticker["AGG"]
    assert agg["price"] is None and agg["r200"] is None and "file" not in agg

    # sort: GLD (|85-50|=35) before ^VIX (5) before AGG (30)... AGG=|20-50|=30
    order = [row["ticker"] for row in payload["rows"] if row["ticker"] in ("GLD", "^VIX", "AGG")]
    assert order == ["GLD", "AGG", "^VIX"]
    # no-rank tickers sort to the bottom
    assert sort_key(by_ticker["TLT"]) == -1.0

    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["rows"][0]["ticker"] == "GLD"


def test_export_without_ranks_flags_it(tmp_path: Path):
    prices = _fixture_prices(tmp_path)
    out = tmp_path / "macro.json"
    payload = export_macro_snapshot(prices, tmp_path / "missing.parquet", out,
                                    asof=pd.Timestamp("2026-07-14"))
    assert payload["sznl_available"] is False
    by_ticker = {row["ticker"]: row for row in payload["rows"]}
    assert by_ticker["GLD"]["s5"] is None
    assert by_ticker["GLD"]["price"] is not None


def _macro_coverage_inputs(tmp_path, monkeypatch):
    import scripts.macro_site_data as macro

    monkeypatch.setattr(macro, "SECTOR_ETFS", ["GLD", "^VIX"])
    dates = pd.bdate_range("2010-01-04", "2026-09-22")
    steps = np.arange(len(dates))
    close = 30 + steps * 0.004 + 3 * np.sin(steps / 17) + np.cos(steps / 71)
    frames = []
    for ticker in ("GLD", "^VIX"):
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "Close": close,
            "High": close + 0.7, "Low": close - 0.6,
        }))
    prices = tmp_path / "coverage_prices.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(prices, index=False)
    ranks = tmp_path / "strategy_ranks.parquet"
    pd.DataFrame([{
        "ticker": "GLD", "Date": pd.Timestamp("2026-09-22"),
        **{f"atr_sznl_{w}d": 87.3 for w in (5, 10, 21, 63, 126, 252)},
    }]).to_parquet(ranks, index=False)
    return prices, ranks


def test_missing_macro_ranks_match_canonical_math_without_changing_inputs(tmp_path, monkeypatch):
    from build_atr_seasonal_ranks import (
        compute_ranks_for_year, generate_trading_dates, prepare_ticker_data,
    )

    prices, ranks = _macro_coverage_inputs(tmp_path, monkeypatch)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (prices, ranks)}
    payload = export_macro_snapshot(prices, ranks, tmp_path / "complete.json",
                                    asof=pd.Timestamp("2026-09-22"))
    rows = {row["ticker"]: row for row in payload["rows"]}
    frame = pd.read_parquet(prices).query("ticker == '^VIX'").set_index("date")
    annual = compute_ranks_for_year(prepare_ticker_data(frame), 2026)
    calendar = generate_trading_dates(2026)
    day = calendar.loc[calendar["Date"] == pd.Timestamp("2026-09-22"), "day_count"].iloc[0]
    for w in (5, 10, 21, 63, 126, 252):
        assert rows["^VIX"][f"s{w}"] == pytest.approx(annual.loc[day, f"atr_sznl_{w}d"].round(1))
        assert rows["GLD"][f"s{w}"] == 87.3
    assert rows["^VIX"]["sznl_asof"] == "2026-09-22"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before


def test_macro_supplement_ignores_current_year_outcomes(tmp_path, monkeypatch):
    prices, ranks = _macro_coverage_inputs(tmp_path, monkeypatch)
    before = export_macro_snapshot(prices, ranks, tmp_path / "before.json",
                                   asof=pd.Timestamp("2026-09-22"))
    frame = pd.read_parquet(prices)
    frame.loc[frame["date"].dt.year == 2026, ["High", "Low", "Close"]] *= 4
    changed = tmp_path / "changed_prices.parquet"
    frame.to_parquet(changed, index=False)
    after = export_macro_snapshot(changed, ranks, tmp_path / "after.json",
                                  asof=pd.Timestamp("2026-09-22"))
    left = next(row for row in before["rows"] if row["ticker"] == "^VIX")
    right = next(row for row in after["rows"] if row["ticker"] == "^VIX")
    assert left["s5"] is not None
    assert [left[f"s{w}"] for w in (5, 10, 21, 63, 126, 252)] == [
        right[f"s{w}"] for w in (5, 10, 21, 63, 126, 252)
    ]


@pytest.mark.parametrize("asof,expected", [
    ("2026-01-01", "2025-12-31"), ("2026-09-20", "2026-09-18"),
])
def test_rank_session_handles_holidays_and_year_rollover(asof, expected):
    assert rank_session(pd.Timestamp(asof))["Date"] == pd.Timestamp(expected)


def test_insufficient_macro_history_blocks_export(tmp_path, monkeypatch):
    prices, ranks = _macro_coverage_inputs(tmp_path, monkeypatch)
    frame = pd.read_parquet(prices)
    short = tmp_path / "short.parquet"
    frame[frame["date"].dt.year >= 2025].to_parquet(short, index=False)
    with pytest.raises(ValueError, match="three prior calendar years"):
        export_macro_snapshot(short, ranks, tmp_path / "incomplete.json",
                              asof=pd.Timestamp("2026-09-22"))


def test_missing_strategy_ranks_cannot_use_macro_supplement(tmp_path, monkeypatch):
    prices, ranks = _macro_coverage_inputs(tmp_path, monkeypatch)
    absent = tmp_path / "absent_gld.parquet"
    pd.read_parquet(ranks).assign(ticker="SPY").to_parquet(absent, index=False)
    payload = export_macro_snapshot(prices, absent, tmp_path / "bad_strategy.json",
                                    asof=pd.Timestamp("2026-09-22"))
    with pytest.raises(ValueError, match="incomplete or stale for: GLD"):
        validate_macro_rank_coverage(payload)


@pytest.mark.parametrize("field,value", [
    ("s5", None), ("s10", float("nan")), ("s21", float("inf")),
    ("s63", -1), ("s126", 101), ("s252", True),
    ("sznl_asof", "2026-09-21"),
])
def test_coverage_gate_rejects_invalid_or_stale_ranks(tmp_path, monkeypatch, field, value):
    prices, ranks = _macro_coverage_inputs(tmp_path, monkeypatch)
    payload = export_macro_snapshot(prices, ranks, tmp_path / "valid.json",
                                    asof=pd.Timestamp("2026-09-22"))
    validate_macro_rank_coverage(payload)
    payload["rows"][0][field] = value
    with pytest.raises(ValueError, match="incomplete or stale"):
        validate_macro_rank_coverage(payload)
