import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_universe import SECTOR_ETFS, TICKER_INFO
from scripts.macro_site_data import (
    export_macro_snapshot,
    extension_ranks,
    percentile_rank,
    sort_key,
)


def test_percentile_rank_matches_page_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    assert percentile_rank(s, 2.0) == pytest.approx(50.0)   # (s <= v) / size * 100
    assert percentile_rank(s, 4.0) == pytest.approx(100.0)
    assert percentile_rank(s, 0.5) == pytest.approx(0.0)
    assert percentile_rank(pd.Series(dtype=float), 1.0) is None


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
            rows.append({"ticker": ticker, "date": date, "Close": base + i * 0.1})
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
