"""Forward-only retirement and dollar-index spelling regressions; no live jobs."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from live_scan_universe import canonical_ticker, exclude_retired_symbols, exclude_retired_tickers

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("given, expected", [
    ("DX-Y.NYB", "DX-Y.NYB"), (" dx-y-nyb ", "DX-Y.NYB"),
    ("BRK.B", "BRK-B"), ("BRK-B", "BRK-B"),
    ("EURUSD=X", "EURUSD=X"), ("^GSPC", "^GSPC"), ("DX=F", "DX=F")])
def test_provider_spelling_preserves_instrument(given, expected):
    assert canonical_ticker(given) == expected


def test_leg_official_dates_and_historical_universe_preserved():
    book = [{"universe_tickers": ["LEG", "SPY", "DX-Y-NYB"]}]
    original = copy.deepcopy(book)
    historical, removed = exclude_retired_symbols(book, asof="2026-08-26")
    assert historical[0]["universe_tickers"] == ["LEG", "SPY", "DX-Y.NYB"]
    assert removed == []
    active, removed = exclude_retired_symbols(book, asof="2026-08-27")
    assert active[0]["universe_tickers"] == ["SPY", "DX-Y.NYB"]
    assert removed == ["LEG"] and book == original
    row = next(r for r in json.loads((ROOT / "config/live_scan_exclusions.json").read_text())["exclusions"] if r["ticker"] == "LEG")
    assert row["listing_removed_on"] == "2026-09-08"
    assert row["trading_suspended_on"] == "2026-08-27"
    assert row["evidence_url"].endswith("000087666126000712/ruleprovisionnotice.htm")


def scan_function(name, namespace):
    # Execute only the actual pure loader/downloader body, never the scanner,
    # its imports, broker, sheet, email or workflow entry points.
    tree = ast.parse((ROOT / "daily_scan.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "daily_scan.py", "exec"), namespace)
    return namespace[name]


def test_scanner_reads_dollar_index_from_cache_using_either_spelling(tmp_path):
    path = tmp_path / "prices.parquet"
    frame = pd.DataFrame({"ticker": ["DX-Y.NYB", "BRK-B"], "date": pd.to_datetime(["2026-10-02"] * 2),
                          "Open": [101.0, 400.0], "High": [102.0, 401.0], "Low": [100.0, 399.0],
                          "Close": [101.93, 400.0], "Volume": [0, 100]})
    frame.to_parquet(path, index=False)
    load = scan_function("load_master_prices_dict", dict(pd=pd, os=os, MASTER_PRICES_PATH=str(path), canonical_ticker=canonical_ticker))
    got = load(["DX-Y-NYB", "DX-Y.NYB", "BRK.B"])
    assert set(got) == {"DX-Y.NYB", "BRK-B"}
    assert got["DX-Y.NYB"].iloc[-1]["Close"] == 101.93


def test_scanner_fallback_requests_the_same_index_not_a_proxy():
    requested = []
    def download(tickers, **kwargs):
        requested.extend(tickers)
        return pd.DataFrame({"Close": [101.93]}, index=pd.to_datetime(["2026-10-02"]))
    fn = scan_function("download_historical_data", dict(pd=pd, yf=SimpleNamespace(download=download),
                       time=SimpleNamespace(sleep=lambda _: None), canonical_ticker=canonical_ticker))
    assert set(fn(["DX-Y-NYB", "DX-Y.NYB"])) == {"DX-Y.NYB"}
    assert requested == ["DX-Y.NYB"]


def test_pitch_tape_excludes_retired_leg_but_retains_input_history(monkeypatch):
    from scripts import build_pitch_state as bps
    monkeypatch.setattr(bps, "HEADLINE_TICKERS", ["SPY", "LEG", "DX-Y.NYB"])
    monkeypatch.setattr(bps, "LIQUID_PLUS_COMMODITIES", [])
    monkeypatch.setattr(bps, "_metrics_for", lambda frame: {"date": str(frame["date"].max().date()), "rank_5d": 50,
        "rank_21d": 50, "z10": 0, "dist_52w_high_pct": 0, "dist_sma200_pct": 1})
    prices = pd.DataFrame({"ticker": ["SPY", "LEG", "DX-Y.NYB"],
                           "date": pd.to_datetime(["2026-10-02", "2026-08-27", "2026-10-02"])})
    original = prices.copy(deep=True)
    warnings = []
    tape = bps.build_tape(prices, pd.Timestamp("2026-10-05"), warnings)
    assert set(tape["universe"]) == {"SPY", "DX-Y.NYB"}
    assert not any("stale" in w or "no metrics" in w for w in warnings)
    assert any("LEG" in w and "history retained" in w for w in warnings)
    pd.testing.assert_frame_equal(prices, original)


def test_seasonal_ranks_drop_retired_leg_before_ranking(monkeypatch):
    from scripts import build_seasonal_state as bss
    cs = pd.DataFrame([{"ticker": t, "Date": pd.Timestamp("2026-10-05"),
                        **{f"atr_sznl_{h}d": 50.0 for h in bss.ALL_HORIZONS}}
                       for t in ["SPY", "LEG"]]).set_index("ticker")
    seen = []
    monkeypatch.setattr(bss.se, "seasonal_cross_section", lambda *a: cs.copy())
    monkeypatch.setattr(bss, "classify_ranks", lambda vec: seen.append(vec) or None)
    prices = pd.DataFrame(columns=["ticker", "date", "Close", "Volume"])
    prices["date"] = pd.to_datetime(prices["date"])
    warnings = []
    bss.build_ranks(prices, pd.Timestamp("2026-10-05"), warnings, ranks=pd.DataFrame(), sectors={})
    assert len(seen) == 1
    assert any("LEG" in w and "history retained" in w for w in warnings)


def test_research_boards_exclude_leg_without_dropping_share_classes(tmp_path, monkeypatch):
    from scripts import build_pitch_state as bps
    from scripts import build_seasonal_state as bss
    import daily_seasonal_ideas as dsi
    candidates = [{"ticker": t, "evidence": {"TICKET": True}} for t in ["LEG", "SPY", "BRK.B"]]
    monkeypatch.setattr(dsi, "build", lambda *a, **k: ("", {"candidates": candidates}))
    board = bss.build_board(pd.Timestamp("2026-10-02"), [])
    assert [row["ticker"] for row in board["rows"]] == ["SPY", "BRK.B"]
    assert board["n_candidates"] == 2
    (tmp_path / "data").mkdir()
    (tmp_path / "data/daily_seasonal_ideas.json").write_text(json.dumps({"candidates": candidates}))
    monkeypatch.setattr(bps, "ROOT", tmp_path)
    out = bps.build_seasonality(pd.Timestamp("2026-10-05"), [])
    assert [row["ticker"] for row in out["board_candidates"]] == ["SPY", "BRK.B"]


def test_overflow_and_seasonal_lookup_use_canonical_dollar_index():
    import overflow_universe
    from scripts import seasonal_edge
    assert overflow_universe._norm("DX-Y-NYB") == "DX-Y.NYB"
    assert seasonal_edge._norm_ticker("DX-Y.NYB") == "DX-Y.NYB"
