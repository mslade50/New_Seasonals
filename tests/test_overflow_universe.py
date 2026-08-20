"""Unit tests for the dynamic overflow universe (loader + screen).

Pure tests — no network, no R2, no Sheets, no live prices. Run with:
    python -m pytest tests/test_overflow_universe.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import overflow_universe as ou
from build_overflow_universe import screen_universe, _atr_pct_series


AS_OF = pd.Timestamp("2025-01-10")


def _make_ticker(ticker, n_bars, price, volume, atr_frac=0.04, last_date=AS_OF):
    """Synthetic long-format OHLCV for one ticker.

    atr_frac controls intrabar range → ATR%. Prices held flat at `price` so
    dollar volume = price*volume and ATR% ≈ atr_frac*100.
    """
    dates = pd.bdate_range(end=last_date, periods=n_bars)
    close = np.full(n_bars, float(price))
    high = close * (1 + atr_frac / 2)
    low = close * (1 - atr_frac / 2)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n_bars, float(volume)),
        }
    )


def _universe():
    frames = [
        _make_ticker("GOODBIG", 300, 50.0, 250_000),     # $12.5MM ADDV, ~4% ATR → pass
        _make_ticker("THINLOW", 300, 50.0, 10_000),      # $0.5MM ADDV → fail liquidity
        _make_ticker("DEADCALM", 300, 50.0, 250_000, atr_frac=0.0),  # ATR%≈0 → fail vol
        _make_ticker("SHORTHIST", 100, 50.0, 250_000),   # <252 bars → fail history
        _make_ticker("PENNY", 300, 1.0, 9_000_000),      # price<3 → fail price guard
        _make_ticker("STALE", 300, 50.0, 250_000,        # last bar months old → fail freshness
                     last_date=AS_OF - pd.Timedelta(days=120)),
        _make_ticker("AAPL", 300, 200.0, 5_000_000),     # in liquid set → excluded
    ]
    return pd.concat(frames, ignore_index=True)


def test_screen_selects_only_liquid_volatile_fresh():
    prices = _universe()
    liquid = {"AAPL"}
    out = screen_universe(prices, liquid, as_of=AS_OF)
    passed = set(out["ticker"])
    assert passed == {"GOODBIG"}, f"unexpected pass set: {passed}"


def test_screen_addv_and_atr_values():
    prices = _universe()
    out = screen_universe(prices, set(), as_of=AS_OF)
    row = out[out["ticker"] == "GOODBIG"].iloc[0]
    # 50 * 250k = 12.5MM
    assert row["addv_63d"] == pytest.approx(12_500_000, rel=1e-6)
    # ATR% ≈ 4% (range = 0.04*close, TR≈0.04*close, ATR%≈4)
    assert row["atr_pct_63d"] == pytest.approx(4.0, rel=0.05)
    assert row["last_close"] == pytest.approx(50.0)
    assert row["n_bars"] == 300


def test_atr_pct_series_matches_indicator_formula():
    df = _make_ticker("X", 50, 100.0, 1_000, atr_frac=0.02)
    s = _atr_pct_series(df.set_index("date"), window=14)
    # flat price, 2% range → ATR% ≈ 2
    assert s.dropna().iloc[-1] == pytest.approx(2.0, rel=0.05)


def test_screen_excludes_liquid_members():
    prices = _universe()
    out = screen_universe(prices, {"GOODBIG"}, as_of=AS_OF)
    assert "GOODBIG" not in set(out["ticker"])


def test_load_overflow_universe_fallback_when_missing(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    fb = ["FOO", "BAR"]
    assert ou.load_overflow_universe(fallback=fb, path=missing) == fb  # verbatim (caller pre-sorts)
    assert ou.load_overflow_universe(fallback=None, path=missing) == []


def test_load_overflow_universe_gate_off_ignores_parquet(tmp_path, monkeypatch):
    # Activation gate OFF (default): the parquet is ignored and the caller's
    # static fallback comes back verbatim — legacy live behavior until
    # OVERFLOW_UNIVERSE_ACTIVE is enabled.
    monkeypatch.delenv(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, raising=False)
    p = str(tmp_path / "u.parquet")
    pd.DataFrame({"ticker": ["brk.b", "aaa", "AAA"], "addv_63d": [9e6, 5e6, 5e6]}).to_parquet(p)
    assert ou.load_overflow_universe(fallback=["ZZZ"], path=p) == ["ZZZ"]
    # tooling path reads regardless of the gate
    assert ou.load_overflow_universe(fallback=["ZZZ"], path=p, respect_active=False) == ["AAA", "BRK-B"]


def test_load_overflow_universe_reads_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, "1")
    p = str(tmp_path / "u.parquet")
    pd.DataFrame({"ticker": ["brk.b", "aaa", "AAA"], "addv_63d": [9e6, 5e6, 5e6]}).to_parquet(p)
    got = ou.load_overflow_universe(fallback=["ZZZ"], path=p)
    assert got == ["AAA", "BRK-B"]  # normalized, deduped, sorted; fallback ignored


def test_load_overflow_meta(tmp_path, monkeypatch):
    monkeypatch.setenv(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, "1")
    p = str(tmp_path / "u.parquet")
    pd.DataFrame(
        {"ticker": ["aaa"], "addv_63d": [5e6], "atr_pct_63d": [3.0], "last_close": [12.0]}
    ).to_parquet(p)
    meta = ou.load_overflow_meta(path=p)
    assert meta["AAA"]["addv_63d"] == pytest.approx(5e6)
    assert ou.load_overflow_meta(path=str(tmp_path / "missing.parquet")) == {}
    # gate OFF → metadata gates are no-ops even with a parquet present
    monkeypatch.delenv(ou.OVERFLOW_UNIVERSE_ACTIVE_ENV, raising=False)
    assert ou.load_overflow_meta(path=p) == {}


def test_filter_by_addv_per_strategy():
    meta = {
        "BIG": {"addv_63d": 12_000_000},
        "MID": {"addv_63d": 6_000_000},
        "SMALL": {"addv_63d": 3_500_000},
    }
    tickers = ["BIG", "MID", "SMALL"]
    # OVS floor 10MM → only BIG
    assert ou.filter_by_addv(tickers, "Overbot Vol Spike", meta) == ["BIG"]
    # 52wh floor 5MM → BIG, MID
    assert ou.filter_by_addv(tickers, "52wh Breakout", meta) == ["BIG", "MID"]
    # base floor 3MM → all
    assert ou.filter_by_addv(tickers, "Oversold Low Volume", meta) == ["BIG", "MID", "SMALL"]


def test_filter_by_addv_noop_when_no_meta():
    tickers = ["A", "B", "C"]
    assert ou.filter_by_addv(tickers, "Overbot Vol Spike", {}) == tickers


def test_filter_by_addv_keeps_names_missing_from_meta():
    meta = {"BIG": {"addv_63d": 12_000_000}}
    assert ou.filter_by_addv(["BIG", "UNKNOWN"], "Overbot Vol Spike", meta) == ["BIG", "UNKNOWN"]


def test_adv_share_cap_math():
    # 2% of $10MM = $200k; at $50 → 4000 shares
    assert ou.adv_share_cap(10_000_000, 50.0, 0.02) == 4000
    assert ou.adv_share_cap(None, 50.0) is None
    assert ou.adv_share_cap(10_000_000, 0.0) is None
    assert ou.adv_share_cap(float("nan"), 50.0) is None


def test_min_addv_for_defaults():
    assert ou.min_addv_for("Overbot Vol Spike") == 10_000_000
    assert ou.min_addv_for("Unknown Strategy") == ou.MIN_ADDV_BASE


@pytest.mark.parametrize(
    "ticker",
    ["DX-Y.NYB", "DX-Y-NYB", "EURUSD=X", "ES=F", "BTC-USD", "^GSPC"],
)
def test_non_equity_research_symbols_never_enter_overflow(ticker):
    assert not ou._is_tradeable_equity(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "BRK-B", "BF-B"])
def test_equity_symbols_remain_overflow_compatible(ticker):
    assert ou._is_tradeable_equity(ticker)


def test_configured_overflow_universe_excludes_macro_research_symbols():
    from strategy_config import CSV_UNIVERSE

    invalid = [
        ticker for ticker in CSV_UNIVERSE
        if "=" in str(ticker)
        or str(ticker).endswith("-USD")
        or str(ticker).endswith(".NYB")
    ]
    assert invalid == []


def test_point_in_time_membership_matches_asof_screen():
    long = _make_ticker("PIT", 320, 50.0, 120_000).set_index("date")
    metrics = ou.point_in_time_overflow_metrics(long)
    eligibility = ou.point_in_time_eligibility(
        metrics, "Oversold Low Volume", precomputed=True
    )

    for position in (251, 275, 319):
        as_of = long.index[position]
        one_name = long.iloc[: position + 1].reset_index().assign(ticker="PIT")
        screened = screen_universe(
            one_name,
            set(),
            as_of=as_of,
            freshness_td=ou.FRESHNESS_TD,
        )
        assert bool(eligibility.loc[as_of]) == ("PIT" in set(screened.get("ticker", [])))


def test_point_in_time_membership_applies_strategy_addv_floor():
    frame = _make_ticker("MID", 300, 50.0, 120_000).set_index("date")
    metrics = ou.point_in_time_overflow_metrics(frame)
    # $6MM ADDV: eligible for OLV's $3MM floor, ineligible for OVS's $10MM.
    assert ou.point_in_time_eligibility(
        metrics, "Oversold Low Volume", precomputed=True
    ).iloc[-1]
    assert not ou.point_in_time_eligibility(
        metrics, "Overbot Vol Spike", precomputed=True
    ).iloc[-1]


def test_point_in_time_membership_cannot_see_future_rows():
    frame = _make_ticker("PIT", 320, 50.0, 120_000).set_index("date")
    cutoff = frame.index[279]
    before = ou.point_in_time_eligibility(frame, "Oversold Low Volume")
    mutated = frame.copy()
    mutated.loc[mutated.index > cutoff, "Close"] *= 100.0
    mutated.loc[mutated.index > cutoff, "High"] *= 100.0
    mutated.loc[mutated.index > cutoff, "Low"] *= 100.0
    after = ou.point_in_time_eligibility(mutated, "Oversold Low Volume")
    pd.testing.assert_series_equal(before.loc[:cutoff], after.loc[:cutoff])
