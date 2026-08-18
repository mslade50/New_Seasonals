from pathlib import Path

import pandas as pd
import pytest

import abs_return_dispersion
from scripts.build_risk_json import load_risk_data_from_master


def _row(ticker: str, date: str, close: float) -> dict:
    return {
        "ticker": ticker,
        "date": pd.Timestamp(date),
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": 1_000_000.0,
    }


def test_risk_inputs_use_authoritative_master_asof(tmp_path: Path, monkeypatch):
    sp500 = [f"T{i:03d}" for i in range(55)]
    monkeypatch.setattr(abs_return_dispersion, "SP500_TICKERS", sp500)
    rows = []
    for ticker in ["SPY", "^VIX", "^VIX3M", *sp500]:
        rows.extend([
            _row(ticker, "2015-08-17", 90.0),
            _row(ticker, "2016-08-18", 100.0),
            _row(ticker, "2026-08-17", 120.0),
        ])
    master = tmp_path / "master_prices.parquet"
    pd.DataFrame(rows).to_parquet(master, index=False)

    spy, closes, sp500_closes = load_risk_data_from_master(master)

    assert spy.index.max() == pd.Timestamp("2026-08-17")
    assert spy.index.min() == pd.Timestamp("2016-08-18")
    assert closes["SPY"].dropna().index.max() == pd.Timestamp("2026-08-17")
    assert sp500_closes.shape[1] == 55


def test_risk_inputs_fail_closed_without_master(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="authoritative master prices"):
        load_risk_data_from_master(tmp_path / "missing.parquet")
