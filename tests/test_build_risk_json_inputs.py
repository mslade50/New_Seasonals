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


def test_daily_producer_uses_same_master_snapshot(tmp_path, monkeypatch):
    import daily_risk_report as report
    sp500 = [f"T{i:03d}" for i in range(55)]
    monkeypatch.setattr(abs_return_dispersion, "SP500_TICKERS", sp500)
    monkeypatch.setattr(report, "current_dir", str(tmp_path))
    def forbid_yahoo(*args, **kwargs):
        raise AssertionError("Risk producer must not fetch a separate Yahoo vintage")
    monkeypatch.setattr(report, "refresh_all_data", forbid_yahoo)
    (tmp_path / "data").mkdir()
    rows = [_row(t, "2026-09-21", 100.) for t in ["SPY", "^VIX", *sp500]]
    pd.DataFrame(rows).to_parquet(tmp_path / "data/master_prices.parquet", index=False)
    spy, _, _ = report.download_data()
    assert spy.index.max() == pd.Timestamp("2026-09-21")


def test_risk_pull_requires_master_snapshot():
    from scripts.pull_scan_caches import SETS
    assert ("master_prices.parquet", "data/master_prices.parquet") in SETS["risk"][0]
