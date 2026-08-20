import json

import numpy as np
import pandas as pd

import data_provider
from pages import strat_backtester
from survivorship_contract import (
    SURVIVORSHIP_BASIS,
    SURVIVORSHIP_CONTRACT_VERSION,
    SURVIVORSHIP_SCOPE_START,
    sha256_file,
)


def _long_prices(ticker, rows=300, volume=150_000.0):
    dates = pd.bdate_range("2020-01-02", periods=rows)
    close = np.full(rows, 50.0)
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "Open": close,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.full(rows, volume),
    })


def _write_survivor_pair(tmp_path, ticker="OLD"):
    artifact = tmp_path / "survivorship.parquet"
    manifest = tmp_path / "survivorship.meta.json"
    frame = _long_prices(ticker)
    frame.to_parquet(artifact, index=False)
    manifest.write_text(json.dumps({
        "contract_version": SURVIVORSHIP_CONTRACT_VERSION,
        "basis": SURVIVORSHIP_BASIS,
        "scope_start": SURVIVORSHIP_SCOPE_START,
        "required_tickers": [ticker],
        "covered_by_primary_sources": [],
        "artifact_tickers": [ticker],
        "artifact_rows": len(frame),
        "artifact_sha256": sha256_file(artifact),
        "unresolved_required": [],
    }), encoding="utf-8")
    return artifact, manifest


def test_data_provider_requires_explicit_survivorship_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "survivorship_contract.SURVIVORSHIP_REQUIRED_TICKERS_V1",
        frozenset({"OLD"}),
    )
    master = tmp_path / "master.parquet"
    _long_prices("LIVE").to_parquet(master, index=False)
    survivor, manifest = _write_survivor_pair(tmp_path)
    monkeypatch.setattr(data_provider, "MASTER_PATH", str(master))
    monkeypatch.setattr(data_provider, "OVERFLOW_PATH", str(tmp_path / "missing-overflow"))
    monkeypatch.setattr(data_provider, "SURVIVORSHIP_PATH", str(survivor))
    monkeypatch.setattr(data_provider, "SURVIVORSHIP_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(data_provider, "_refresh_from_r2_if_needed", lambda: None)
    monkeypatch.setattr(data_provider, "_refresh_survivorship_from_r2_if_needed", lambda: None)
    data_provider._SURVIVORSHIP_VALIDATION_CACHE.clear()

    assert data_provider.get_history(["OLD"]) == {}
    loaded = data_provider.get_history(["OLD"], include_survivorship=True)
    assert set(loaded) == {"OLD"}
    assert data_provider.get_survivorship_tickers() == {"OLD"}


def test_candidate_generation_applies_point_in_time_overflow_gate(monkeypatch):
    frame = _long_prices("OLD").drop(columns="ticker").set_index("date")
    frame["ATR"] = 2.0
    frame["RangePct"] = (frame["High"] - frame["Low"]) / frame["Close"]
    frame["vol_ratio"] = 1.0
    monkeypatch.setattr(
        strat_backtester,
        "get_historical_mask",
        lambda df, _settings, _sznl, _ticker: pd.Series(True, index=df.index),
    )
    strategy = {
        "name": "Oversold Low Volume",
        "settings": {},
        "universe_tickers": ["OLD"],
        "_overflow_pass": True,
        "_pit_overflow_filter": True,
    }
    candidates, _ = strat_backtester.generate_candidates_fast(
        {"OLD": frame}, [strategy], {}, frame.index.min()
    )
    assert len(candidates) == len(frame) - 251
    assert pd.Timestamp(candidates[0][0]) == frame.index[251]

    thin = frame.copy()
    thin["Volume"] = 10_000.0
    candidates, _ = strat_backtester.generate_candidates_fast(
        {"OLD": thin}, [strategy], {}, thin.index.min()
    )
    assert candidates == []
