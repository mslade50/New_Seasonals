import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts import build_site


def _prices(path, spy_date):
    pd.DataFrame({"ticker": ["SPY"], "date": [pd.Timestamp(spy_date)]}).to_parquet(path)


def test_source_freshness_accepts_recent_ledger_and_previous_session_prices(tmp_path, monkeypatch):
    _cbd, _expected, prev_td = build_site.trading_day_offsets()
    prices = tmp_path / "prices.parquet"
    _prices(prices, prev_td)
    monkeypatch.setattr(build_site, "MASTER_PRICES", str(prices))
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(build_site, "_ledger_provenance", lambda: {
        "build_utc": now, "source": "test", "git_sha": "abc", "rows": "1",
    })
    assert build_site.source_freshness_errors() == []


def test_source_freshness_rejects_old_ledger_and_old_prices(tmp_path, monkeypatch):
    _cbd, _expected, prev_td = build_site.trading_day_offsets()
    prices = tmp_path / "prices.parquet"
    _prices(prices, prev_td - pd.Timedelta(days=10))
    monkeypatch.setattr(build_site, "MASTER_PRICES", str(prices))
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=4)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(build_site, "_ledger_provenance", lambda: {
        "build_utc": old, "source": "test", "git_sha": "abc", "rows": "1",
    })
    problems = build_site.source_freshness_errors()
    assert any("ledger is" in problem for problem in problems)
    assert any("SPY price cache ends" in problem for problem in problems)
