"""Guard: Risk Agent state builder, data cache allowlist and lab helpers."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_risk_agent_state as brs  # noqa: E402
import risk_agent_data as rad  # noqa: E402
import risk_agent_lab as lab  # noqa: E402
from risk_agent_grammar import chain_quote_key  # noqa: E402

N = 1800


def _bars(ticker, dates, seed, start=100.0, drift=0.0003):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(len(dates))))
    return pd.DataFrame({"ticker": ticker, "date": dates, "Open": close * 0.999,
                         "High": close * 1.006, "Low": close * 0.994, "Close": close,
                         "Volume": 1e6})


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    dates = pd.bdate_range(end="2026-10-08", periods=N)
    frames = [_bars("SPY", dates, 1), _bars("QQQ", dates, 2), _bars("ES=F", dates, 3),
              _bars("^VIX", dates, 4, 15), _bars("^VIX3M", dates, 5, 18),
              _bars("TLT", dates[:-40], 6)]          # TLT stale by 40 sessions
    pd.concat(frames).to_parquet(rad.local_path("master_prices.parquet", tmp_path))
    snap = "2026-10-08"
    rows = []
    for exp, dte in (("20261106", 29), ("20261231", 84)):
        for k in (700, 750, 774, 800, 850):
            for right in ("C", "P"):
                rows.append({"date": snap, "ticker": "SPY", "spot": 774.2, "pulled_at": 1.0,
                             "expiry": exp, "dte": dte, "strike": float(k), "right": right,
                             "con_id": 1000 + k, "bid": 1.0, "ask": 1.2, "mid": 1.1, "iv": 0.15,
                             "delta": 0.5 if right == "C" else -0.5, "gamma": 0.01, "theta": -0.1,
                             "vega": 0.1, "oi": 10.0, "volume": 5.0})
    rows.append({**rows[0], "expiry": "20261016", "dte": 8, "strike": 774.0})   # 8 DTE kept
    rows.append({**rows[0], "expiry": "20261009", "dte": 1})                    # 1 DTE kept (weeklies in scope)
    pos = pd.DataFrame(rows)
    pos.to_parquet(rad.local_path("options/positioning_history.parquet", tmp_path))
    pd.DataFrame({"date": dates[-300:], "ticker": "SPY", "iv30": np.linspace(.1, .2, 300),
                  "source": "t", "pulled_at": 1.0}).to_parquet(
        rad.local_path("options/iv_history.parquet", tmp_path))
    monkeypatch.setitem(sys.modules, "risk_agent_ledger", types.SimpleNamespace(
        load=lambda path, pull=False: [],
        replay=lambda rec: {"nav": 201000.0, "cash": 150000.0, "realized_pnl": 1000.0,
                            "positions": {}, "pending": [], "last_mark_date": "2026-10-08",
                            "marks": []}))
    return tmp_path


def test_state_builds_and_quotes_match_tape(cache, tmp_path):
    jp = tmp_path / "journal.jsonl"
    jp.write_text(json.dumps({"kind": "decision", "asof": "2026-10-07", "decision": {
        "mode": "decision", "posture": {"summary": "x"}, "forecasts": [],
        "positions": [], "watchlist": [{"idea": "w"}]}}) + "\n", encoding="utf-8")
    st = brs.build_state(cache_dir=cache, journal_path=jp)
    assert st["recent_decisions"][0]["asof"] == "2026-10-07" and st["watchlist"] == [{"idea": "w"}]
    assert st["schema_version"] == "risk_agent_state.v2"
    assert st["asof"] == "2026-10-08"
    assert st["session"]["next_session"] == "2026-10-09"
    assert st["sleeve"]["nav"] == 201000.0
    for sym, q in st["quotes"].items():
        assert q["close"] == st["tape"][sym]["close"]
        assert q["atr"] == st["tape"][sym]["atr"]
    assert "SPY" in st["quotes"] and "ES=F" in st["quotes"]
    assert st["tape"]["ES=F"]["kind"] == "future" and "ES" in st["tape"]["ES=F"]["roots"]
    assert st["tape"]["SPY"]["iv30"] is not None
    assert st["vol"]["vix_vix3m_ratio"] is not None
    assert len(json.dumps(st)) < 250_000


def test_stale_etf_dropped_with_warning(cache, tmp_path):
    st = brs.build_state(cache_dir=cache, journal_path=tmp_path / "none.jsonl")
    assert "TLT" not in st["tape"] and "TLT" not in st["quotes"]
    assert any("TLT" in w and "older than" in w for w in st["warnings"])
    assert any("IWM" in w and "absent" in w for w in st["warnings"])


def test_ledger_missing_degrades(cache, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "risk_agent_ledger", None)   # ImportError
    jp = tmp_path / "j.jsonl"
    jp.write_text("{}\n", encoding="utf-8")
    st = brs.build_state(cache_dir=cache, journal_path=jp)
    assert st["sleeve"]["nav"] == 200000.0
    assert any("risk_agent_ledger" in w for w in st["warnings"])


def test_chain_key_format_and_dte_window(cache):
    ch = brs.build_chains("2026-10-08", cache)
    assert set(ch) == {"SPY"}
    q = ch["SPY"]["quotes"]
    assert chain_quote_key("2026-11-06", 774.0, "C") in q or chain_quote_key("2026-11-06", 750.0, "C") in q
    assert "2026-11-06|750|C" in q
    assert any(k.startswith("2026-10-09|") for k in q)          # 1 DTE kept
    assert any(k.startswith("2026-10-16|") for k in q)          # 8 DTE kept
    for key, v in q.items():
        e, k, r = key.split("|")
        assert key == chain_quote_key(e, float(k), r)
        assert {"bid", "ask", "mid", "iv", "delta", "con_id", "oi", "volume"} <= set(v)
    assert ch["SPY"]["asof"] == "2026-10-08"


def test_stress_roughly_monotone(cache, tmp_path):
    st = brs.build_state(cache_dir=cache, journal_path=tmp_path / "none.jsonl")
    s = st["stress"]["SPY"]
    vals = [s[h] for h in ("5", "10", "21", "42", "63", "126")]
    assert vals[-1] > vals[0] > 0
    assert all(b >= a - 0.02 for a, b in zip(vals, vals[1:]))


def test_sync_refuses_denied_key(tmp_path):
    with pytest.raises(PermissionError):
        rad.sync(["master_prices.parquet", "live_fills.parquet"], cache_dir=tmp_path)
    for bad in ("exposure_state.json", "pitch_journal.jsonl", "backtest_trades_full.parquet",
                "site/anything.json", "rd2_environment.json"):
        with pytest.raises(PermissionError):
            rad.local_path(bad)
    assert not list(tmp_path.iterdir())          # nothing downloaded or created
    for k in rad.DEFAULT_KEYS:
        assert rad.local_path(k, tmp_path).name == k.replace("/", "__")


def test_catalog_lists_cached_objects(cache):
    cat = rad.catalog(cache)
    row = next(r for r in cat if r["key"] == "master_prices.parquet")
    assert row["rows"] == N * 5 + (N - 40) and row["last_date"].startswith("2026-10-08")


def test_lab_study_and_prices(cache):
    px = lab.prices(["SPY"], cache_dir=cache)
    assert list(px.columns) == ["SPY"]
    f = lab.fwd_returns(px["SPY"], 5)
    assert np.isnan(f.iloc[-1])      # needs lag+h future bars
    mask = px["SPY"] > px["SPY"].rolling(50).mean()
    out = lab.study(mask, f, decluster_td=5)
    assert out["n"] > 10 and out["n"] < out["uncond"]["n"]
    assert 0 <= out["sign_p"] <= 1
    assert abs(out["edge_mean_pct"] - (out["mean_pct"] - out["uncond"]["mean_pct"])) < 1e-9
