import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trend_sleeve import (
    TREND_FRAG_THRESHOLD,
    apply_trend_fragility_gate,
    read_trend_fragility_gate,
)


def _write_frag(path, dates, values):
    pd.DataFrame({"63d": values}, index=pd.to_datetime(dates)).to_parquet(path)


def test_gate_blocks_above_50_and_zeros_all_weights(tmp_path):
    path = tmp_path / "frag.parquet"
    dates = pd.bdate_range("2026-07-17", periods=10)
    _write_frag(path, dates, [51.0] * 10)

    gate = read_trend_fragility_gate(path, asof=dates[-1])
    assert gate["valid"] is True
    assert gate["active"] is True
    assert gate["score"] == 51.0

    targets = pd.DataFrame({"Ticker": ["SPY", "GLD"], "Weight": [0.2, 0.1]})
    gated = apply_trend_fragility_gate(targets, gate)
    assert gated["Weight"].eq(0.0).all()
    assert gated["Fragility_Gate"].eq("CASH").all()


def test_exactly_50_remains_open(tmp_path):
    path = tmp_path / "frag.parquet"
    dates = pd.bdate_range("2026-07-17", periods=10)
    _write_frag(path, dates, [TREND_FRAG_THRESHOLD] * 10)

    gate = read_trend_fragility_gate(path, asof=dates[-1])
    assert gate["valid"] is True
    assert gate["active"] is False

    targets = pd.DataFrame({"Ticker": ["SPY"], "Weight": [0.2]})
    gated = apply_trend_fragility_gate(targets, gate)
    assert gated.loc[0, "Weight"] == 0.2
    assert gated.loc[0, "Fragility_Gate"] == "OPEN"


def test_missing_and_stale_dial_fail_closed(tmp_path):
    missing = read_trend_fragility_gate(
        tmp_path / "missing.parquet", asof="2026-07-31")
    assert missing["active"] is True
    assert missing["valid"] is False

    path = tmp_path / "frag.parquet"
    _write_frag(path, ["2026-07-01"], [10.0])
    stale = read_trend_fragility_gate(path, asof="2026-07-31")
    assert stale["active"] is True
    assert stale["valid"] is False
    assert "stale" in stale["reason"]


def test_gate_uses_only_readings_available_by_signal_date(tmp_path):
    path = tmp_path / "frag.parquet"
    _write_frag(
        path,
        ["2026-07-29", "2026-07-30", "2026-08-03"],
        [49.0, 49.0, 90.0],
    )
    gate = read_trend_fragility_gate(path, asof="2026-07-31")
    assert gate["score"] == 49.0
    assert gate["active"] is False
    assert gate["asof"] == "2026-07-30"
