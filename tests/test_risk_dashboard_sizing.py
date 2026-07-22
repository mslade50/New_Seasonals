"""Risk-dashboard sizing surfaces must use the append-only PIT statistic."""

from pathlib import Path

import pandas as pd

from fragility_core import load_pit_sizing_state


def test_pit_sizing_state_is_ma10_of_stored_63d(tmp_path: Path):
    dates = pd.bdate_range("2026-07-01", periods=12)
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
              70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    path = tmp_path / "rd2_fragility.parquet"
    pd.DataFrame({"5d": values, "21d": values, "63d": values},
                 index=dates).to_parquet(path)

    state = load_pit_sizing_state(path=str(path), asof=dates[-1])

    assert state is not None
    assert state["dial_63d"] == 120.0
    assert state["score"] == sum(values[-10:]) / 10
    assert state["throttle_on"] is True
    assert state["stale"] is False
    assert state["asof"] == dates[-1].strftime("%Y-%m-%d")


def test_pit_sizing_state_marks_stale_without_hiding_value(tmp_path: Path):
    path = tmp_path / "rd2_fragility.parquet"
    pd.DataFrame({"63d": [45.0]}, index=pd.to_datetime(["2026-07-10"])).to_parquet(path)

    state = load_pit_sizing_state(
        path=str(path), asof="2026-07-20", stale_td=3)

    assert state is not None
    assert state["score"] == 45.0
    assert state["stale"] is True
    assert state["age_td"] > state["stale_td"]


def test_site_prefers_pit_sizing_series_over_recompute_series():
    risk_js = (Path(__file__).resolve().parents[1] / "site" / "assets" / "risk.js").read_text(
        encoding="utf-8")

    assert 'riskValues = sz.spark.ma' in risk_js
    assert 'Sizing Fragility' in risk_js
    assert 'display recompute · legacy payload · not a sizing input' in risk_js
