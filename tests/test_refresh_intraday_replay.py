"""Protect source continuity, causal rolls and immutable replay history."""
import copy
import json

import pandas as pd
import pytest

from scripts import build_intraday_replay as baseline
from scripts import refresh_intraday_replay as refresh


def _roll_bars():
    rows = []
    for day in ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15"]:
        for symbol in ["NQU6", "NQZ6"]:
            rows.append({"time": pd.Timestamp(day + " 12:00", tz="UTC"), "symbol": symbol,
                         "volume": 100 if symbol == "NQU6" else 200 if day >= "2026-09-10" else 10})
    return pd.DataFrame(rows).set_index("time")


def test_roll_uses_prior_volume_and_never_todays_leader():
    source = _roll_bars()
    front = refresh.volume_front(source)
    assert front.loc["2026-09-10"].symbol.tolist() == ["NQU6"]
    assert front.loc["2026-09-11"].symbol.tolist() == ["NQU6"]
    assert front.loc["2026-09-14"].symbol.tolist() == ["NQZ6"]
    truncated = refresh.volume_front(source.loc[:"2026-09-14"])
    pd.testing.assert_frame_equal(front.loc[:"2026-09-14"], truncated)


def test_ambiguous_roll_fails_instead_of_choosing_a_contract():
    source = _roll_bars()
    source.loc["2026-09-08", "volume"] = 100
    with pytest.raises(ValueError, match="Ambiguous volume front"):
        refresh.volume_front(source)


def _futures_minutes():
    frames = []
    for n, day in enumerate(["2026-01-02", "2026-01-05", "2026-01-06"]):
        index = pd.date_range(day + " 09:30", periods=390, freq="min", tz=refresh.NY)
        price = pd.Series([100 + n * 100 + i * .01 for i in range(390)], index=index)
        frames.append(pd.DataFrame({"open": price, "high": price + .01, "low": price - .01,
                                   "close": price, "volume": 100, "instrument_id": 1}))
    return pd.concat(frames)


def test_utc_store_and_new_york_reference_produce_the_same_candidates():
    source = _futures_minutes()
    ny = refresh.legend_candidates({"NQ": source}, "2026-01-01", "2026-01-06")
    utc = refresh.legend_candidates({"NQ": source.tz_convert("UTC")}, "2026-01-01", "2026-01-06")
    assert ny.entry_date.tolist() == [pd.Timestamp("2026-01-06")]
    pd.testing.assert_frame_equal(utc, ny)


def test_missing_cash_minutes_block_a_coverage_extension(monkeypatch, tmp_path):
    source = _futures_minutes().tz_convert("UTC").drop(pd.Timestamp("2026-01-05 10:00", tz=refresh.NY))
    schedule = pd.DataFrame({"close": [pd.Timestamp("2026-01-05 16:00", tz=refresh.NY)]},
                            index=[pd.Timestamp("2026-01-05")])
    monkeypatch.setattr(baseline, "OB_MARKETS", {"NQ": {"bps": 15, "mult": 2}})
    with pytest.raises(ValueError, match="unvalidated missing sessions"):
        refresh.ob_extension({"NQ": source}, schedule, None, None, "2026-01-02", "2026-01-05", tmp_path)


def _strategy():
    return {"id": "open_breakout", "span": ["2026-08-28", "2026-08-28"],
            "trades": [{"trade_id": "retained", "Exit_Date": "2026-08-28", "PnL_flat": 25}],
            "daily": [["2026-08-28", 25]],
            "by_market": {"NQ": [["2026-08-28", 25]], "ES": [["2026-08-28", 0]]},
            "stats": {}}


def _addition(**overrides):
    return {"trade_id": "new", "Exit_Date": "2026-08-31", "Ticker": "MES", "PnL_flat": -10, **overrides}


def test_extension_preserves_history_and_includes_validated_zero_days():
    original = _strategy()
    retained = copy.deepcopy(original)
    extended = refresh.extend_strategy(original, [_addition()], "2026-09-01")
    assert original == retained
    assert extended["trades"][0] == retained["trades"][0]
    assert extended["daily"] == [["2026-08-28", 25], ["2026-08-31", -10], ["2026-09-01", 0]]
    assert extended["by_market"]["ES"] == [["2026-08-28", 0], ["2026-08-31", -10], ["2026-09-01", 0]]
    assert extended["stats"]["sum_usd"] == 15


@pytest.mark.parametrize("day", ["2026-08-28", "2026-09-02", "2026-08-29"])
def test_extension_rejects_overlap_future_or_weekend_trades(day):
    with pytest.raises(ValueError, match="Extension"):
        refresh.extend_strategy(_strategy(), [_addition(Exit_Date=day)], "2026-09-01")


def test_extension_rejects_duplicate_trade_ids():
    with pytest.raises(ValueError, match="Duplicate trade IDs"):
        refresh.extend_strategy(_strategy(), [_addition(trade_id="retained")], "2026-09-01")


def test_baseline_producer_cannot_reset_extended_coverage(monkeypatch, tmp_path):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"strategies": [{"id": "open_breakout", "span": ["2018-01-02", "2026-10-05"]}]}))
    monkeypatch.setattr(baseline, "OUT", path)
    with pytest.raises(SystemExit, match="Refusing to replace extended replay"):
        baseline.main()
