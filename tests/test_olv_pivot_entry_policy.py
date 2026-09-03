"""Production guards for OLV's causal 40/40 pivot-entry policy."""

import copy
import os
import sys
import types
from email import message_from_string

import numpy as np
import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pages"))


class _NoOp:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cache_data(self, *args, **kwargs):
        def deco(fn):
            return fn

        return deco

    cache_resource = cache_data


sys.modules.setdefault("streamlit", _NoOp())

import daily_scan
import verify_fills
from olv_pivot_entry import (
    OLV_PIVOT_HIGH_COL,
    OLV_PIVOT_HIGH_DATE_COL,
    OLV_PIVOT_HIGH_SOURCE_AGE_COL,
    OLV_PIVOT_LOW_COL,
    OLV_PIVOT_LOW_DATE_COL,
    OLV_PIVOT_LOW_SOURCE_AGE_COL,
    causal_close_pivot_context,
    resolve_olv_pivot_entry,
)
from strategy_config import STRATEGY_BOOK
from strat_backtester import (
    INDICATOR_CACHE_VERSION,
    _indicator_cache_has_required_schema,
    process_signals_fast,
)


def _policy():
    olv = next(s for s in STRATEGY_BOOK if s["name"] == "Oversold Low Volume")
    return copy.deepcopy(olv["execution"]["pivot_entry_policy"])


def test_production_policy_is_live_and_exact():
    policy = _policy()
    assert policy["enabled"] is True
    assert (policy["left_bars"], policy["right_bars"]) == (40, 40)
    assert policy["max_source_age_bars"] == 252
    assert policy["version"] == "olv_close_pivot_40_v2_20260901"
    assert policy["default_offset_atr"] == pytest.approx(0.25)


def test_indicator_cache_contract_requires_pivot_columns():
    assert INDICATOR_CACHE_VERSION == "v3"
    stale = pd.DataFrame({"Close": [100.0]})
    assert not _indicator_cache_has_required_schema(stale)
    fresh = stale.assign(
        **{
            OLV_PIVOT_HIGH_COL: [99.0],
            OLV_PIVOT_HIGH_DATE_COL: [pd.Timestamp("2025-01-01")],
            OLV_PIVOT_HIGH_SOURCE_AGE_COL: [100.0],
            OLV_PIVOT_LOW_COL: [90.0],
            OLV_PIVOT_LOW_DATE_COL: [pd.Timestamp("2024-12-01")],
            OLV_PIVOT_LOW_SOURCE_AGE_COL: [120.0],
        }
    )
    assert _indicator_cache_has_required_schema(fresh)


def test_causal_pivot_appears_only_after_right_40_closes():
    dates = pd.bdate_range("2024-01-02", periods=100)
    # Unique centered high at p=40. It is first knowable at q=80.
    values = np.r_[np.arange(60.0, 101.0), np.arange(99.0, 40.0, -1.0)]
    close = pd.Series(values, index=dates)
    context = causal_close_pivot_context(close)

    assert pd.isna(context.iloc[79][OLV_PIVOT_HIGH_COL])
    assert context.iloc[80][OLV_PIVOT_HIGH_COL] == pytest.approx(100.0)
    assert pd.Timestamp(context.iloc[80][OLV_PIVOT_HIGH_DATE_COL]) == dates[40]
    assert context.iloc[80][OLV_PIVOT_HIGH_SOURCE_AGE_COL] == pytest.approx(40.0)
    assert context.iloc[99][OLV_PIVOT_HIGH_SOURCE_AGE_COL] == pytest.approx(59.0)

    # Adding later bars cannot rewrite context already knowable at q=80.
    extended_dates = pd.bdate_range(dates[0], periods=130)
    extended = pd.Series(
        np.r_[values, np.linspace(45.0, 130.0, 30)], index=extended_dates
    )
    extended_context = causal_close_pivot_context(extended)
    pd.testing.assert_series_equal(
        context[OLV_PIVOT_HIGH_COL],
        extended_context.loc[dates, OLV_PIVOT_HIGH_COL],
        check_names=False,
    )


def test_causal_low_age_is_counted_from_source_bar():
    dates = pd.bdate_range("2024-01-02", periods=100)
    # Unique centered low at p=40, first exposed at q=80 with source age 40.
    values = np.r_[np.arange(100.0, 59.0, -1.0), np.arange(61.0, 120.0)]
    context = causal_close_pivot_context(pd.Series(values, index=dates))
    assert pd.isna(context.iloc[79][OLV_PIVOT_LOW_COL])
    assert context.iloc[80][OLV_PIVOT_LOW_COL] == pytest.approx(60.0)
    assert pd.Timestamp(context.iloc[80][OLV_PIVOT_LOW_DATE_COL]) == dates[40]
    assert context.iloc[80][OLV_PIVOT_LOW_SOURCE_AGE_COL] == pytest.approx(40.0)
    assert context.iloc[99][OLV_PIVOT_LOW_SOURCE_AGE_COL] == pytest.approx(59.0)


@pytest.mark.parametrize(
    "distance, expected_action, expected_offset, expected_rule",
    [
        (2.0, "stage", 0.25, "default"),
        (2.000001, "stage", 0.50, "above_high_2_3"),
        (3.0, "stage", 0.50, "above_high_2_3"),
        (3.000001, "stage", 0.25, "default"),
        (4.0, "stage", 0.25, "default"),
        (4.000001, "stage", 0.75, "above_high_4_5"),
        (5.0, "stage", 0.75, "above_high_4_5"),
        (5.000001, "skip", 0.25, "above_high_gt5"),
    ],
)
def test_policy_boundaries(distance, expected_action, expected_offset, expected_rule):
    decision = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=100.0 - distance * 2.0,
        pivot_low=40.0,
        pivot_high_source_age_bars=100,
        pivot_low_source_age_bars=120,
        policy=_policy(),
    )
    assert decision["nearest_type"] == "High"
    assert decision["action"] == expected_action
    assert decision["offset_atr"] == pytest.approx(expected_offset)
    assert decision["matched_rule"] == expected_rule


def test_nearest_low_and_disabled_policy_fall_back_cleanly():
    low_nearest = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=80.0,
        pivot_low=99.0,
        pivot_high_source_age_bars=100,
        pivot_low_source_age_bars=120,
        policy=_policy(),
    )
    assert low_nearest["nearest_type"] == "Low"
    assert low_nearest["offset_atr"] == pytest.approx(0.25)
    assert not low_nearest["skip"]

    disabled = _policy()
    disabled["enabled"] = False
    shadow = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=88.0,
        pivot_low=40.0,
        pivot_high_source_age_bars=100,
        pivot_low_source_age_bars=120,
        policy=disabled,
    )
    assert shadow["proposed_action"] == "skip"
    assert shadow["action"] == "stage"
    assert shadow["offset_atr"] == pytest.approx(0.25)


def test_source_age_252_is_valid_and_253_reselects_other_side():
    valid = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=88.0,
        pivot_low=40.0,
        pivot_high_source_age_bars=252,
        pivot_low_source_age_bars=100,
        policy=_policy(),
    )
    assert valid["nearest_type"] == "High"
    assert valid["nearest_source_age_bars"] == pytest.approx(252)
    assert valid["skip"]
    assert not valid["pivot_high_expired"]

    expired = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=88.0,
        pivot_low=40.0,
        pivot_high_source_age_bars=253,
        pivot_low_source_age_bars=100,
        policy=_policy(),
    )
    assert expired["nearest_type"] == "Low"
    assert expired["nearest_source_age_bars"] == pytest.approx(100)
    assert expired["pivot_high_expired"]
    assert not expired["pivot_low_expired"]
    assert expired["offset_atr"] == pytest.approx(0.25)
    assert not expired["skip"]


def test_low_expires_independently_before_nearest_selection():
    decision = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=95.0,
        pivot_low=99.0,
        pivot_high_source_age_bars=100,
        pivot_low_source_age_bars=253,
        policy=_policy(),
    )
    # The stale low is closer in price, but the fresh high must be selected.
    assert decision["nearest_type"] == "High"
    assert decision["pivot_low_expired"]
    assert decision["offset_atr"] == pytest.approx(0.50)
    assert decision["matched_rule"] == "above_high_2_3"


@pytest.mark.parametrize(
    "high_age, low_age",
    [(253, 253), (None, None)],
)
def test_both_unusable_levels_fail_safe_to_default_entry(high_age, low_age):
    decision = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=88.0,
        pivot_low=99.0,
        pivot_high_source_age_bars=high_age,
        pivot_low_source_age_bars=low_age,
        policy=_policy(),
    )
    assert decision["nearest_type"] == ""
    assert decision["nearest_level"] is None
    assert decision["nearest_source_age_bars"] is None
    assert decision["offset_atr"] == pytest.approx(0.25)
    assert decision["matched_rule"] == "default"
    assert not decision["skip"]


def test_policy_without_age_cap_retains_backward_compatibility():
    policy = _policy()
    policy.pop("max_source_age_bars")
    decision = resolve_olv_pivot_entry(
        signal_close=100.0,
        atr=2.0,
        pivot_high=88.0,
        pivot_low=40.0,
        policy=policy,
    )
    assert decision["nearest_type"] == "High"
    assert decision["skip"]
    assert decision["max_source_age_bars"] is None


@pytest.mark.parametrize("bad_age", [-1, "not-a-number", np.inf])
def test_invalid_age_cap_fails_loudly(bad_age):
    policy = _policy()
    policy["max_source_age_bars"] = bad_age
    with pytest.raises(ValueError, match="max_source_age_bars"):
        resolve_olv_pivot_entry(
            signal_close=100.0,
            atr=2.0,
            pivot_high=88.0,
            pivot_low=40.0,
            pivot_high_source_age_bars=100,
            pivot_low_source_age_bars=100,
            policy=policy,
        )


def _engine_case(
    distance_atr,
    policy_enabled=True,
    pivot_high_source_age_bars=100,
    pivot_low_source_age_bars=120,
):
    n = 12
    dates = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            # Touches the deepest tested 0.75-ATR entry (98.5) without reaching
            # any strategy stop; keeps this test focused on entry resolution.
            "Low": [98.4] * n,
            "Close": [100.0] * n,
            "ATR": [2.0] * n,
            "RangePct": [0.02] * n,
            "vol_ratio": [1.0] * n,
            "Sznl": [50.0] * n,
            "atr_sznl_5d": [50.0] * n,
            "rank_ret_126d": [50.0] * n,
            "rank_ret_252d": [50.0] * n,
            OLV_PIVOT_HIGH_COL: [100.0 - 2.0 * distance_atr] * n,
            OLV_PIVOT_HIGH_DATE_COL: [dates[0] - pd.Timedelta(days=100)] * n,
            OLV_PIVOT_HIGH_SOURCE_AGE_COL: [pivot_high_source_age_bars] * n,
            OLV_PIVOT_LOW_COL: [40.0] * n,
            OLV_PIVOT_LOW_DATE_COL: [dates[0] - pd.Timedelta(days=120)] * n,
            OLV_PIVOT_LOW_SOURCE_AGE_COL: [pivot_low_source_age_bars] * n,
        },
        index=dates,
    )
    candidates = [(int(dates[0].value), "TEST", "TEST", 0, 0)]
    signal_data = {
        ("TEST", 0): {
            "atr": 2.0,
            "close": 100.0,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "vol_ratio": 1.0,
            "sznl": 50.0,
            "range_pct": 2.0,
            "atr_sznl_5d": 50.0,
            "rank_ret_126d": 50.0,
            "rank_ret_252d": 50.0,
        }
    }
    pivot_policy = _policy()
    pivot_policy["enabled"] = policy_enabled
    strategy = {
        "name": "TEST OLV PIVOT",
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": False,
        },
        "execution": {
            "risk_bps": 35,
            "slippage_bps": 2,
            "stop_atr": 1.25,
            "tgt_atr": 2.5,
            "hold_days": 10,
            "fill_window_days": 3,
            "use_stop_loss": True,
            "use_take_profit": True,
            "pivot_entry_policy": pivot_policy,
        },
        "universe_tickers": ["TEST"],
    }
    return process_signals_fast(
        candidates,
        signal_data,
        {"TEST": df},
        [strategy],
        starting_equity=100_000,
    )


@pytest.mark.parametrize(
    "distance, expected_offset, expected_price",
    [(2.5, 0.50, 99.0), (3.5, 0.25, 99.5), (4.5, 0.75, 98.5)],
)
def test_backtester_uses_same_dynamic_offset(distance, expected_offset, expected_price):
    result = _engine_case(distance)
    assert len(result) == 1
    assert result.iloc[0]["Entry Offset ATR"] == pytest.approx(expected_offset)
    assert result.iloc[0]["Price"] == pytest.approx(expected_price)


def test_backtester_skips_above_five_before_staging():
    assert _engine_case(6.0).empty


def test_backtester_kill_switch_restores_legacy_entry():
    result = _engine_case(6.0, policy_enabled=False)
    assert len(result) == 1
    assert result.iloc[0]["Entry Offset ATR"] == pytest.approx(0.25)
    assert result.iloc[0]["Price"] == pytest.approx(99.5)


def test_backtester_age_boundary_and_fresh_side_reselection():
    assert _engine_case(6.0, pivot_high_source_age_bars=252).empty

    result = _engine_case(
        6.0,
        pivot_high_source_age_bars=253,
        pivot_low_source_age_bars=100,
    )
    assert len(result) == 1
    assert result.iloc[0]["Entry Offset ATR"] == pytest.approx(0.25)
    assert result.iloc[0]["Pivot Nearest Type"] == "Low"
    assert result.iloc[0]["Pivot Source Age Bars"] == pytest.approx(100)
    assert bool(result.iloc[0]["Pivot High Expired"])
    assert not bool(result.iloc[0]["Pivot Low Expired"])


class _FakeWorksheet:
    def __init__(self):
        self.values = []

    def get_all_records(self):
        return []

    def clear(self):
        self.values = []

    def update(self, values):
        self.values = values

    def get_all_values(self):
        return self.values


class _FakeSpreadsheet:
    def __init__(self, worksheet):
        self._worksheet = worksheet

    def worksheet(self, _name):
        return self._worksheet

    @property
    def sheet1(self):
        return self._worksheet


class _FakeClient:
    def __init__(self, worksheet):
        self._sheet = _FakeSpreadsheet(worksheet)

    def open(self, _name):
        return self._sheet


def test_scanner_stages_numeric_offset_over_static_entry_string(monkeypatch):
    worksheet = _FakeWorksheet()
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeClient(worksheet))
    monkeypatch.setattr(
        daily_scan, "_sheets_write_with_retry", lambda _label, fn: fn()
    )
    strategy = {
        "id": "test-olv-pivot",
        "name": "Oversold Low Volume",
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
        },
        "execution": {
            "risk_bps": 35,
            "stop_atr": 1.25,
            "tgt_atr": 2.5,
            "hold_days": 10,
            "fill_window_days": 3,
            "use_stop_loss": True,
            "use_take_profit": True,
        },
    }
    signal = {
        "Strategy_ID": strategy["id"],
        "Ticker": "TEST",
        "Date": pd.Timestamp("2026-08-31").date(),
        "Action": "BUY",
        "Shares": 100,
        "Entry": 100.123456,
        "ATR": 2.345678,
        "Time Exit": pd.Timestamp("2026-09-15").date(),
        "Risk_Amt": 293.21,
        "Entry_Offset_ATR": 0.75,
        "Pivot_Rule_Version": "olv_close_pivot_40_v2_20260901",
        "Pivot_Nearest_Type": "High",
        "Pivot_Level": 89.0,
        "Pivot_Date": "2026-05-01",
        "Pivot_Source_Age_Bars": 200,
        "Pivot_High_Source_Age_Bars": 200,
        "Pivot_Low_Source_Age_Bars": 110,
        "Pivot_High_Expired": False,
        "Pivot_Low_Expired": False,
        "Pivot_Max_Source_Age_Bars": 252,
        "Pivot_Distance_ATR": 4.74,
        "Pivot_Matched_Rule": "above_high_4_5",
        "Live_Filters": [],
    }
    daily_scan.save_staging_orders([signal], [strategy])
    headers, values = worksheet.values[0], worksheet.values[1]
    row = dict(zip(headers, values))
    assert row["Order_Type"] == "REL_CLOSE"
    assert float(row["Offset_ATR_Mult"]) == pytest.approx(0.75)
    assert float(row["Entry_Offset_ATR"]) == pytest.approx(0.75)
    assert float(row["Frozen_ATR"]) == pytest.approx(2.345678)
    assert float(row["Signal_Close"]) == pytest.approx(100.123456)
    assert float(row["Pivot_Source_Age_Bars"]) == pytest.approx(200)
    assert float(row["Pivot_Max_Source_Age_Bars"]) == pytest.approx(252)
    assert row["Pivot_Matched_Rule"] == "above_high_4_5"


def test_verify_fills_prefers_explicit_dynamic_offset():
    row = {
        "Strategy_ID": "test-olv-pivot",
        "Entry_Type": "Limit Order -0.25 ATR (Persistent)",
        "Entry_Type_Short": "LMT $98.50 GTC",
        "Entry_Offset_ATR": 0.75,
    }
    strategy_map = {
        "test-olv-pivot": {
            "order_class": "REL_CLOSE",
            "offset": 0.25,
            "tif": "GTC",
            "entry_type_raw": row["Entry_Type"],
        }
    }
    order_class, offset, tif = verify_fills.classify_order(row, strategy_map)
    assert (order_class, tif) == ("REL_CLOSE", "GTC")
    assert offset == pytest.approx(0.75)

    # The explicit offset must also survive historical rows whose concise
    # Entry_Type_Short label is absent and therefore use config routing.
    fallback_row = dict(row)
    fallback_row.pop("Entry_Type_Short")
    order_class, offset, tif = verify_fills.classify_order(
        fallback_row, strategy_map)
    assert (order_class, tif) == ("REL_CLOSE", "GTC")
    assert offset == pytest.approx(0.75)


def test_signal_log_preserves_dynamic_frozen_input_precision(monkeypatch):
    worksheet = _FakeWorksheet()
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeClient(worksheet))
    signals = pd.DataFrame(
        [{
            "Ticker": "TEST",
            "Date": pd.Timestamp("2026-08-31").date(),
            "Strategy_ID": "test-olv-pivot",
            "Entry": 116.0497859,
            "ATR": 1.0583841,
            "Stop": 97.19,
            "Target": 105.99,
            "Entry_Offset_ATR": 0.75,
            "Entry_Type_Short": "LMT $115.26 GTC",
        }]
    )
    daily_scan.save_signals_to_gsheet(signals)
    headers, values = worksheet.values[0], worksheet.values[1]
    row = dict(zip(headers, values))
    assert float(row["Entry"]) == pytest.approx(116.049786)
    assert float(row["ATR"]) == pytest.approx(1.058384)
    # Non-load-bearing display levels retain the legacy compact rounding.
    assert float(row["Stop"]) == pytest.approx(97.19)

    order_class, offset, tif = verify_fills.classify_order(
        row,
        {
            "test-olv-pivot": {
                "order_class": "REL_CLOSE",
                "offset": 0.25,
                "tif": "GTC",
                "entry_type_raw": "Limit Order -0.25 ATR (Persistent)",
            }
        },
    )
    prices = pd.DataFrame(
        {"Open": [116.0], "High": [116.2], "Low": [115.255], "Close": [115.8]},
        index=[pd.Timestamp("2026-09-01")],
    )
    status, _fill_date, fill_price = verify_fills.check_fill(
        order_class,
        "BUY",
        float(row["Entry"]),
        float(row["ATR"]),
        offset,
        None,
        prices,
        pd.Timestamp("2026-08-31").date(),
        pd.Timestamp("2026-09-15").date(),
        tif,
        fill_window=3,
    )
    assert status == "FILLED"
    assert fill_price == pytest.approx(115.26)


def test_live_order_staging_consumes_numeric_offset():
    ibkr_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "trading_ibkr")
    if not os.path.isdir(ibkr_dir):
        pytest.skip(f"live execution dir not present: {ibkr_dir}")
    sys.path.insert(0, ibkr_dir)
    try:
        import order_staging
    except ImportError as exc:
        pytest.skip(f"order_staging not importable here ({exc})")

    row = {
        "Order_Type": "REL_CLOSE",
        "Action": "BUY",
        "Signal_Close": 100.123456,
        "Frozen_ATR": 2.345678,
        "Offset_ATR_Mult": 0.75,
    }
    price, needs_open = order_staging.calculate_limit_price(row, 0.0)
    assert needs_open is False
    assert price == pytest.approx(round(100.123456 - 0.75 * 2.345678, 2))


def _olv_email_signal(**overrides):
    signal = {
        "Strategy_ID": "test-olv-pivot",
        "Strategy_Name": "Oversold Low Volume",
        "Ticker": "TEST",
        "Action": "BUY",
        "Shares": 100,
        "Risk_Amt": 293.0,
        "Notional": 10012.0,
        "Entry": 100.12,
        "ATR": 2.345678,
        "Stop": 97.19,
        "Target": 105.98,
        "Time Exit": pd.Timestamp("2026-09-15").date(),
        "Days_To_Exit": 10,
        "Entry_Type": "Limit Order -0.25 ATR (Persistent)",
        "Entry_Type_Short": "LMT $98.36 GTC",
        "Limit_Price": 98.36,
        "Entry_Offset_ATR": 0.75,
        "Fill_Window_Days": 3,
        "Pivot_Nearest_Type": "High",
        "Pivot_Level": 89.00,
        "Pivot_Date": "2026-05-01",
        "Pivot_Source_Age_Bars": 200,
        "Pivot_High_Source_Age_Bars": 200,
        "Pivot_Low_Source_Age_Bars": 110,
        "Pivot_High_Expired": False,
        "Pivot_Low_Expired": False,
        "Pivot_Max_Source_Age_Bars": 252,
        "Pivot_Distance_ATR": 4.74,
        "Pivot_Matched_Rule": "above_high_4_5",
        "Rank_252D": 64.0,
        "Setup_Type": "MeanReversion",
        "Setup_Timeframe": "Position",
        "Setup_Thesis": "LEGACY THESIS MUST NOT RENDER",
        "Setup_Filters": ["LEGACY FILTER MUST NOT RENDER"],
        "Live_Filters": [
            ("2D rank < 25th %ile", "12.0", False),
            ("5D rank < 33th %ile", "18.0", False),
            ("21D rank < 15th %ile (3d consecutive)", "8.0", False),
            ("252D rank between 50-90th %ile", "64.0", False),
            ("10D vol rank < 15th %ile", "7", False),
            ("Market > 200 SMA", "PASS", True),
        ],
        "Use_Stop": True,
        "Use_Target": True,
        "Exit_Primary": "LEGACY EXIT HEADLINE MUST NOT RENDER",
        "Exit_Notes": "LEGACY EXIT NOTES MUST NOT RENDER",
        "Sizing_Variable": "",
        "Sizing_Notes": (
            "Standard (1.0x) | Pivot40 High 4.74ATR: entry -0.75ATR | "
            "Risk: 52.5bps ($293)"
        ),
        "Stats": "WR: 60% | PF: 2.0 | Exp: 0.5r",
        "Earnings_Cov": "",
    }
    signal.update(overrides)
    return signal


def test_olv_email_brief_states_distance_age_and_entry_effect():
    signal = _olv_email_signal()
    brief = daily_scan.build_olv_email_brief(signal)
    combined = " ".join(brief.values())

    assert "2D 12.0 (<25)" in brief["why"]
    assert "21D 8.0 (<15 for at least 3 sessions)" in brief["why"]
    assert "10D volume rank 7 (<15)" in brief["why"]
    assert "252D 64.0 (50\u201390)" in brief["why"]
    assert "$11.12 / 4.74 ATR above" in brief["pivot"]
    assert "pivot high at $89.00" in brief["pivot"]
    assert (
        "May 1, 2026; 200 sessions old; expires after 252"
        in brief["pivot"]
    )
    assert "Stage $98.36 buy limit" in brief["action"]
    assert "close \u22120.75 ATR" in brief["action"]
    assert "$1.17 below the standard close \u22120.25 ATR limit" in brief["action"]
    assert "Shares and risk budget stay unchanged" in brief["action"]
    assert "the lower limit reduces notional" in brief["action"]
    assert "above_high_4_5" not in combined
    assert "Pivot40" not in combined
    assert len(combined.split()) < 140


@pytest.mark.parametrize(
    "distance_atr, offset_atr, limit_price, expected_action",
    [
        (
            2.5,
            0.50,
            100.12 - 0.50 * 2.345678,
            "close \u22120.5 ATR), $0.58 below the standard",
        ),
        (
            3.5,
            0.25,
            100.12 - 0.25 * 2.345678,
            "standard $99.53 buy limit",
        ),
    ],
)
def test_olv_email_explains_other_pivot_entry_bands(
        distance_atr, offset_atr, limit_price, expected_action):
    brief = daily_scan.build_olv_email_brief(
        _olv_email_signal(
            Pivot_Distance_ATR=distance_atr,
            Pivot_Level=100.12 - distance_atr * 2.345678,
            Entry_Offset_ATR=offset_atr,
            Limit_Price=limit_price,
        )
    )

    assert f"{distance_atr:.2f} ATR above" in brief["pivot"]
    assert expected_action in brief["action"]


def test_olv_email_reports_when_close_is_below_pivot():
    brief = daily_scan.build_olv_email_brief(
        _olv_email_signal(
            Pivot_Distance_ATR=-1.25,
            Pivot_Level=100.12 + 1.25 * 2.345678,
            Entry_Offset_ATR=0.25,
            Limit_Price=100.12 - 0.25 * 2.345678,
        )
    )

    assert "$2.93 / 1.25 ATR below" in brief["pivot"]
    assert "-1.25 ATR" not in brief["pivot"]


def test_olv_email_brief_makes_default_and_expired_levels_explicit():
    signal = _olv_email_signal(
        Limit_Price=99.53,
        Entry_Offset_ATR=0.25,
        Pivot_Nearest_Type="Low",
        Pivot_Level=97.00,
        Pivot_Date="2026-07-01",
        Pivot_Source_Age_Bars=44,
        Pivot_Distance_ATR=(100.12 - 97.00) / 2.345678,
        Pivot_High_Expired=True,
        Pivot_High_Source_Age_Bars=253,
        Pivot_Matched_Rule="default",
    )
    brief = daily_scan.build_olv_email_brief(signal)

    assert "closing-pivot low at $97.00" in brief["pivot"]
    assert (
        "Ignored stale high (253 sessions old) beyond the 252-session cap"
        in brief["pivot"]
    )
    assert "Stage the standard $99.53 buy limit" in brief["action"]
    assert "pivot did not alter the entry" in brief["action"].lower()
    assert "shares and risk budget are unchanged" in brief["action"].lower()


def test_olv_email_brief_handles_no_surviving_pivot():
    brief = daily_scan.build_olv_email_brief(
        _olv_email_signal(
            Limit_Price=99.53,
            Entry_Offset_ATR=0.25,
            Pivot_Nearest_Type="",
            Pivot_Level="",
            Pivot_Date="",
            Pivot_Source_Age_Bars="",
            Pivot_Distance_ATR="",
            Pivot_High_Expired=True,
            Pivot_Low_Expired=True,
            Pivot_High_Source_Age_Bars=300,
            Pivot_Low_Source_Age_Bars=270,
            Pivot_Matched_Rule="default",
        )
    )
    assert "No valid 40/40 closing pivot within 252 sessions" in brief["pivot"]
    assert "high (300 sessions old)" in brief["pivot"]
    assert "low (270 sessions old)" in brief["pivot"]
    assert "standard $99.53 buy limit" in brief["action"]


def test_olv_email_uses_actual_staged_action_not_proposed_rule():
    """A disabled policy may retain its proposed rule in audit fields."""
    signal = _olv_email_signal(
        Limit_Price=99.53,
        Entry_Offset_ATR="",
        Pivot_Matched_Rule="above_high_gt5",
    )
    brief = daily_scan.build_olv_email_brief(signal)

    assert "Stage the standard $99.53 buy limit" in brief["action"]
    assert "No order" not in brief["action"]
    assert daily_scan.format_signal_email_entry(signal) == (
        "Persistent limit @ $99.53 (signal close \u2212 0.25 ATR)"
    )


def test_dynamic_email_entry_uses_effective_limit_not_static_label_or_close():
    entry = daily_scan.format_signal_email_entry(_olv_email_signal())
    assert entry == "Persistent limit @ $98.36 (signal close \u2212 0.75 ATR)"
    assert "$100.12" not in entry
    assert "-0.25 ATR" not in entry


@pytest.mark.parametrize(
    "signal, expected",
    [
        (
            {"Entry_Type": "Limit (Open - 0.5 ATR)", "Entry": 100.0},
            "Limit (Open - 0.5 ATR)",
        ),
        (
            {"Entry_Type": "Signal Close", "Entry": 100.0},
            "Signal Close @ $100.00",
        ),
        (
            {
                "Entry_Type": "Limit at T+1 Close",
                "Entry": 100.0,
                "Limit_Price": 99.0,
            },
            "Limit at T+1 Close @ $100.00",
        ),
        (
            {
                "Strategy_Name": "Some Other Strategy",
                "Entry_Type": "Limit Order -0.25 ATR (Persistent)",
                "Entry": 100.0,
                "Limit_Price": 99.0,
                "Entry_Offset_ATR": 0.75,
            },
            "Limit Order -0.25 ATR (Persistent) @ $100.00",
        ),
    ],
)
def test_email_entry_formatter_preserves_non_dynamic_behavior(signal, expected):
    assert daily_scan.format_signal_email_entry(signal) == expected


class _CaptureSMTP:
    def __init__(self, *args, **kwargs):
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        return None

    def login(self, *_args):
        return None

    def sendmail(self, _sender, _receiver, message):
        self.message = message


def test_olv_email_card_is_concise_and_uses_live_pivot_fields(monkeypatch):
    smtp = _CaptureSMTP()
    monkeypatch.setenv("EMAIL_USER", "sender@example.invalid")
    monkeypatch.setenv("EMAIL_PASS", "not-a-real-password")
    monkeypatch.setattr(daily_scan.smtplib, "SMTP", lambda *_a, **_k: smtp)
    monkeypatch.setitem(
        sys.modules,
        "event_sleeve",
        types.SimpleNamespace(sleeve_status_cards=lambda: []),
    )

    raw_limit = 100.123456 - 0.75 * 2.345678
    signal = _olv_email_signal(
        Entry=100.123456,
        Limit_Price=raw_limit,
        Shares=1_000,
        Notional=100_123.456,
        Risk_Amt=2_930.0,
    )
    assert daily_scan.send_email_summary([signal]) is True
    message = message_from_string(smtp.message)
    part = message.get_payload()[0]
    body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8")

    for label in ("SIGNAL:", "WHY:", "PIVOT:", "ACTION:", "PURPOSE:"):
        assert body.count(label) == 1
    assert "Persistent limit @ $98.36" in body
    assert "LMT $98.36 GTC" in body
    assert "$1.18 below the standard" in body
    assert "$98,360 notional" in body
    assert "+$98,360 Net Exposure" in body
    assert "$100,123 notional" not in body
    assert "+$100,123 Net Exposure" not in body
    assert "$11.12 / 4.74 ATR above" in body
    assert "Shares and risk budget stay unchanged" in body
    assert "the lower limit reduces notional" in body
    assert "First of: +2.5 ATR target, or day 10" in body
    assert "no resting stop" in body

    assert "LEGACY THESIS MUST NOT RENDER" not in body
    assert "LEGACY FILTER MUST NOT RENDER" not in body
    assert "LEGACY EXIT HEADLINE MUST NOT RENDER" not in body
    assert "LEGACY EXIT NOTES MUST NOT RENDER" not in body
    assert "Pivot40" not in body
    assert "above_high_4_5" not in body
    assert "Stop: $" not in body
    assert "Target: $" not in body
    assert "@ $100.12" not in body


def test_non_olv_email_card_retains_legacy_detail(monkeypatch):
    smtp = _CaptureSMTP()
    monkeypatch.setenv("EMAIL_USER", "sender@example.invalid")
    monkeypatch.setenv("EMAIL_PASS", "not-a-real-password")
    monkeypatch.setattr(daily_scan.smtplib, "SMTP", lambda *_a, **_k: smtp)
    monkeypatch.setitem(
        sys.modules,
        "event_sleeve",
        types.SimpleNamespace(sleeve_status_cards=lambda: []),
    )
    signal = _olv_email_signal(
        Strategy_ID="generic-test",
        Strategy_Name="Generic Strategy",
        Entry_Type="Signal Close",
        Entry_Type_Short="MOC",
        Entry_Offset_ATR="",
        Limit_Price=None,
        Setup_Thesis="GENERIC THESIS",
        Live_Filters=[("Generic live filter", "42", False)],
        Exit_Primary="Stop, target, or day 10",
        Exit_Notes="GENERIC EXIT NOTES",
        Sizing_Notes="Special 1.2x sizing",
    )

    assert daily_scan.send_email_summary([signal]) is True
    message = message_from_string(smtp.message)
    part = message.get_payload()[0]
    body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8")

    assert "GENERIC THESIS" in body
    assert "Generic live filter" in body
    assert "42" in body
    assert "[SIZING] Sizing: Special 1.2x sizing" in body
    assert "[RUN] GENERIC EXIT NOTES" in body
    assert "Stop: $97.19 | Target: $105.98" in body
    assert "Signal Close @ $100.12" in body
    assert "$10,012 notional" in body
    assert "+$10,012 Net Exposure" in body
    assert "SIGNAL:" not in body
    assert "PIVOT:" not in body
    assert "PURPOSE:" not in body
