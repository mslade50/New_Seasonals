"""Production guards for OLV's causal 40/40 pivot-entry policy."""

import copy
import os
import sys

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
    OLV_PIVOT_LOW_COL,
    OLV_PIVOT_LOW_DATE_COL,
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
    assert policy["default_offset_atr"] == pytest.approx(0.25)


def test_indicator_cache_contract_requires_pivot_columns():
    assert INDICATOR_CACHE_VERSION == "v2"
    stale = pd.DataFrame({"Close": [100.0]})
    assert not _indicator_cache_has_required_schema(stale)
    fresh = stale.assign(
        **{
            OLV_PIVOT_HIGH_COL: [99.0],
            OLV_PIVOT_HIGH_DATE_COL: [pd.Timestamp("2025-01-01")],
            OLV_PIVOT_LOW_COL: [90.0],
            OLV_PIVOT_LOW_DATE_COL: [pd.Timestamp("2024-12-01")],
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
        policy=disabled,
    )
    assert shadow["proposed_action"] == "skip"
    assert shadow["action"] == "stage"
    assert shadow["offset_atr"] == pytest.approx(0.25)


def _engine_case(distance_atr, policy_enabled=True):
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
            OLV_PIVOT_LOW_COL: [40.0] * n,
            OLV_PIVOT_LOW_DATE_COL: [dates[0] - pd.Timedelta(days=120)] * n,
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
        "Pivot_Rule_Version": "olv_close_pivot_40_v1_20260831",
        "Pivot_Nearest_Type": "High",
        "Pivot_Level": 89.0,
        "Pivot_Date": "2026-05-01",
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
