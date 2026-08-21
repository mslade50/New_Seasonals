"""Tests for daily_execution_report enrichment: exit-leg parsing (target /
stop / time stop), orderRef strategy attribution, Trend Sleeve / Discretionary
fallbacks, OPT exclusion, and the ET hour gate helpers.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from daily_execution_report import (
    EDT_CRON,
    EST_CRON,
    ET,
    build_rows,
    cron_should_send,
    enrich_position,
    parse_ib_date,
    parse_ref,
)


def _pos(symbol, qty, sec_type="STK", avg=100.0, last=102.0, upnl=200.0):
    return {
        "symbol": symbol, "sec_type": sec_type, "position": qty,
        "avg_cost": avg, "market_price": last,
        "market_value": (last or 0) * qty, "unrealized_pnl": upnl,
    }


def _leg(symbol, action, order_type, ref=None, lmt=None, aux=None,
         tif="GTC", good_after=None, status="PreSubmitted", sec_type="STK"):
    return {
        "symbol": symbol, "sec_type": sec_type, "action": action,
        "order_type": order_type, "lmt": lmt, "aux": aux, "tif": tif,
        "good_after": good_after, "status": status, "order_ref": ref,
    }


def test_cron_gate_dst_regime():
    from datetime import datetime
    july = datetime(2026, 7, 9, 17, 56, tzinfo=ET)   # EDT, lagged past 4 PM
    jan = datetime(2026, 1, 9, 18, 5, tzinfo=ET)     # EST, lagged past 4 PM
    # The regime decides, not the clock — a lagged run still sends.
    assert cron_should_send(EDT_CRON, july) is True
    assert cron_should_send(EST_CRON, july) is False
    assert cron_should_send(EDT_CRON, jan) is False
    assert cron_should_send(EST_CRON, jan) is True
    # Unknown cron fails open (send) so a workflow edit can't silently
    # drop the email.
    assert cron_should_send("0 12 * * 1-5", july) is True


def test_parse_ref():
    assert parse_ref("AAPL|BUY|OLV|2026-07-01") == ("OLV", "2026-07-01")
    assert parse_ref("AAPL|BUY|Overbot Vol Spike|2026-07-01|near") == \
        ("Overbot Vol Spike", "2026-07-01")
    assert parse_ref(None) == (None, None)
    assert parse_ref("just-a-string") == (None, None)


def test_parse_ib_date():
    assert parse_ib_date("20260715 15:59:00") == "2026-07-15"
    assert parse_ib_date("20260715 09:30:00 US/Eastern") == "2026-07-15"
    assert parse_ib_date(None) is None
    assert parse_ib_date("bogus") is None


def test_long_bracket_enrichment():
    pos = _pos("AAPL", 100, avg=100.0, last=102.0, upnl=200.0)
    ref = "AAPL|BUY|OLV|2026-07-01"
    orders = [
        _leg("AAPL", "SELL", "LMT", ref=ref, lmt=105.0),
        _leg("AAPL", "SELL", "STP", ref=ref, aux=95.0,
             good_after="20260702 09:30:00"),
        _leg("AAPL", "SELL", "MKT", ref=ref, good_after="20260715 15:59:00"),
        # Opposite-side working order on the same symbol must be ignored.
        _leg("AAPL", "BUY", "LMT", ref="AAPL|BUY|Other|2026-07-07", lmt=98.0),
    ]
    r = enrich_position(pos, orders, trend_syms=set())
    assert r["strategy"] == "OLV"
    assert r["targets"] == [105.0]
    assert r["stop"] == 95.0
    assert r["time_stop"] == "2026-07-15"
    assert r["staged"] == "2026-07-01"
    assert abs(r["pnl_pct"] - 2.0) < 1e-9


def test_short_position_uses_buy_legs():
    pos = _pos("SQQQ", -200, avg=20.0, last=19.0, upnl=200.0)
    ref = "SQQQ|SELL|3x ETF Overbot Fade|2026-07-03"
    orders = [
        _leg("SQQQ", "BUY", "LMT", ref=ref, lmt=18.0),
        _leg("SQQQ", "BUY", "MKT", ref=ref, good_after="20260720 15:59:00"),
    ]
    r = enrich_position(pos, orders, trend_syms=set())
    assert r["strategy"] == "3x ETF Overbot Fade"
    assert r["targets"] == [18.0]
    assert r["stop"] is None  # no STP leg -> NA
    assert r["time_stop"] == "2026-07-20"


def test_dead_and_expired_legs_ignored():
    pos = _pos("XOM", 50)
    orders = [
        _leg("XOM", "SELL", "LMT", ref="XOM|BUY|OVS|2026-07-06",
             lmt=110.0, status="Cancelled"),
        _leg("XOM", "SELL", "STP", ref="XOM|BUY|OVS|2026-07-06",
             aux=90.0, status="Filled"),
    ]
    r = enrich_position(pos, orders, trend_syms=set())
    assert r["strategy"] == "Discretionary"
    assert r["targets"] == [] and r["stop"] is None


def test_scaleout_two_targets_one_strategy():
    pos = _pos("USO", 300)
    near = "USO|BUY|Overbot Vol Spike|2026-07-06|near"
    far = "USO|BUY|Overbot Vol Spike|2026-07-06|far"
    orders = [
        _leg("USO", "SELL", "LMT", ref=near, lmt=75.0),
        _leg("USO", "SELL", "LMT", ref=far, lmt=78.0),
        _leg("USO", "SELL", "MKT", ref=near, good_after="20260710 15:59:00"),
        _leg("USO", "SELL", "MKT", ref=far, good_after="20260713 15:59:00"),
    ]
    r = enrich_position(pos, orders, trend_syms=set())
    assert r["strategy"] == "Overbot Vol Spike"
    assert r["targets"] == [75.0, 78.0]
    assert r["time_stop"] == "2026-07-10"  # earliest MKT leg


def test_fallbacks_and_opt_exclusion():
    account = {
        "positions": [
            _pos("GLD", 40),                      # in trend state -> Trend Sleeve
            _pos("NVDA", 10),                     # no legs, no trend -> Discretionary
            _pos("XOM", 3, sec_type="OPT"),       # excluded
        ],
        "orders": [],
    }
    rows, opt_excluded = build_rows(account, trend_syms={"GLD"})
    assert opt_excluded == ["XOM"]
    by_sym = {r["symbol"]: r for r in rows}
    assert set(by_sym) == {"GLD", "NVDA"}
    assert by_sym["GLD"]["strategy"] == "Trend Sleeve"
    assert by_sym["NVDA"]["strategy"] == "Discretionary"


def test_sort_order_strategies_first():
    account = {
        "positions": [_pos("NVDA", 10), _pos("GLD", 40), _pos("AAPL", 100)],
        "orders": [
            _leg("AAPL", "SELL", "MKT", ref="AAPL|BUY|OLV|2026-07-01",
                 good_after="20260715 15:59:00"),
        ],
    }
    rows, _ = build_rows(account, trend_syms={"GLD"})
    assert [r["strategy"] for r in rows] == ["OLV", "Trend Sleeve", "Discretionary"]


def test_event_sleeve_fallback_and_sort():
    # Event orders are parent-only (no exit legs), so attribution must come
    # from the sleeve state, and event rows sort between strategies and trend.
    account = {
        "positions": [_pos("SVXY", 1228), _pos("GLD", 40), _pos("NVDA", 10),
                      _pos("AAPL", 100)],
        "orders": [
            _leg("AAPL", "SELL", "MKT", ref="AAPL|BUY|OLV|2026-07-01",
                 good_after="20260715 15:59:00"),
        ],
    }
    rows, _ = build_rows(account, trend_syms={"GLD"},
                         event_map={"SVXY": "Event Sleeve (V4)"})
    assert [r["strategy"] for r in rows] == [
        "OLV", "Event Sleeve (V4)", "Trend Sleeve", "Discretionary"]


def test_event_symbol_map_from_state():
    from daily_execution_report import event_symbol_map
    state = {"positions": {"V4_POSTOPEX_VOL": {
        "shares": 1228, "entry_date": "2026-08-21", "ref_close": 61.03,
        "exit_on": "2026-08-26", "exit_order_type": "MOC"}}}
    assert event_symbol_map(state) == {"SVXY": "Event Sleeve (V4)"}
    assert event_symbol_map({"positions": {}}) == {}


def test_reconcile_event_shortfall_only():
    from daily_execution_report import reconcile_event
    state = {"positions": {"V4_POSTOPEX_VOL": {
        "shares": 1228, "entry_date": "2020-01-02"}}}
    # Missing shares = the staged order never filled -> alarm.
    issues = reconcile_event(state, [])
    assert len(issues) == 1 and "V4_POSTOPEX_VOL" in issues[0]
    # Exact match and EXCESS (book overlap on the same symbol) stay quiet.
    assert reconcile_event(state, [_pos("SVXY", 1228)]) == []
    assert reconcile_event(state, [_pos("SVXY", 1500)]) == []
    # Short trades alarm on a shortfall in the SHORT direction.
    short_state = {"positions": {"T3_SEP_POSTQUAD_SHORT": {
        "shares": 750, "entry_date": "2020-01-02"}}}
    assert len(reconcile_event(short_state, [])) == 1
    assert reconcile_event(short_state, [_pos("IWM", -750)]) == []
    assert reconcile_event(short_state, [_pos("IWM", -900)]) == []
    assert len(reconcile_event(short_state, [_pos("IWM", -100)])) == 1


def test_reconcile_event_skips_entered_today():
    from daily_execution_report import et_now, reconcile_event
    state = {"positions": {"V4_POSTOPEX_VOL": {
        "shares": 1228, "entry_date": et_now().strftime("%Y-%m-%d")}}}
    # The entry MOC fills at 4:00 and can race the agent's book push.
    assert reconcile_event(state, []) == []
