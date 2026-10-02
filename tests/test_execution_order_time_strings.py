"""goodTillDate / goodAfterTime strings the executor sends to IBKR.

IBKR rejected 'YYYYMMDD HH:MM:SS US/Eastern' on CME/CBOT futures (343 'End Time'
on the XK and MNQ GTD parents, 2026-10-02), while stocks accept it. FUT legs now
carry the zone-free UTC form 'YYYYMMDD-HH:MM:SS'; stock strings are unchanged.
"""
import contextlib
import io
import json
import re
from types import SimpleNamespace as N

import pytest

from tests.execution_harness import SimBroker, bind, install_helpers, load_executor

ET_FORM = re.compile(r"^\d{8} \d{2}:\d{2}:\d{2} US/Eastern$")
UTC_FORM = re.compile(r"^\d{8}-\d{2}:\d{2}:\d{2}$")


@pytest.fixture(autouse=True)
def helpers(monkeypatch):
    install_helpers(monkeypatch)
    monkeypatch.delenv("LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS", raising=False)


@pytest.fixture()
def executor():
    return load_executor()


@pytest.mark.parametrize("sec_type", ["STK", None, "", "CASH"])
def test_non_futures_keep_the_eastern_form(executor, sec_type):
    assert executor._execution_deadline("2030-11-05", "15:59:00", sec_type) == "20301105 15:59:00 US/Eastern"
    assert executor._execution_deadline("2030-11-05", "15:59:00") == "20301105 15:59:00 US/Eastern"


@pytest.mark.parametrize("value,clock,expected", [
    ("2030-11-05", "15:59:00", "20301105-20:59:00"),   # EST: UTC-5
    ("2030-07-09", "16:00:00", "20300709-20:00:00"),   # EDT: UTC-4
    ("20301231", "16:00:00", "20301231-21:00:00"),
])
def test_futures_use_the_utc_form(executor, value, clock, expected):
    for sec_type in ("FUT", "fut"):
        out = executor._execution_deadline(value, clock, sec_type)
        assert out == expected and UTC_FORM.match(out)


def test_futures_still_refuse_bad_or_past_dates(executor):
    with pytest.raises(ValueError):
        executor._execution_deadline("2020-01-02", "16:00:00", "FUT")
    with pytest.raises(ValueError):
        executor._execution_deadline("11/05/2030", "16:00:00", "FUT")


class FutBroker(SimBroker):
    def qualifyContracts(self, contract):
        contract.conId = contract.conId or 77
        contract.multiplier = "5"
        return [contract]

    def reqContractDetails(self, contract):
        return [N(contract=contract, minTick=0.25, priceMagnifier=1)]


def run_entry(executor, broker, p):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert executor._do_entry_bracket(broker, p, "primary") == 0
    return json.loads(stream.getvalue())


def test_futures_bracket_sends_utc_gtd_and_time_exit(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "futures_sizing", N(
        get_spec=lambda symbol: N(symbol="MES", exchange="CME", multiplier=5.0, min_tick=0.25,
                                  price_magnifier=1),
        snap_to_tick=lambda price, tick: round(round(price / tick) * tick, 8)))
    broker = FutBroker()
    ex = bind(load_executor(), broker, tmp_path)
    p = dict(_command_id="fut-gtd", _broker_account="PRIMARY", symbol="MES", action="BUY",
             sec_type="FUT", currency="USD", quantity=1, entry=100, stop=90, target=None,
             entry_type="LMT", fut_expiry="203012", exchange="CME", fut_ib_symbol="MES",
             fut_trading_class="MES", fut_multiplier=5, fut_min_tick=0.25,
             expiry="2030-11-05", time_stop="2030-11-12")
    result = run_entry(ex, broker, p)
    assert result["ok"], result
    orders = [row[1] for row in broker.sent]
    parent = orders[0]
    assert parent.tif == "GTD" and parent.goodTillDate == "20301105-21:00:00"
    timed = [o for o in orders[1:] if getattr(o, "goodAfterTime", "")]
    assert [o.goodAfterTime for o in timed] == ["20301112-20:59:00"]
    assert all("US/" not in str(getattr(o, f, "")) for o in orders
               for f in ("goodTillDate", "goodAfterTime"))


# --- Futures whose 15:59 ET falls in a closed window (2026-10-02) ------------
# CBOT grains and CME livestock are shut at 15:59 ET; a TIME leg waking there
# would fill at the evening reopen. contract_reference.json carries the close
# of the session before that window (from IBKR tradingHours) and the executor
# fires at close minus one minute on the exit date. Missing field = 15:59 ET.
GRAIN = ("13:20", "US/Central")
LIVESTOCK = ("13:05", "US/Central")
REFERENCE = [
    {"symbol": "MES", "alias": "MES", "trading_class": "MES"},
    {"symbol": "CL", "alias": "CL", "trading_class": "CL"},
    {"symbol": "GC", "alias": "GC", "trading_class": "GC"},
    {"symbol": "ZS", "alias": "ZS", "trading_class": "ZS",
     "time_exit_session_close": "13:20", "time_exit_session_tz": "US/Central"},
    {"symbol": "YK", "alias": "XK", "trading_class": "XK",
     "time_exit_session_close": "13:20", "time_exit_session_tz": "US/Central"},
    {"symbol": "LE", "alias": "LE", "trading_class": "LE",
     "time_exit_session_close": "13:05", "time_exit_session_tz": "US/Central"},
]


@pytest.mark.parametrize("session,value,expected", [
    (GRAIN, "2030-11-05", "20301105-19:19:00"),      # CST: 13:19 CT = 14:19 ET
    (GRAIN, "2030-07-09", "20300709-18:19:00"),      # CDT: 13:19 CT = 14:19 ET
    (LIVESTOCK, "2030-11-05", "20301105-19:04:00"),
    (LIVESTOCK, "2030-07-09", "20300709-18:04:00"),
    (GRAIN, "2030-11-01", "20301101-18:19:00"),      # Friday before the Nov 3 change
    (GRAIN, "2030-11-04", "20301104-19:19:00"),      # Monday after it
    (GRAIN, "2031-03-07", "20310307-19:19:00"),      # Friday before the Mar 9 change
    (GRAIN, "2031-03-10", "20310310-18:19:00"),      # Monday after it
    (GRAIN, "2026-11-03", "20261103-19:19:00"),      # the live XK order's exit date
])
def test_closed_window_futures_exit_one_minute_before_the_session_close(executor, session, value, expected):
    assert executor._time_exit_deadline(value, "15:59:00", "FUT", session) == expected


@pytest.mark.parametrize("value", ["2030-11-05", "2030-07-09"])
def test_open_at_1559_and_missing_field_futures_are_byte_identical(executor, value):
    today = executor._execution_deadline(value, "15:59:00", "FUT")
    assert executor._time_exit_deadline(value, "15:59:00", "FUT", None) == today
    # A recorded close later than 15:59 ET can never push the exit later.
    assert executor._time_exit_deadline(value, "15:59:00", "FUT", ("16:00", "US/Central")) == today


def test_stock_cash_and_open_clock_ignore_the_session(executor):
    assert executor._time_exit_deadline("2030-11-05", "15:59:00", "STK", GRAIN) == "20301105 15:59:00 US/Eastern"
    assert executor._time_exit_deadline("2030-11-05", "15:59:00", "CASH", GRAIN) == "20301105 15:59:00 US/Eastern"
    assert executor._time_exit_deadline("2030-11-05", "09:30:00", "FUT", GRAIN) == "20301105-14:30:00"


@pytest.mark.parametrize("session", [("1:20pm", "US/Central"), ("13:20", "Mars/Olympus"), ("13:20", None)])
def test_unreadable_session_fails_closed(executor, session):
    with pytest.raises(ValueError):
        executor._time_exit_deadline("2030-11-05", "15:59:00", "FUT", session)


def _with_reference(executor, tmp_path, rows=REFERENCE):
    (tmp_path / "contract_reference.json").write_text(json.dumps(rows))
    executor._THIS_DIR = str(tmp_path)
    return executor


@pytest.mark.parametrize("symbol,trading_class,expected", [
    ("YK", "XK", GRAIN), ("ZS", "ZS", GRAIN), ("LE", "LE", LIVESTOCK), ("XK", "", GRAIN),
    ("MES", "MES", None), ("CL", "CL", None), ("GC", "GC", None), ("ZZ", "ZZ", None),
])
def test_session_lookup_by_class_then_symbol_or_alias(executor, tmp_path, symbol, trading_class, expected):
    ex = _with_reference(executor, tmp_path)
    assert ex._fut_time_exit_session(N(symbol=symbol, tradingClass=trading_class)) == expected


def test_missing_reference_table_keeps_today(executor, tmp_path):
    executor._THIS_DIR = str(tmp_path)
    assert executor._fut_time_exit_session(N(symbol="YK", tradingClass="XK")) is None


def _fut_entry(symbol, ib_symbol, trading_class):
    return dict(_command_id="fut-closed-window", _broker_account="PRIMARY", symbol=symbol, action="BUY",
                sec_type="FUT", currency="USD", quantity=1, entry=100, stop=90, target=None,
                entry_type="LMT", fut_expiry="203012", exchange="CME", fut_ib_symbol=ib_symbol,
                fut_trading_class=trading_class, fut_multiplier=5, fut_min_tick=0.25,
                time_stop="2030-11-12")


@pytest.mark.parametrize("symbol,ib_symbol,trading_class,expected", [
    ("XK", "YK", "XK", "20301112-19:19:00"),
    ("LE", "LE", "LE", "20301112-19:04:00"),
    ("MES", "MES", "MES", "20301112-20:59:00"),
])
def test_entry_bracket_time_child_uses_the_session_close(tmp_path, monkeypatch, symbol, ib_symbol, trading_class, expected):
    import sys
    monkeypatch.setitem(sys.modules, "futures_sizing", N(
        get_spec=lambda s: N(symbol=ib_symbol, exchange="CME", multiplier=5.0, min_tick=0.25, price_magnifier=1),
        snap_to_tick=lambda price, tick: round(round(price / tick) * tick, 8)))
    broker = FutBroker()
    ex = _with_reference(bind(load_executor(), broker, tmp_path), tmp_path)
    result = run_entry(ex, broker, _fut_entry(symbol, ib_symbol, trading_class))
    assert result["ok"], result
    timed = [row[1] for row in broker.sent if row[1].orderType == "MKT" and row[1].parentId]
    assert [o.goodAfterTime for o in timed] == [expected]


@pytest.mark.parametrize("symbol,trading_class,expected", [
    ("YK", "XK", "20301112-19:19:00"),
    ("ZS", "ZS", "20301112-19:19:00"),
    ("MES", "MES", "20301112-20:59:00"),
])
def test_exit_attach_time_leg_uses_the_session_close(tmp_path, symbol, trading_class, expected):
    from ib_insync import Future, Position
    held = Position("PRIMARY", Future(symbol, "203012", "", conId=42, multiplier="5", currency="USD",
                                      tradingClass=trading_class), 2, 100)
    broker = SimBroker([held])
    ex = _with_reference(bind(load_executor(), broker, tmp_path), tmp_path)
    p = dict(_command_id="attach-closed-window", _broker_account="PRIMARY", con_id=42, symbol=symbol,
             sec_type="FUT", time_stop="2030-11-12")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert ex._do_exit_attach(broker, p, "primary") == 0
    result = json.loads(stream.getvalue())
    assert result["ok"], result
    assert [row[1].goodAfterTime for row in broker.sent] == [expected]
