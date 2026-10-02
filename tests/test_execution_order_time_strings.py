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
