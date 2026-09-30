"""Read-only quote behavior; no broker connection or order submission."""
from datetime import datetime, timezone
import sys
from types import SimpleNamespace as N

import pytest
from ib_insync import Contract
from broker_runtime import ticket_last_price as quotes
from broker_runtime.prepare_ticket_last_price import prepare, HOOK


@pytest.mark.parametrize("context", [
    {"sec_type":"FUT", "exchange":"CME", "expiry":"202613"},
    {"sec_type":"FUT", "exchange":"CME", "expiry":""},
    {"sec_type":"FUT", "exchange":"SMART", "expiry":"202612"},
    {"sec_type":"OPT"}, {"sec_type":"STK", "currency":"EUR"},
])
def test_invalid_ticket_identity_is_rejected(context):
    with pytest.raises(ValueError): quotes.instrument({"ticker":"MNQ","context":context})


def test_exact_ticket_identity_normalized():
    assert quotes.instrument({"ticker":" mnq ","context":{"sec_type":"FUT","exchange":"CME","expiry":"202612"}}) == {
        "symbol":"MNQ","sec_type":"FUT","currency":"USD","exchange":"CME","expiry":"202612"}


class QuoteIB:
    def __init__(self, contract, last=30844.5, kind=1):
        self.quote = N(contract=contract,last=last,marketDataType=kind,time=None,bid=30844,ask=30845,close=30000)
        self.requests=[]
    def reqMarketDataType(self, kind): self.requests.append(("type",kind))
    def reqMktData(self, contract, fields, **kw):
        self.requests.append(("quote",contract,fields,kw)); return self.quote
    def sleep(self, _): self.quote.time=datetime.now(timezone.utc)
    def cancelMktData(self, contract): self.requests.append(("cancel",contract))


def clock(monkeypatch):
    stamps=iter([0,0,.01])
    monkeypatch.setattr(quotes.time,"monotonic",lambda:next(stamps))


def test_returns_native_last_not_midpoint_or_previous_close(monkeypatch):
    clock(monkeypatch)
    contract=Contract(conId=42,secType="FUT",currency="USD")
    ib=QuoteIB(contract)
    got=quotes.read_last(ib,contract,seconds=.001)
    assert got["last"]==30844.5 and got["con_id"]==42 and got["market_data_type"]==1
    assert ib.requests[1][3]=={"snapshot":False,"regulatorySnapshot":False}
    assert ib.requests[-1]==("cancel",contract)


@pytest.mark.parametrize("last,kind", [(None,1),(float("nan"),1),(0,1),(-1,1),(30000,2),(30000,3),(30000,4)])
def test_unavailable_frozen_and_delayed_last_never_autofill(monkeypatch,last,kind):
    clock(monkeypatch)
    contract=Contract(conId=42,secType="FUT",currency="USD")
    ib=QuoteIB(contract,last,kind)
    with pytest.raises(ValueError,match="last price unavailable"):
        quotes.read_last(ib,contract,seconds=.001)
    assert ib.requests[-1]==("cancel",contract)


def test_different_contract_quote_is_rejected(monkeypatch):
    clock(monkeypatch)
    wanted=Contract(conId=42,secType="FUT",currency="USD")
    ib=QuoteIB(Contract(conId=43,secType="FUT",currency="USD"))
    with pytest.raises(ValueError):quotes.read_last(ib,wanted,seconds=.001)


def test_futures_quote_qualifies_selected_month_and_catalog_class(monkeypatch):
    row={"symbol":"EUR","trading_class":"6E","exchange":"CME"}
    monkeypatch.setitem(sys.modules,"futures_front",N(_spec_row=lambda sym:row,
        _matched_details=lambda cds,sym,spec:[cd for cd in cds if cd.contract.tradingClass==spec["trading_class"]]))
    good=Contract(conId=42,secType="FUT",symbol="EUR",tradingClass="6E",currency="USD",lastTradeDateOrContractMonth="20261214")
    other=Contract(conId=43,secType="FUT",symbol="EUR",tradingClass="M6E",currency="USD",lastTradeDateOrContractMonth="20261214")
    class IB:
        def reqContractDetails(self,c):
            assert c.symbol=="EUR" and c.exchange=="CME"
            return [N(contract=good,contractMonth="202612"),N(contract=other,contractMonth="202612")]
    wanted=quotes.instrument({"ticker":"6E","context":{"sec_type":"FUT","exchange":"CME","expiry":"202612"}})
    assert quotes.contract_for(IB(),wanted).conId==42
    good.lastTradeDateOrContractMonth="20270315"
    with pytest.raises(ValueError):quotes.contract_for(IB(),dict(wanted,expiry="202611"))


def test_read_only_extension_is_scoped_before_option_chain_logic():
    original='def main():\n    mode = str(q.get("mode") or "full").lower()\n    get_chain()\n'
    changed=prepare(original)
    assert changed.replace(HOOK,"")==original
    assert changed.index('if mode == "last_price"')<changed.index('get_chain()')
    assert 'return\n' in HOOK
    with pytest.raises(ValueError):prepare(changed)
