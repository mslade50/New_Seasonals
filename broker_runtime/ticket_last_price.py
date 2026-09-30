"""Read-only last-price lookup for an exact order-ticket instrument.

Called by the existing workbench query subprocess. Never imports an executor,
submits an order, or substitutes a midpoint/close/delayed price for Last.
"""
from datetime import datetime, timezone
import math
import re
import time


def instrument(query):
    context = query.get("context") or {}
    symbol = str(query.get("ticker") or "").upper().strip()
    sec = str(context.get("sec_type") or "STK").upper()
    currency = str(context.get("currency") or "USD").upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9 ._-]{0,23}", symbol):
        raise ValueError("valid symbol required")
    if sec not in {"STK", "FUT", "CASH"} or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("unsupported ticket instrument")
    result = dict(symbol=symbol, sec_type=sec, currency=currency)
    if sec == "FUT":
        exchange = str(context.get("exchange") or "").upper()
        expiry = str(context.get("expiry") or "")
        if exchange not in {"CME", "CBOT", "NYMEX", "COMEX"}:
            raise ValueError("futures venue required")
        if not re.fullmatch(r"\d{6}(\d{2})?", expiry):
            raise ValueError("exact futures contract month required")
        datetime.strptime(expiry, "%Y%m%d" if len(expiry) == 8 else "%Y%m")
        result.update(exchange=exchange, expiry=expiry)
    elif sec == "CASH":
        if not re.fullmatch(r"[A-Z]{3}", symbol) or symbol == currency or "USD" not in {symbol, currency}:
            raise ValueError("FX ticket requires a distinct USD pair")
    elif currency != "USD":
        raise ValueError("stock ticket requires USD")
    return result


def contract_for(ib, wanted):
    from ib_insync import Stock, Future, Forex
    if wanted["sec_type"] == "FUT":
        from futures_front import _spec_row, _matched_details
        row = _spec_row(wanted["symbol"])
        if row and str(row.get("exchange") or "").upper() != wanted["exchange"]:
            raise ValueError("futures venue differs from the broker catalog")
        root = str((row or {}).get("symbol") or wanted["symbol"]).upper()
        cds = ib.reqContractDetails(Future(root, wanted["expiry"], wanted["exchange"], currency=wanted["currency"]))
        cds = _matched_details(cds, wanted["symbol"], row)
        cds = [cd for cd in cds if str(cd.contract.lastTradeDateOrContractMonth).startswith(wanted["expiry"])
               or (len(wanted["expiry"]) == 6 and str(getattr(cd, "contractMonth", "")) == wanted["expiry"])]
        if len(cds) != 1:
            raise ValueError("selected futures contract is not unique")
        contract = cds[0].contract
    else:
        candidate = (Stock(wanted["symbol"], "SMART", wanted["currency"])
                     if wanted["sec_type"] == "STK" else Forex(wanted["symbol"] + wanted["currency"]))
        matches = ib.qualifyContracts(candidate)
        if len(matches) != 1:
            raise ValueError("selected instrument is not unique")
        contract = matches[0]
    if (int(contract.conId or 0) <= 0 or contract.secType != wanted["sec_type"]
            or contract.currency != wanted["currency"]):
        raise ValueError("broker contract identity differs")
    return contract


def read_last(ib, contract, *, seconds=8):
    requested = datetime.now(timezone.utc)
    ib.reqMarketDataType(1)
    ticker = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            ib.sleep(.1)
            stamp = ticker.time
            if (int(ticker.contract.conId or 0) == int(contract.conId)
                    and ticker.marketDataType == 1 and isinstance(stamp, datetime)
                    and stamp.tzinfo is not None and stamp >= requested
                    and math.isfinite(float(ticker.last or 0)) and float(ticker.last or 0) > 0):
                return dict(last=float(ticker.last), asof=stamp.timestamp(),
                            market_data_type=1, con_id=int(contract.conId), source="IBKR Last")
        raise ValueError("live last price unavailable; enter a reference manually")
    finally:
        ib.cancelMktData(contract)


def resolve(query):
    ib = None
    try:
        wanted = instrument(query)
        from ib_insync import IB
        try:
            from eq_order_entry import IB_IP, IB_PORT
        except ImportError:
            IB_IP, IB_PORT = "127.0.0.1", 7496
        ib = IB()
        ib.RequestTimeout = 8
        # The existing agent serializes workbench requests on this client.
        ib.connect(IB_IP, IB_PORT, clientId=133, timeout=8, readonly=True)
        contract = contract_for(ib, wanted)
        return dict(instrument=wanted, **read_last(ib, contract))
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")
    finally:
        if ib is not None:
            ib.disconnect()
