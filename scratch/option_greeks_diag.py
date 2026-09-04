"""Why are model greeks missing for some names? Compare reqTickers (snapshot) vs
reqMktData (streaming) for a few near-ATM calls. Read-only."""
import datetime
import sys

from ib_insync import IB, Option, Stock

sym = (sys.argv[1] if len(sys.argv) > 1 else "TSLA").upper()
ib = IB()
ib.connect("127.0.0.1", 7496, clientId=133, timeout=8, readonly=True)
try:
    ib.reqMarketDataType(3)
    stk = Stock(sym, "SMART", "USD"); ib.qualifyContracts(stk)
    [u] = ib.reqTickers(stk); spot = u.marketPrice() or u.close
    print(f"{sym} spot {spot}  (underlying mdType={getattr(u,'marketDataType',None)})")
    params = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    exps = sorted(set().union(*[set(p.expirations) for p in params]))
    strikes = sorted(set().union(*[set(p.strikes) for p in params]))
    tc = params[0].tradingClass
    today = datetime.date.today()
    exp = next(e for e in exps if (datetime.date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days >= 7)
    near = [s for s in strikes if spot * 0.95 <= s <= spot * 1.10][:5]
    opts = [Option(sym, exp, s, "C", "SMART", tradingClass=tc) for s in near]
    ib.qualifyContracts(*opts)

    print("=== reqTickers (snapshot), 2s wait ===")
    tks = ib.reqTickers(*opts); ib.sleep(2)
    for t in tks:
        g = t.modelGreeks
        print(f"  {t.contract.strike}: bid {t.bid} ask {t.ask} delta {g.delta if g else None} iv {g.impliedVol if g else None}")

    print("=== reqMktData (streaming), 6s wait ===")
    sk = [ib.reqMktData(o, "", False, False) for o in opts]
    ib.sleep(6)
    for t in sk:
        g = t.modelGreeks
        print(f"  {t.contract.strike}: bid {t.bid} ask {t.ask} delta {g.delta if g else None} iv {g.impliedVol if g else None}")
    for o in opts:
        ib.cancelMktData(o)
finally:
    ib.disconnect()
