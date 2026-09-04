"""Does intraday historical option data work where daily (EODChart) didn't?"""
from ib_insync import IB, Option

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=135, timeout=8, readonly=True)
try:
    ib.reqMarketDataType(3)
    opt = Option("AAPL", "20260731", 290, "C", "SMART", tradingClass="AAPL")
    ib.qualifyContracts(opt)
    print(f"conId {opt.conId}  exch {opt.exchange}")
    for dur, bar, what in [("2 D", "1 hour", "MIDPOINT"), ("2 D", "1 hour", "TRADES"),
                           ("1 D", "30 mins", "MIDPOINT"), ("2 D", "1 hour", "OPTION_IMPLIED_VOLATILITY")]:
        try:
            b = ib.reqHistoricalData(opt, "", dur, bar, what, True, 1)
            print(f"  {dur}/{bar}/{what}: {len(b)} bars" + (f"  {b[0].date}..{b[-1].date}  last={b[-1].close}" if b else ""))
        except Exception as e:
            print(f"  {dur}/{bar}/{what}: {type(e).__name__} {e}")
finally:
    ib.disconnect()
